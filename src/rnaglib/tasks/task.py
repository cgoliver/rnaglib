import os
import hashlib
from pathlib import Path
import json
from functools import cached_property
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
from typing import Union, Optional

from rnaglib.data_loading import RNADataset, Collater
from rnaglib.transforms import FeaturesComputer
from rnaglib.splitters import Splitter, RandomSplitter
from rnaglib.utils import DummyResidueModel, DummyGraphModel


class Task:
    """Abstract class for a benchmarking task using the rnaglib datasets.
    This class handles the logic for building the underlying dataset which is held in an
    rnaglib.data_loading.RNADataset
    object. Once the dataset is created, the splitter is invoked to create the train/val/test indices.
    Tasks also define an evaluate() function to yield appropriate model performance metrics.

    :param root: path to a folder where the task information will be stored for fast loading.
    :param recompute: whether to recompute the task info from scratch or use what is stored in root.
    :param splitter: rnaglib.splitters.Splitter object that handles splitting of data into train/val/test indices.
    If None uses task's default_splitter() attribute.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        recompute: bool = False,
        splitter: Splitter = None,
        debug: bool = False,
        save: bool = True,
        in_memory: bool = True,
    ):
        self.root = root
        self.dataset_path = os.path.join(self.root, "dataset")
        self.recompute = recompute
        self.debug = debug
        self.save = save
        self.in_memory = in_memory
        self.metadata = self.init_metadata()

        # Load or create dataset
        if not os.path.exists(self.dataset_path) or recompute:
            print(">>> Creating task dataset from scratch...")
            self.dataset = self.process()
        else:
            self.dataset, self.metadata, (self.train_ind, self.val_ind, self.test_ind) = self.load()

        # Set splitter after dataset is available
        self.splitter = self.default_splitter if splitter is None else splitter

        # Split dataset if it wasn't loaded from file
        if not hasattr(self, "train_ind"):
            self.split(self.dataset)

        self.dataset.features_computer = self.get_task_vars()

        if self.save:
            self.write()

        # compute metadata
        self.describe()

    def process(self) -> RNADataset:
        """Tasks must implement this method. Executing the method should result in a list of ``.json`` files
        saved in ``{root}/dataset``. All the RNA graphs should contain all the annotations needed to run the task (e.g. node/edge attributes)
        """
        raise NotImplementedError

    def init_metadata(self) -> dict:
        """Optionally adds some key/value pairs to self.metadata."""
        return {}

    def get_task_vars(self) -> FeaturesComputer:
        """Define a FeaturesComputer object to set which input and output variables will be used in the task."""
        return FeaturesComputer()

    @property
    def default_splitter(self):
        return RandomSplitter()

    def split(self, dataset):
        """Calls the splitter and returns train, val, test splits."""
        splits = self.splitter(dataset)
        self.train_ind, self.val_ind, self.test_ind = splits
        return splits

    def set_datasets(self, recompute=True):
        """Sets the train, val and test datasets
        Call this each time you modify ``self.dataset``."""

        if not hasattr(self, "train_ind") or recompute:
            self.train_ind, self.val_ind, self.test_ind = self.split(self.dataset)
        self.train_dataset = self.dataset.subset(self.train_ind)
        self.val_dataset = self.dataset.subset(self.val_ind)
        self.test_dataset = self.dataset.subset(self.test_ind)

    def set_loaders(self, recompute=True, **dataloader_kwargs):
        """Sets the dataloader properties.
        Call this each time you modify ``self.dataset``."""

        self.set_datasets(recompute=recompute)

        # If no collater is provided we need one
        if dataloader_kwargs is None:
            dataloader_kwargs = {"collate_fn": Collater(self.train_dataset)}
        if "collate_fn" not in dataloader_kwargs:
            collater = Collater(self.train_dataset)
            dataloader_kwargs["collate_fn"] = collater

        # Now build the loaders
        self.train_dataloader = DataLoader(dataset=self.train_dataset, **dataloader_kwargs)
        dataloader_kwargs["shuffle"] = False
        self.val_dataloader = DataLoader(dataset=self.val_dataset, **dataloader_kwargs)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, **dataloader_kwargs)

    def get_split_datasets(self, recompute=True):
        # If datasets were not already computed or if we want to recompute them to account
        # for changes in the global dataset
        if recompute or "train_dataset" not in self.__dict__:
            print(">>> Splitting the dataset...")
            self.set_datasets(recompute=recompute)
            print(">>> Done")
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_split_loaders(self, recompute=True, **dataloader_kwargs):
        # If dataloaders were not already precomputed or if we want to recompute them to account
        # for changes in the global dataset
        if recompute or "train_dataloader" not in self.__dict__:
            self.set_loaders(recompute=recompute, **dataloader_kwargs)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def evaluate(self, model, loader) -> dict:
        raise NotImplementedError

    @cached_property
    def task_id(self):
        """Task hash is a hash of all RNA ids and node IDs in the dataset"""
        h = hashlib.new("sha256")
        if not self.in_memory:
            raise ValueError("task id is only available (and tractable) for small, in-memory datasets")
        for rna in self.dataset.rnas:
            h.update(rna.name.encode("utf-8"))
            for nt in sorted(rna.nodes()):
                h.update(nt.encode("utf-8"))
        [h.update(str(i).encode("utf-8")) for i in self.train_ind]
        [h.update(str(i).encode("utf-8")) for i in self.val_ind]
        [h.update(str(i).encode("utf-8")) for i in self.test_ind]
        return h.hexdigest()

    def write(self):
        """Save task data and splits to root. Creates a folder in ``root`` called
        ``'graphs'`` which stores the RNAs that form the dataset, and three `.txt` files (`'{train, val, test}_idx.txt'`,
        one for each split with a list of indices.
        """
        if not os.path.exists(self.dataset_path) or self.recompute:
            print(">>> Saving dataset.")
            self.dataset.save(self.dataset_path, recompute=self.recompute)
        with open(Path(self.root) / "train_idx.txt", "w") as idx:
            [idx.write(str(ind) + "\n") for ind in self.train_ind]
        with open(Path(self.root) / "val_idx.txt", "w") as idx:
            [idx.write(str(ind) + "\n") for ind in self.val_ind]
        with open(Path(self.root) / "test_idx.txt", "w") as idx:
            [idx.write(str(ind) + "\n") for ind in self.test_ind]
        with open(Path(self.root) / "metadata.json", "w") as meta:
            json.dump(self.metadata, meta, indent=4)
        # task id is only available (and tractable) for small, in-memory datasets
        if self.in_memory:
            with open(Path(self.root) / "task_id.txt", "w") as tid:
                tid.write(self.task_id)
        print(">>> Done")

    def load(self):
        """Load dataset and splits from disk."""
        # load splits
        print(">>> Loading precomputed dataset...")
        train_ind = [int(ind) for ind in open(os.path.join(self.root, "train_idx.txt"), "r").readlines()]
        val_ind = [int(ind) for ind in open(os.path.join(self.root, "val_idx.txt"), "r").readlines()]
        test_ind = [int(ind) for ind in open(os.path.join(self.root, "test_idx.txt"), "r").readlines()]
        dataset = RNADataset(dataset_path=self.dataset_path, in_memory=self.in_memory)

        with open(Path(self.root) / "metadata.json", "r") as meta:
            metadata = json.load(meta)

        return dataset, metadata, (train_ind, val_ind, test_ind)

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def describe(self, recompute=False):
        """
        Get description of task dataset, including dimensions needed for model initialization
        and other relevant statistics. Prints the description and returns it as a dict.
        Returns:
            dict: Contains dataset information and model dimensions
        """
        if not recompute and "description" in self.metadata:
            return self.metadata["description"]

        print(">>> Computing description of task...")
        self.get_split_loaders(recompute=False)

        # Get dimensions from first graph
        first_item = self.dataset[0]
        if "graph" in first_item:
            compute_num_edge_attributes = True
        else:
            compute_num_edge_attributes = False
        first_node_map = {n: i for i, n in enumerate(sorted(first_item["rna"].nodes()))}
        first_features_dict = self.dataset.features_computer(first_item)
        first_features_array = first_features_dict["nt_features"][next(iter(first_node_map.keys()))]
        num_node_features = first_features_array.shape[0]

        # Dynamic class counting
        class_counts = {}
        classes = set()
        if compute_num_edge_attributes:
            unique_edge_attrs = set()

        # Collect statistics from dataset
        import tqdm

        for item in tqdm.tqdm(self.dataset):
            if compute_num_edge_attributes:
                graph = item["graph"]
                unique_edge_attrs.update(graph.edge_attr.tolist())
            node_map = {n: i for i, n in enumerate(sorted(item["rna"].nodes()))}
            features_dict = self.dataset.features_computer(item)
            if "nt_targets" in features_dict:
                list_y = [features_dict["nt_targets"][n] for n in node_map.keys()]
                # In the case of single target, pytorch CE loss expects shape (n,) and not (n,1)
                # For multi-target cases, we stack to get (n,d)
                if len(list_y[0]) == 1:
                    y = torch.cat(list_y)
                else:
                    y = torch.stack(list_y)
            if "rna_targets" in features_dict:
                y = features_dict["rna_targets"].clone().detach()

            graph_classes = y.unique().tolist()
            classes.update(graph_classes)

            # Count classes in this graph
            for cls in graph_classes:
                cls_int = int(cls)
                if cls_int not in class_counts:
                    class_counts[cls_int] = 0
                class_counts[cls_int] += torch.sum(y == cls).item()

        info = {
            "num_node_features": num_node_features,
            "num_classes": len(classes),
            "dataset_size": len(self.dataset),
            "class_distribution": class_counts,
        }
        if compute_num_edge_attributes:
            info["num_edge_attributes"] = len(unique_edge_attrs)

        # Print description
        print("Dataset Description:")
        print(f"Number of node features: {info['num_node_features']}")
        print(f"Number of classes: {info['num_classes']}")
        if compute_num_edge_attributes:
            print(f"Number of edge attributes: {info['num_edge_attributes']}")
        print(f"Dataset size: {info['dataset_size']} graphs")
        print("\nClass distribution:")
        for cls in sorted(class_counts.keys()):
            print(f"Class {cls}: {class_counts[cls]} nodes")

        self.metadata["description"] = info
        if self.save:
            with open(Path(self.root) / "metadata.json", "w") as meta:
                json.dump(self.metadata, meta, indent=4)
        return info


class ClassificationTask(Task):
    def __init__(self, graph_level=False, num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = self.metadata["description"]["num_classes"] if num_classes is None else num_classes
        self.graph_level = graph_level

    @property
    def dummy_model(self) -> torch.nn:
        if self.graph_level:
            return DummyGraphModel(num_classes=self.num_classes)
        return DummyResidueModel(num_classes=self.num_classes)

    def compute_metrics(self, all_preds, all_probs, all_labels):
        if self.graph_level:
            metrics = {
                "accuracy": accuracy_score(all_labels, all_preds),
                "f1": f1_score(all_labels, all_preds, average="binary" if self.num_classes == 2 else "macro"),
                "auc": roc_auc_score(
                    all_labels, all_probs, average="binary" if self.num_classes == 2 else "macro", multi_class="ovo"
                ),
                "mcc": matthews_corrcoef(all_labels, all_preds),
            }
        else:
            # Here we have a list of preds [(n1,), (n2,)...] for each residue in each RNA
            # Either compute the overall flattened results, or aggregate by system
            metrics = []
            for pred, prob, label in zip(all_preds, all_probs, all_labels):
                # Can't compute metrics over just one class
                if len(np.unique(label)) == 1:
                    continue
                metrics.append(
                    [
                        accuracy_score(label, pred),
                        f1_score(label, pred),
                        roc_auc_score(label, prob),
                        matthews_corrcoef(label, pred),
                    ]
                )
            metrics = np.array(metrics)
            mean_metrics = np.mean(metrics, axis=0)
            metrics = {
                "accuracy": mean_metrics[0],
                "f1": mean_metrics[1],
                "auc": mean_metrics[2],
                "mcc": mean_metrics[3],
            }

            # Get the flattened result
            all_preds = np.concatenate(all_preds)
            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            metrics_global = {
                "global_accuracy": accuracy_score(all_labels, all_preds),
                "global_f1": f1_score(all_labels, all_preds),
                "global_auc": roc_auc_score(all_labels, all_probs),
                "global_mcc": matthews_corrcoef(all_labels, all_preds),
            }
            metrics.update(metrics_global)
        return metrics


class ResidueClassificationTask(ClassificationTask):
    def __init__(self, **kwargs):
        super().__init__(graph_level=False, **kwargs)


class RNAClassificationTask(ClassificationTask):
    def __init__(self, **kwargs):
        super().__init__(graph_level=True, **kwargs)


class ResidueClassificationTask(ClassificationTask):
    def __init__(self, **kwargs):
        super().__init__(graph_level=False, **kwargs)
