"""Abstract task classes"""

from collections.abc import Sequence
from functools import cached_property
import hashlib
import json
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
import torch
from torch.utils.data import DataLoader
import tqdm

from rnaglib.data_loading import Collater, RNADataset
from rnaglib.transforms import FeaturesComputer, SizeFilter
from rnaglib.utils import DummyGraphModel, DummyResidueModel, dump_json, tonumpy
from rnaglib.dataset_transforms import RandomSplitter, Splitter, RedundancyRemover
from rnaglib.dataset_transforms import StructureDistanceComputer, CDHitComputer


class Task:
    """Abstract class for a benchmarking task using the rnaglib datasets.
    This class handles the logic for building the underlying dataset which is held in an
    rnaglib.data_loading.RNADataset
    object. Once the dataset is created, the splitter is invoked to create the train/val/test indices.
    Tasks also define an evaluate() function to yield appropriate model performance metrics.

    :param root: path to a folder where the task information will be stored for fast loading.
    :param recompute: whether to recompute the task info from scratch or use what is stored in root.
    :param splitter: rnaglib.dataset_transforms.Splitter object that handles splitting of data into train/val/test indices.
    If None uses task's default_splitter() attribute.
    """

    def __init__(
            self,
            root: str | os.PathLike,
            recompute: bool = False,
            splitter: Splitter = None,
            debug: bool = False,
            save: bool = True,
            in_memory: bool = False,
            size_thresholds: Sequence = None,
            multi_label=False,
    ):
        self.root = root
        self.dataset_path = os.path.join(self.root, "dataset")
        self.recompute = recompute
        self.debug = debug
        self.save = save
        self.in_memory = in_memory
        self.size_thresholds = size_thresholds
        self.multi_label = multi_label
        self.metadata = self.init_metadata()

        # Load or create dataset
        if not os.path.exists(self.dataset_path) or recompute:
            print(">>> Creating task dataset from scratch...")
            # instantiate the Size filter if required
            if self.size_thresholds is not None:
                self.size_filter = SizeFilter(size_thresholds[0], size_thresholds[1])
            self.dataset = self.process()
            self.post_process()
        else:
            self.load()

        self.dataset.features_computer = self.get_task_vars()

        # Set splitter after dataset is available
        # Split dataset if it wasn't loaded from file
        self.splitter = self.default_splitter if splitter is None else splitter
        if not hasattr(self, "train_ind"):
            self.split(self.dataset)

        if self.save:
            self.write()

        # compute metadata
        self.describe()

    def process(self) -> RNADataset:
        """Tasks must implement this method.

        Executing the method should result in a list of ``.json`` files
        saved in ``{root}/dataset``. All the RNA graphs should contain all the annotations needed to run the task
        (e.g. node/edge attributes)
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
        """The splitter used if no other splitter is specified."""
        return RandomSplitter()

    def post_process(self):
        """
        The most common post_processing steps to remove redundancy.

        Other tasks should implement their own if this is not the desired default behavior
        """
        cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
        cd_hit_rr = RedundancyRemover(distance_name="cd_hit", threshold=0.9)
        self.dataset = cd_hit_computer(self.dataset)
        self.dataset = cd_hit_rr(self.dataset)

        us_align_computer = StructureDistanceComputer(name="USalign")
        us_align_rr = RedundancyRemover(distance_name="USalign", threshold=0.8)
        self.dataset = us_align_computer(self.dataset)
        self.dataset = us_align_rr(self.dataset)
        self.dataset.save_distances()

    def split(self, dataset):
        """Calls the splitter and returns train, val, test splits."""
        splits = self.splitter(dataset)
        self.train_ind, self.val_ind, self.test_ind = splits
        return splits

    def set_datasets(self, recompute=True):
        """Sets the train, val and test datasets
        Call this each time you modify ``self.dataset``.
        """
        if not hasattr(self, "train_ind") or recompute:
            self.train_ind, self.val_ind, self.test_ind = self.split(self.dataset)
        self.train_dataset = self.dataset.subset(self.train_ind)
        self.val_dataset = self.dataset.subset(self.val_ind)
        self.test_dataset = self.dataset.subset(self.test_ind)

    def set_loaders(self, recompute=True, **dataloader_kwargs):
        """Sets the dataloader properties.
        Call this each time you modify ``self.dataset``.
        """
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
        """Save task data and splits to root.

        Creates a folder in ``root`` called ``'graphs'``
        which stores the RNAs that form the dataset, and three `.txt` files (`'{train, val, test}_idx.txt'`,
        one for each split with a list of indices.
        """
        if not os.path.exists(self.dataset_path) or not os.listdir(self.dataset_path) or self.recompute:
            print(">>> Saving dataset.")
            self.dataset.save(self.dataset_path, recompute=self.recompute)
        if hasattr(self, "train_ind"):
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
        self.dataset = RNADataset(
            dataset_path=self.dataset_path,
            in_memory=self.in_memory,
            debug=self.debug,
            recompute_mapping=self.recompute)

        with Path.open(Path(self.root) / "metadata.json") as meta:
            self.metadata = json.load(meta)
        if (
                os.path.exists(os.path.join(self.root, "train_idx.txt")) and
                os.path.exists(os.path.join(self.root, "val_idx.txt")) and
                os.path.exists(os.path.join(self.root, "test_idx.txt"))
        ):
            self.train_ind = [int(ind) for ind in open(os.path.join(self.root, "train_idx.txt")).readlines()]
            self.val_ind = [int(ind) for ind in open(os.path.join(self.root, "val_idx.txt")).readlines()]
            self.test_ind = [int(ind) for ind in open(os.path.join(self.root, "test_idx.txt")).readlines()]

        return self.dataset, self.metadata, (self.train_ind, self.val_ind, self.test_ind)

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def describe(self, recompute=False):
        """Get description of task dataset.

        Including dimensions needed for model initialization
        and other relevant statistics. Prints the description and returns it as a dict.

        Returns:
            dict: Contains dataset information and model dimensions
        """
        if not recompute and "description" in self.metadata:
            info = self.metadata["description"]
        else:
            print(">>> Computing description of task...")
            self.get_split_loaders(recompute=False)

            # Get dimensions from first graph
            first_item = self.dataset[0]
            compute_num_edge_attributes = "graph" in first_item

            first_node_map = {n: i for i, n in enumerate(sorted(first_item["rna"].nodes()))}
            first_features_dict = self.dataset.features_computer(first_item)
            first_features_array = first_features_dict["nt_features"][next(iter(first_node_map.keys()))]
            num_node_features = first_features_array.shape[0]

            # Dynamic class counting
            class_counts = {}
            classes = set()
            unique_edge_attrs = set()  # only used with graphs

            # Collect statistics from dataset
            for item in tqdm.tqdm(self.dataset):
                if compute_num_edge_attributes:
                    graph = item["graph"]
                    unique_edge_attrs.update(graph.edge_attr.tolist())
                node_map = {n: i for i, n in enumerate(sorted(item["rna"].nodes()))}
                features_dict = self.dataset.features_computer(item)
                if "nt_targets" in features_dict:
                    list_y = [features_dict["nt_targets"][n] for n in node_map]
                    # In the case of single target, pytorch CE loss expects shape (n,) and not (n,1)
                    # For multi-target cases, we stack to get (n,d)
                    if len(list_y[0]) == 1:
                        y = torch.cat(list_y)
                    else:
                        y = torch.stack(list_y)
                if "rna_targets" in features_dict:
                    y = features_dict["rna_targets"].clone().detach()

                # In the multi_label case, a full binary matrix is better supported by both pytorch and sklearn
                if not self.multi_label:
                    graph_classes = y.unique().tolist()
                else:
                    graph_classes = torch.where(y > 0)[-1].unique().tolist()
                classes.update(graph_classes)

                # Count classes in this graph
                for cls in graph_classes:
                    cls_int = int(cls)
                    if cls_int not in class_counts:
                        class_counts[cls_int] = 0
                    if not self.multi_label:
                        class_counts[cls_int] += torch.sum(y == cls).item()
                    else:
                        class_counts[cls_int] += torch.sum(torch.where(y > 0)[-1] == cls).item()

            info = {
                "num_node_features": num_node_features,
                "num_classes": len(classes),
                "dataset_size": len(self.dataset),
                "class_distribution": class_counts,
            }
            if compute_num_edge_attributes:
                info["num_edge_attributes"] = len(unique_edge_attrs)
            self.metadata["description"] = info
            if self.save:
                with open(Path(self.root) / "metadata.json", "w") as meta:
                    json.dump(self.metadata, meta, indent=4)

        # Print description
        print("Dataset Description:")
        for k, v in info.items():
            if k != "class_distribution":
                print(k, " : ", v)
            else:
                print("Class distribution:")
                for cls in sorted(v.keys()):
                    print(f"\tClass {cls}: {v[cls]} {'nodes'}")
        print()
        return info

    def create_dataset_from_list(self, rnas):
        """Computes an RNADataset object from the"""
        if self.in_memory:
            dataset = RNADataset(rnas=rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=rnas)
        return dataset

    def add_rna_to_building_list(self, all_rnas, rna):
        if self.in_memory:
            all_rnas.append(rna)
        else:
            all_rnas.append(rna.name)
            dump_json(os.path.join(self.dataset_path, f"{rna.name}.json"), rna)

    def compute_distances(self):
        self.dataset = self.dataset.similarity_matrix_computer.compute_distances(self.dataset)

    def remove_redundancy(self):
        self.dataset = self.dataset.similarity_matrix_computer.remove_redundancy(self.dataset)


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

    def dummy_inference(self):
        all_probs = []
        all_preds = []
        all_labels = []
        dummy_model = self.dummy_model
        with torch.no_grad():
            for batch in self.test_dataloader:
                graph = batch["graph"]
                out = dummy_model(graph)
                labels = graph.y

                # get preds and probas + cast to numpy
                if self.num_classes == 2:
                    probs = torch.sigmoid(out.flatten())
                    preds = (probs > 0.5).float()
                else:
                    probs = torch.softmax(out, dim=1)
                    preds = probs.argmax(dim=1)
                probs = tonumpy(probs)
                preds = tonumpy(preds)
                labels = tonumpy(labels)

                # split predictions per RNA if residue level
                if not self.graph_level:
                    cumulative_sizes = tuple(tonumpy(graph.ptr))
                    probs = [
                        probs[start:end]
                        for start, end in zip(
                            cumulative_sizes[:-1],
                            cumulative_sizes[1:],
                            strict=False,
                        )
                    ]
                    preds = [
                        preds[start:end]
                        for start, end in zip(
                            cumulative_sizes[:-1],
                            cumulative_sizes[1:],
                            strict=False,
                        )
                    ]
                    labels = [
                        labels[start:end]
                        for start, end in zip(
                            cumulative_sizes[:-1],
                            cumulative_sizes[1:],
                            strict=False,
                        )
                    ]
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        if self.graph_level:
            all_probs = np.stack(all_probs)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
        return 0, all_preds, all_probs, all_labels

    def compute_one_metric(self, preds, probs, labels):
        one_metric = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(
                labels,
                preds,
                average="binary" if self.num_classes == 2 else "macro",
            ),
        }
        if not self.multi_label:
            one_metric["mcc"] = matthews_corrcoef(labels, preds)
        try:
            one_metric["auc"] = roc_auc_score(
                labels,
                probs,
                average=None if self.num_classes == 2 else "macro",
                multi_class="ovo",
            )
        except:
            return one_metric
        return one_metric

    def compute_metrics(self, all_preds, all_probs, all_labels):
        if self.graph_level:
            return self.compute_one_metric(all_preds, all_probs, all_labels)
        # Here we have a list of preds [(n1,), (n2,)...] for each residue in each RNA
        # Either compute the overall flattened results, or aggregate by system
        sorted_keys = []
        metrics = []
        for pred, prob, label in zip(all_preds, all_probs, all_labels, strict=False):
            # Can't compute metrics over just one class
            if len(np.unique(label)) == 1:
                continue
            one_metric = self.compute_one_metric(pred, prob, label)
            metrics.append([v for k, v in sorted(one_metric.items())])
            sorted_keys = sorted(one_metric.keys())
        metrics = np.array(metrics)
        mean_metrics = np.mean(metrics, axis=0)
        metrics = {k: v for k, v in zip(sorted_keys, mean_metrics, strict=False)}

        # Get the flattened result, renamed to include "global"
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        global_metrics = self.compute_one_metric(all_preds, all_probs, all_labels)
        metrics_global = {f"global_{k}": v for k, v in global_metrics.items()}
        metrics.update(metrics_global)
        return metrics


class ResidueClassificationTask(ClassificationTask):
    def __init__(self, **kwargs):
        super().__init__(graph_level=False, **kwargs)


class RNAClassificationTask(ClassificationTask):
    def __init__(self, **kwargs):
        super().__init__(graph_level=True, **kwargs)
