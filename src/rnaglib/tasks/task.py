import os
import hashlib
from pathlib import Path
import json
from functools import cached_property
from typing import Union, Optional

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score


from rnaglib.data_loading import RNADataset, Collater
from rnaglib.transforms import FeaturesComputer
from rnaglib.splitters import Splitter, RandomSplitter
from rnaglib.utils import DummyResidueModel


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
    ):
        self.root = root
        self.dataset_path = os.path.join(self.root, "dataset")
        self.recompute = recompute
        self.debug = debug
        self.metadata = self.init_metadata()

        # Load or create dataset
        if not os.path.exists(self.dataset_path) or recompute:
            print("Creating task dataset from scratch...")
            self.dataset = self.process()
        else:
            (
                self.dataset,
                self.metadata,
                (self.train_ind, self.val_ind, self.test_ind),
            ) = self.load()

        # Set splitter after dataset is available
        self.splitter = self.default_splitter if splitter is None else splitter

        # Split dataset if it wasn't loaded from file
        if not hasattr(self, "train_ind"):
            self.train_ind, self.val_ind, self.test_ind = self.split(self.dataset)

        self.dataset.features_computer = self.get_task_vars()

        if save:
            self.write()

    def process(self) -> RNADataset:
        """Tasks must implement this method. Executing the method should result in a list of ``.json`` files
        saved in ``{root}/dataset``. All the RNA graphs should contain all the annotations needed to run the task (e.g. node/edge attributes)
        """
        raise NotImplementedError

    def init_metadata(self) -> dict:
        """Optionally adds some key/value pairs to self.metadata."""
        return {}

    @property
    def get_task_vars(self) -> FeaturesComputer:
        """Define a FeaturesComputer object to set which input and output variables will be used in the task."""
        return FeaturesComputer()

    @property
    def default_splitter(self):
        return RandomSplitter()

    def split(self, dataset):
        """Calls the splitter and returns train, val, test splits."""
        return self.splitter(dataset)

    def set_datasets(self):
        """Sets the train, val and test datasets
        Call this each time you modify ``self.dataset``."""
        self.train_ind, self.val_ind, self.test_ind = self.split(self.dataset)
        self.train_dataset = self.dataset.subset(self.train_ind)
        self.val_dataset = self.dataset.subset(self.val_ind)
        self.test_dataset = self.dataset.subset(self.test_ind)

    def set_loaders(self, **dataloader_kwargs):
        """Sets the dataloader properties.
        Call this each time you modify ``self.dataset``."""

        self.set_datasets()

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
            self.set_datasets()
        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_split_loaders(self, recompute=True, **dataloader_kwargs):
        # If dataloaders were not already precomputed or if we want to recompute them to account
        # for changes in the global dataset
        if recompute or "train_dataloader" not in self.__dict__:
            self.set_loaders(**dataloader_kwargs)
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def evaluate(self, model, loader) -> dict:
        raise NotImplementedError

    @cached_property
    def task_id(self):
        """Task hash is a hash of all RNA ids and node IDs in the dataset"""
        h = hashlib.new("sha256")
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
        with open(Path(self.root) / "task_id.txt", "w") as tid:
            tid.write(self.task_id)
        with open(Path(self.root) / "metadata.json", "w") as meta:
            json.dump(self.metadata, meta, indent=4)
        print(">>> Done")

    def load(self):
        """Load dataset and splits from disk."""
        # load splits
        print(">>> Loading splits...")
        train_ind = [int(ind) for ind in open(os.path.join(self.root, "train_idx.txt"), "r").readlines()]
        val_ind = [int(ind) for ind in open(os.path.join(self.root, "val_idx.txt"), "r").readlines()]
        test_ind = [int(ind) for ind in open(os.path.join(self.root, "test_idx.txt"), "r").readlines()]

        dataset = RNADataset(dataset_path=self.dataset_path)

        with open(Path(self.root) / "metadata.json", "r") as meta:
            metadata = json.load(meta)

        return dataset, metadata, (train_ind, val_ind, test_ind)

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def describe(self):
        """
        Get description of task dataset, including dimensions needed for model initialization
        and other relevant statistics. Prints the description and returns it as a dict.
        Returns:
            dict: Contains dataset information and model dimensions
        """
        # Get dimensions from first graph
        first_graph = self.dataset[0]["graph"]
        num_node_features = first_graph.x.shape[1]

        # Dynamic class counting
        class_counts = {}
        unique_edge_attrs = set()
        classes = set()

        # Collect statistics from dataset
        for i in range(len(self.dataset)):
            graph = self.dataset[i]["graph"]
            unique_edge_attrs.update(graph.edge_attr.tolist())
            graph_classes = graph.y.unique().tolist()
            classes.update(graph_classes)

            # Count classes in this graph
            for cls in graph_classes:
                cls_int = int(cls)
                if cls_int not in class_counts:
                    class_counts[cls_int] = 0
                class_counts[cls_int] += sum(graph.y == cls).item()

        info = {
            "num_node_features": num_node_features,
            "num_classes": len(classes),
            "num_edge_attributes": len(unique_edge_attrs),
            "dataset_size": len(self.dataset),
            "class_distribution": class_counts,
        }

        # Print description
        print("Dataset Description:")
        print(f"Number of node features: {info['num_node_features']}")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Number of edge attributes: {info['num_edge_attributes']}")
        print(f"Dataset size: {info['dataset_size']} graphs")
        print("\nClass distribution:")
        for cls in sorted(class_counts.keys()):
            print(f"Class {cls}: {class_counts[cls]} nodes")

        return info


class ResidueClassificationTask(Task):
    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    @property
    def dummy_model(self) -> torch.nn:
        return DummyResidueModel()

    def evaluate(self, model: torch.nn, loader) -> dict:
        """
        Evaluate model performance on a dataset
        NOTE: This only works for binary classification at the moment.

        Args:
            model: The model to evaluate
            loader: Data loader to use
            device: Device to run evaluation on
            criterion: Loss function

        Returns:
            dict: Dictionary containing metrics including loss if criterion provided
        """
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"]
                graph = graph.to(model.device)
                probs = model(graph)

                if model.criterion is not None:
                    loss = model.criterion(probs, graph.y.long())
                    total_loss += loss.item()

                # Take probabilities for positive class only
                preds = (probs > 0.5).float()

                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(graph.y.long().cpu().tolist())

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds),
            "auc": roc_auc_score(all_labels, all_probs),
            "mcc": matthews_corrcoef(all_labels, all_preds),
        }

        if model.criterion is not None:
            metrics["loss"] = total_loss / len(loader)

        return metrics


class RNAClassificationTask(Task):
    pass
