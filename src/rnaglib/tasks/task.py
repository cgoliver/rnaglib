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
        self.splitter = self.default_splitter if splitter is None else splitter

        self.metadata = self.init_metadata()

        # create or load dataset
        if not os.path.exists(self.dataset_path) or recompute:
            print("Creating task dataset from scratch...")
            dataset = self.process()
            train_ind, val_ind, test_ind = self.split(dataset)
        else:
            dataset, metadata, (train_ind, val_ind, test_ind) = self.load()
            self.metadata = metadata

        self.dataset = dataset
        self.dataset.features_computer = self.get_task_vars()

        self.train_ind = train_ind
        self.val_ind = val_ind
        self.test_ind = test_ind

        self.set_loaders()

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

    def set_loaders(self):
        """Sets the dataset and loader properties.
        Call this each time you modify ``self.dataset``."""

        self.train_dataset, self.val_dataset, self.test_dataset = (
            self.get_split_datasets()
        )

        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.get_split_loaders()
        )

    def get_split_datasets(self):
        train_set = self.dataset.subset(self.train_ind)
        val_set = self.dataset.subset(self.val_ind)
        test_set = self.dataset.subset(self.test_ind)
        return train_set, val_set, test_set

    def get_split_loaders(self, **dataloader_kwargs):
        # If datasets were not already precomputed
        if "train_dataset" not in self.__dict__:
            self.get_split_datasets()

        # If no collater is provided we need one
        if dataloader_kwargs is None:
            dataloader_kwargs = {"collate_fn": Collater(self.train_dataset)}
        if "collate_fn" not in dataloader_kwargs:
            collater = Collater(self.train_dataset)
            dataloader_kwargs["collate_fn"] = collater

        # Now build the loaders
        train_loader = DataLoader(dataset=self.train_dataset, **dataloader_kwargs)
        dataloader_kwargs["shuffle"] = False
        val_loader = DataLoader(dataset=self.val_dataset, **dataloader_kwargs)
        test_loader = DataLoader(dataset=self.test_dataset, **dataloader_kwargs)
        return train_loader, val_loader, test_loader

    def evaluate(self, model) -> dict:
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
        train_ind = [
            int(ind)
            for ind in open(os.path.join(self.root, "train_idx.txt"), "r").readlines()
        ]
        val_ind = [
            int(ind)
            for ind in open(os.path.join(self.root, "val_idx.txt"), "r").readlines()
        ]
        test_ind = [
            int(ind)
            for ind in open(os.path.join(self.root, "test_idx.txt"), "r").readlines()
        ]

        dataset = RNADataset(dataset_path=self.dataset_path)

        with open(Path(self.root) / "metadata.json", "r") as meta:
            metadata = json.load(meta)

        return dataset, metadata, (train_ind, val_ind, test_ind)

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ResidueClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def dummy_model(self) -> torch.nn:
        return DummyResidueModel()

    def evaluate(self, model: torch.nn, device: str = "cpu") -> dict:
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_dataloader:
                graph = batch["graph"]
                graph = graph.to(device)
                out = model(graph)

                preds = out > 0.5
                all_probs.extend(out.cpu().flatten().tolist())
                all_preds.extend(preds.cpu().flatten().tolist())
                all_labels.extend(graph.cpu().y.flatten().tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)

        return {"accuracy": accuracy, "mcc": mcc, "f1": f1}


class RNAClassificationTask(Task):
    pass
