import os
import hashlib
import json
from functools import cached_property
from typing import Union, Optional

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score

from rnaglib.data_loading import Collater
from rnaglib.splitters import RandomSplitter


class Task:
    """ Abstract class for a benchmarking task using the rnaglib datasets.
    This class handles the logic for building the underlying dataset which is held in an
    rnaglib.data_loading.RNADataset
    object. Once the dataset is created, the splitter is invoked to create the train/val/test indices.
    Tasks also define an evaluate() function to yield appropriate model performance metrics.

    :param root: path to a folder where the task information will be stored for fast loading.
    :param recompute: whether to recompute the task info from scratch or use what is stored in root.
    :param splitter: rnaglib.splitters.Splitter object that handles splitting of data into train/val/test indices.
    If None uses task's default_splitter() attribute.
    """

    def __init__(self, root, recompute=False, splitter=None, debug: bool = False):
        self.root = root
        self.dataset_path = os.path.join(self.root, 'dataset')
        self.recompute = recompute
        self.debug = debug

        self.dataset = self.build_dataset(root)

        if splitter is None:
            self.splitter = self.default_splitter
        else:
            self.splitter = splitter
        self.split()

        os.makedirs(root, exist_ok=True)
        if not os.path.exists(root) or recompute:
            self.write()

    def build_dataset(self, root):
        raise NotImplementedError

    @property
    def default_splitter(self):
        return RandomSplitter()

    def split(self):
        """ Sets train, val, and test indices as attributes of the task. Can be accessed
        as ``self.train_ind``, etc. Will load splits if they are saved in `root` otherwise,
        recomputes from scratch by invoking ``self.splitter()``.
        """
        if not os.path.exists(os.path.join(self.root, "train_idx.txt")) or self.recompute:
            print(">>> Computing splits...")
            train_ind, val_ind, test_ind = self.splitter(self.dataset)
        else:
            print(">>> Loading splits...")
            train_ind = [int(ind) for ind in open(os.path.join(self.root, "train_idx.txt"), 'r').readlines()]
            val_ind = [int(ind) for ind in open(os.path.join(self.root, "val_idx.txt"), 'r').readlines()]
            test_ind = [int(ind) for ind in open(os.path.join(self.root, "test_idx.txt"), 'r').readlines()]
        self.train_ind = train_ind
        self.val_ind = val_ind
        self.test_ind = test_ind
        return train_ind, val_ind, test_ind

    def get_split_datasets(self):
        train_ind, val_ind, test_ind = self.split()
        train_set = self.dataset.subset(train_ind)
        val_set = self.dataset.subset(val_ind)
        test_set = self.dataset.subset(test_ind)
        self.train_dataset = train_set
        self.val_dataset = val_set
        self.test_dataset = test_set
        return train_set, val_set, test_set

    def get_split_loaders(self, **dataloader_kwargs):
        # If datasets were not already precomputed
        if 'train_dataset' not in self.__dict__:
            self.get_split_datasets()

        # If no collater is provided we need one
        if dataloader_kwargs is None:
            dataloader_kwargs = {'collate_fn': Collater(self.train_dataset)}
        if 'collate_fn' not in dataloader_kwargs:
            collater = Collater(self.train_dataset)
            dataloader_kwargs['collate_fn'] = collater

        # Now build the loaders
        train_loader = DataLoader(dataset=self.train_dataset, **dataloader_kwargs)
        dataloader_kwargs['shuffle'] = False
        val_loader = DataLoader(dataset=self.val_dataset, **dataloader_kwargs)
        test_loader = DataLoader(dataset=self.test_dataset, **dataloader_kwargs)
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.test_dataloader = test_loader
        return train_loader, val_loader, test_loader

    def evaluate(self, model, test_loader, criterion, device):
        raise NotImplementedError

    @cached_property
    def task_id(self):
        """ Task hash is a hash of all RNA ids and node IDs in the dataset"""
        h = hashlib.new('sha256')
        for rna in self.dataset.rnas:
            h.update(rna.name.encode("utf-8"))
            for nt in sorted(rna.nodes()):
                h.update(nt.encode("utf-8"))
        [h.update(str(i).encode("utf-8")) for i in self.train_ind]
        [h.update(str(i).encode("utf-8")) for i in self.val_ind]
        [h.update(str(i).encode("utf-8")) for i in self.test_ind]
        return h.hexdigest()

    def write(self):
        """ Save task data and splits to root. Creates a folder in ``root`` called
        ``'graphs'`` which stores the RNAs that form the dataset, and three `.txt` files (`'{train, val, test}_idx.txt'`,
        one for each split with a list of indices.
        """
        if not os.path.exists(self.dataset_path) or self.recompute:
            print(">>> Saving dataset.")
            self.dataset.save(self.dataset_path, recompute=self.recompute)
        with open(os.path.join(self.root, 'train_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.train_ind]
        with open(os.path.join(self.root, 'val_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.val_ind]
        with open(os.path.join(self.root, 'test_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.test_ind]
        with open(os.path.join(self.root, "task_id.txt"), "w") as tid:
            tid.write(self.task_id)
        print(">>> Done")

    def __eq__(self, other):
        return self.task_id == other.task_id

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class ResidueClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, loader, criterion, device):
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                graph = batch['graph']
                graph = graph.to(device)
                out = model(graph)
                loss = criterion(out, torch.flatten(graph.y).long())
                total_loss += loss.item()

                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(graph.cpu().y.tolist())

        avg_loss = total_loss / len(loader)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        mcc = matthews_corrcoef(all_labels, all_preds)

        # print(f'Accuracy: {accuracy:.4f}')
        # print(f'F1 Score: {f1:.4f}')
        # print(f'AUC: {auc:.4f}')
        # print(f'MCC: {mcc:.4f}')

        return accuracy, f1, auc, avg_loss, mcc


class RNAClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, test_loader, criterion, device):
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        test_loss = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                test_loss += loss.item()

                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()
        avg_loss = test_loss / len(test_loader)

        # Calculate MCC
        mcc = matthews_corrcoef(all_labels, all_preds)

        # Calculate F1 score
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f'Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Matthews Correlation Coefficient: {mcc:.4f}')
        print(f'F1 Score: {f1:.4f}')

        return avg_loss, accuracy, mcc, f1


