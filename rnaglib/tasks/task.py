import os
import hashlib
import json
from functools import cached_property

from rnaglib.data_loading import RNADataset

class Task:
    def __init__(self, root, splitter=None):
        self.root = root
        if splitter is None:
            self.splitter = self.default_splitter()
        else:
            self.splitter = splitter

        self.dataset = self._build_dataset(root)
        self.split()

        if not os.path.exists(root):
            os.mkdir(root)
            self.write()
        pass

    def write(self):
        with open(os.path.join(self.root, 'dataset.txt'), 'w') as ds:
            [ds.write("\n".join([rna['rna_name'] for rna in self.dataset.to_list()]))]
        with open(os.path.join(self.root, 'train_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.train_ind]
        with open(os.path.join(self.root, 'val_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.val_ind]
        with open(os.path.join(self.root, 'test_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.test_ind]
        with open(os.path.join(self.root, "task_id.txt"), "w") as tid:
                  tid.write(self.task_id)
        pass

    def split(self):
        """ Sets train, val, and test indices"""
        if not os.path.exists(os.path.join(self.root, "train_idx.txt")):
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
        pass
    
    def _build_dataset(self, root):
        # check if dataset exists and load
        if os.path.exists(os.path.join(self.root, "dataset.txt")):
            print(">>> Loading dataset")
            with open(os.path.join(root, "dataset.txt"), 'r') as ds:
                all_graphs = [g.strip() for g in ds.readlines()]
            return RNADataset(all_graphs=all_graphs,
                              nt_targets=[self.target_var],
                              nt_features=[self.input_var],
                              )
        else:
            print(">>> Building dataset")
            return self.build_dataset()

    def build_dataset(self, root):
        raise NotImplementedError

    @cached_property
    def task_id(self):
        """ Task hash is a hash of all RNA ids and node IDs in the dataset"""
        h = hashlib.new('sha256')
        for rna in self.dataset.to_list():
            h.update(rna['rna_name'].encode("utf-8"))
            for nt in sorted(rna['rna'].nodes()):
                h.update(nt.encode("utf-8"))
        [h.update(str(i).encode("utf-8")) for i in self.train_ind]
        [h.update(str(i).encode("utf-8")) for i in self.val_ind]
        [h.update(str(i).encode("utf-8")) for i in self.test_ind]
        return h.hexdigest()

    def __eq__(self, other):
        return self.task_id == other.task_id

class ResidueClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, test_predictions):
        from sklearn.metrics import matthews_corrcoef
        true = [matthews_corrcoef(test_predictions[i], self.dataset[idx]['graph']['y']) \
                for i,idx in enumerate(self.test_ind)]
        return {'mcc': sum(true) / len(true)}
    pass

class RNAClassificationTask(Task):
    pass

