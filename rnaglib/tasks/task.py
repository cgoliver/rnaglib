import os
import hashlib
import json
from functools import cached_property

from rnaglib.data_loading import RNADataset


class Task:
    def __init__(self, root, recompute=False, splitter=None):
        self.root = root
        self.recompute = recompute

        if splitter is None:
            self.splitter = self.default_splitter()
        else:
            self.splitter = splitter

        self.dataset = self._build_dataset(root)
        self.split()

        if not os.path.exists(root) or recompute:
            try:
                os.mkdir(root)
            except FileExistsError:
                pass
            self.write()
        pass

    def write(self):
        print(">>> Saving dataset.")
        try:
            os.mkdir(os.path.join(self.root, 'graphs'))
        except FileExistsError:
            pass
        print("saving")
        self.dataset.save(os.path.join(self.root, 'graphs'))

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
        pass

    def _build_dataset(self, root):
        # check if dataset exists and load
        if os.path.exists(os.path.join(self.root, 'graphs')) and not self.recompute:
            return RNADataset(saved_dataset=os.path.join(self.root, 'graphs'))
        return self.build_dataset(root)

    def build_dataset(self, root):
        raise NotImplementedError

    @cached_property
    def task_id(self):
        """ Task hash is a hash of all RNA ids and node IDs in the dataset"""
        h = hashlib.new('sha256')
        for rna in self.dataset.rnas:
            h.update(rna.graph['pdbid'][0].encode("utf-8"))
            for nt in sorted(rna.nodes()):
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
                for i, idx in enumerate(self.test_ind)]
        return {'mcc': sum(true) / len(true)}

    pass


class RNAClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, graph_level_attribute, test_predictions):
        from sklearn.metrics import matthews_corrcoef
        true = [self.dataset[idx]['graph'][graph_level_attribute] for idx in self.test_ind]
        mcc_scores = [matthews_corrcoef([true[i]], [test_predictions[i]]) for i in range(len(self.test_ind))]
        return {'mcc': sum(mcc_scores) / len(mcc_scores)}

    pass
