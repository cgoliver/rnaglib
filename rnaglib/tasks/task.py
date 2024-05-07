import hashlib
from functools import cached_property

class Task:
    def __init__(self, root=None, splitter=None):
        self.root = root
        if splitter is None:
            self.splitter = self.default_splitter()
        else:
            self.splitter = splitter


        self.dataset = self._build_dataset(root)
        self.split()

        pass

    def split(self):
        """ Sets train, val, and test indices"""
        train_ind, val_ind, test_ind = self.splitter(self.dataset)
        self.train_ind = train_ind
        self.val_ind = val_ind
        self.test_ind = test_ind
        pass
    
    def _build_dataset(self, root):
        # check if dataset exists and load

        # else call self.build_dataset()
        return self.build_dataset(root)

    def build_dataset(self, root):
        raise NotImplementedError


    @cached_property
    def task_hash(self):
        """ Task hash is a hash of all RNA ids and node IDs in the dataset"""
        h = hashlib.new('sha256')
        for rna in self.dataset.to_list():
            h.update(rna['rna_name'].encode("utf-8"))
            for nt in rna['rna'].nodes():
                h.update(nt.encode("utf-8"))
        [h.update(str(i).encode("utf-8")) for i in self.train_ind]
        [h.update(str(i).encode("utf-8")) for i in self.val_ind]
        return h.hexdigest()

    def __eq__(self, other):
        return self.task_hash() == other.task_hash()

class ResidueClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    pass

class RNAClassificationTask(Task):
    pass

