import hashlib
from functools import cached_property

class Task:
    def __init__(self, root=None, splitter= None):
        self.root = root
        self.splitter = splitter
        self.dataset = self._build_dataset(root)
        pass

    def split(self):
        raise NotImplementedError
    
    def _build_dataset(self, root):
        # check if dataset exists and load

        # else call self.build_dataset()
        return self.build_dataset(root)

    def build_dataset(self, root):
        raise NotImplementedError


    @cached_property
    def task_hash(self):
        h = hashlib.new('sha256')
        rna_ids = ""
        for rna in self.dataset.to_list():
            rna_ids += rna['rna_name']
        h.update(rna_ids.encode("utf-8"))
        return h.hexdigest()

    def __eq__(self, other):
        return self.task_hash() == other.task_hash()

class ResidueClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    pass

class RNAClassificationTask(Task):
    pass

