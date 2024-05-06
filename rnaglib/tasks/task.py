class Task:
    def __init__(self, root=None, splitter= None):
        self.root = root
        self.splitter = splitter
        self.dataset = self._build_dataset(root)
        pass

    def split(self):
        raise NotImplementedError
    
    def _build_dataset(self):
        raise NotImplementedError

class ResidueClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    pass

class RNAClassificationTask(Task):
    pass

