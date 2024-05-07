from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import RandomSplitter

from rnaglib.data_loading import DEFAULT_INDEX

class BindingSiteDetection(ResidueClassificationTask):
    def __init__(self, root=None, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass
    pass

    def default_splitter(self):
        return RandomSplitter()
        pass

    def build_dataset(self, root):
        dataset = RNADataset()
        return dataset
