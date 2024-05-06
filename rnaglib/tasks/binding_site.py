from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask

from rnaglib.data_loading import DEFAULT_INDEX

class BindingSiteDetection(ResidueClassificationTask):
    def __init__(self, root=None, **kwargs):
        super().__init__(root=root, **kwargs)
        pass
    pass

    def build_dataset(self, root):
        dataset = RNADataset()
        for rna in dataset:
            print(rna['rna'])
        pass
