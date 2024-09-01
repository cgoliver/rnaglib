import unittest
import tempfile

from rnaglib.data_loading import RNADataset
from rnaglib.data_loading import FeaturesComputer
from rnaglib.representations import GraphRepresentation
from rnaglib.transforms import RNAFMTransform

class TransformsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = RNADataset(debug=True)
        pass

    def test_RNAFMTransform(self):
        tr = RNAFMTransform()
        tr(self.dataset[0])
        pass

    def test_post_transform(self):
        tr = RNAFMTransform()
        feat = FeaturesComputer(post_transform=tr)
        dataset = RNADataset(debug=True,
                             features_computer=feat,
                             representations=GraphRepresentation(framework='pyg'),
                             )
        assert dataset[0]['graph'].x is not None
        pass
    pass

