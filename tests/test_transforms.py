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

    def test_RNAFMTransform(self):
        tr = RNAFMTransform()
        tr(self.dataset[0])
        pass

    def test_pre_transform(self):
        """ Add rnafm embeddings during dataset construction from database,
        then look up the stored attribute at getitem time.
        """
        tr = RNAFMTransform()
        feat = FeaturesComputer(nt_features=['nt_code', tr.name], transforms=tr)
        dataset = RNADataset(debug=True,
                             features_computer=feat,
                             pre_transforms=tr,
                             representations=GraphRepresentation(framework='pyg'),
                             )

        assert dataset[0]['graph'].x is not None

    def test_post_transform(self):
        """ Apply transform during getitem call.
        """
        tr = RNAFMTransform()
        feat = FeaturesComputer(nt_features=['nt_code', tr.name], transforms=tr)
        dataset = RNADataset(debug=True,
                             features_computer=feat,
                             transforms=tr,
                             representations=GraphRepresentation(framework='pyg'),
                             )
        assert dataset[0]['graph'].x is not None
        pass


    pass

