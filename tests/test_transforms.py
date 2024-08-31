import unittest
import tempfile

from rnaglib.data_loading import RNADataset
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
    pass

