import unittest
import tempfile

from rnaglib.tasks import RNAFamilyTask
from rnaglib.data_loading import RNADataset
from rnaglib.splitters import RandomSplitter

class SplitterTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = RNADataset()
        pass

    def check_split_sizes(self, train, val, test):
        assert sum(map(len, [train, val, test])) == len(self.dataset)
        pass

    def test_RandomSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            splitter = RandomSplitter()
            train, val, test = splitter(self.dataset)
            pass
