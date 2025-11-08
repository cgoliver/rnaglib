import tempfile
import unittest
import subprocess
import shutil

from rnaglib.dataset import RNADataset
from rnaglib.dataset_transforms import RandomSplitter
from rnaglib.dataset_transforms import ClusterSplitter
from rnaglib.dataset_transforms import CDHitComputer, StructureDistanceComputer


def has_cdhit():
    """Check if cd-hit is available."""
    return shutil.which("cd-hit") is not None


def has_usalign():
    """Check if USalign is available."""
    return shutil.which("USalign") is not None


class SplitterTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = RNADataset(debug=True, get_pdbs=True)

    def check_splits(self, train, val, test):
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_RandomSplitter(self):
        splitter = RandomSplitter()
        train, val, test = splitter(self.dataset)
        self.check_splits(train, val, test)

    @unittest.skipUnless(has_usalign(), "USalign not available")
    def test_USalignSplitter(self):
        distance_computer = StructureDistanceComputer(name="USalign", structures_path=self.dataset.structures_path)
        splitter = ClusterSplitter(similarity_threshold=0.7, distance_name="USalign", debug=True)
        self.dataset = distance_computer(self.dataset)
        train, val, test = splitter(self.dataset)
        self.check_splits(train, val, test)

    @unittest.skipUnless(has_cdhit(), "cd-hit not available")
    def test_CDHitSplitter(self):
        distance_computer = CDHitComputer(similarity_threshold=0.7)
        splitter = ClusterSplitter(similarity_threshold=0.7, distance_name="cd_hit", debug=True)
        self.dataset = distance_computer(self.dataset)
        train, val, test = splitter(self.dataset)
        self.check_splits(train, val, test)
