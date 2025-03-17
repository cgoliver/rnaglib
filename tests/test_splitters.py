import unittest
import tempfile
from pathlib import Path

from rnaglib.data_loading import RNADataset
from rnaglib.dataset_transforms import RandomSplitter
from rnaglib.dataset_transforms import ClusterSplitter
from rnaglib.dataset_transforms import CDHitComputer, StructureDistanceComputer
from rnaglib.transforms import FeaturesComputer
from rnaglib.utils import available_pdbids


class SplitterTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = RNADataset(debug=True, get_pdbs=True, overwrite=False)
        pass

    def check_splits(self, train, val, test):
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        # can't do these until we can hash a data point
        """
        assert len(set(train) & set(test)) == 0
        assert len(set(train) & set(val)) == 0
        assert len(set(val) & set(test)) == 0
        """

    def test_RandomSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            splitter = RandomSplitter()
            train, val, test = splitter(self.dataset)
            self.check_splits(train, val, test)
            pass

    def test_USalignSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distance_computer = StructureDistanceComputer(name="USalign",structures_path=self.dataset.structures_path)
            splitter = ClusterSplitter(similarity_threshold=0.7,
                                       distance_name="USalign", debug=True)
            dataset = self.dataset
            dataset.features_computer = FeaturesComputer(nt_targets="is_modified")
            dataset = distance_computer(dataset)
            train, val, test = splitter(dataset)
            self.check_splits(train, val, test)
        pass

    def test_CDHitSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            distance_computer = CDHitComputer(similarity_threshold=0.7)
            splitter = ClusterSplitter(similarity_threshold=0.7,
                                       distance_name="cd_hit", debug=True)
            dataset = self.dataset
            dataset.features_computer = FeaturesComputer(nt_targets="is_modified")
            dataset = distance_computer(dataset)
            train, val, test = splitter(dataset)
            self.check_splits(train, val, test)
        pass
