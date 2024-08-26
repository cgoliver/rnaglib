import unittest
import tempfile
from pathlib import Path

from rnaglib.tasks import RNAFamilyTask
from rnaglib.data_loading import RNADataset
from rnaglib.splitters import RandomSplitter
from rnaglib.splitters import RNAalignSplitter
from rnaglib.splitters import CDHitSplitter
from rnaglib.utils import available_pdbids

class SplitterTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # pick small RNAs
        # TODO add rna size to index
        def size_filter(rna):
            return len(rna.nodes()) < 30

        dataset = RNADataset.from_database(rna_filter=size_filter)
        # TODO allow slicing on datasets
        self.dataset = [rna for i, rna in enumerate(dataset) if i < 50]
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

    def test_RNAalignSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
                splitter = RNAalignSplitter(structures_dir=Path.home() / ".rnaglib" / "structures" / "all",
                                            similarity_threshold=.3)
                train, val, test = splitter(self.dataset)
                self.check_splits(train, val, test)

        pass

    def test_CDHitSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
                splitter = CDHitSplitter(similarity_threshold=.5)
                train, val, test = splitter(self.dataset)
                self.check_splits(train, val, test)

        pass
