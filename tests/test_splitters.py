import unittest
import tempfile
from pathlib import Path

from rnaglib.tasks import RNAFamilyTask
from rnaglib.data_loading import RNADataset
from rnaglib.splitters import RandomSplitter
from rnaglib.splitters import RNAalignSplitter
from rnaglib.utils import available_pdbids

class SplitterTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pdbids =  ['3skt', '5u3g', '5j02',
                   '2yie', '2fcy', '3gx3',
                   '4nyb', '1hr2', '4mgm',
                   '1y90', '2quw', '4meg',
                   '4lvx', '4rge', '4pcj',
                   '3c44', '5o69', '2lwk',
                   '2g5k', '5fj1', '5d5l'
                   ]
 
                 
        self.dataset = RNADataset(all_rnas=pdbids, redundancy='all')
        pass

    def check_split_sizes(self, train, val, test):
        assert sum(map(len, [train, val, test])) == len(self.dataset)
        pass

    def test_RandomSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            splitter = RandomSplitter()
            train, val, test = splitter(self.dataset)
            self.check_split_sizes(train, val, test)
            pass

    def test_RNAalignSplitter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
                splitter = RNAalignSplitter(structures_dir=Path.home() / ".rnaglib" / "structures" / "all",
                                            similarity_threshold=.1)
                train, val, test = splitter(self.dataset)
                self.check_split_sizes(train, val, test)

        pass
