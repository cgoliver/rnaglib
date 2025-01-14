import unittest
import tempfile

import networkx as nx

from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import GraphRepresentation
from rnaglib.transforms import RNAFMTransform
from rnaglib.transforms import RfamTransform
from rnaglib.transforms import SecondaryStructureTransform
from rnaglib.transforms import Compose
from rnaglib.transforms import SizeFilter
from rnaglib.transforms import ChainSplitTransform
from rnaglib.transforms import ConnectedComponentPartition


class TransformsTest(unittest.TestCase):

    def check_ndata(self, g, attribute: str):
        _, ndata = next(iter(g.nodes(data=True)))
        assert attribute in ndata

    def check_gdata(self, g, attribute: str):
        assert attribute in g.graph

    @classmethod
    def setUpClass(self):
        self.dataset = RNADataset(debug=True, in_memory=True)

    def test_RfamTransform(self):
        tr = RfamTransform()
        tr(self.dataset[0])
        tr(self.dataset)

    def test_RfamTransform_parallel(self):
        tr = RfamTransform(parallel=True)
        tr(self.dataset[0])
        tr(self.dataset)

    def test_RNAFMTransform(self):
        tr = RNAFMTransform()
        tr(self.dataset[0])
        list(tr(self.dataset))
        pass

    def test_SecondaryStructureTransform(self):
        tr = SecondaryStructureTransform(self.dataset.structures_path)
        tr(self.dataset[0])

    """
    def test_RNAFMTransform_chunk(self):
        tr = RNAFMTransform()
        d = RNADataset(redundancy="all", in_memory=False)
        big_g = d.get_pdbid("4x66")
        print(len(big_g["rna"].nodes()))
        tr(big_g)
    """

    def test_filter(self):
        f = SizeFilter(max_size=50)
        new_dset = list(f(self.dataset))
        assert len(new_dset) < len(self.dataset)

    def test_filter_parallel(self):
        f = SizeFilter(max_size=50, parallel=True)
        new_dset = list(f(self.dataset))
        assert len(new_dset) < len(self.dataset)

    def test_partition(self):
        t = ChainSplitTransform()
        new_data = list(t(self.dataset))
        assert len(new_data) > len(self.dataset)

    def test_connected_component_partition(self):
        t = ConnectedComponentPartition()
        new_data = list(t(self.dataset))
        assert len(new_data) > len(self.dataset)

    def test_simple_compose(self):
        g = self.dataset[0]
        tr_1 = RNAFMTransform()
        tr_2 = RfamTransform()
        t = Compose([tr_1, tr_2])
        t(self.dataset[0])
        self.check_gdata(g["rna"], "rfam")
        self.check_ndata(g["rna"], "rnafm")

    pass
