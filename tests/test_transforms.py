import unittest
import tempfile

import networkx as nx

from rnaglib.data_loading import RNADataset
from rnaglib.data_loading import FeaturesComputer
from rnaglib.representations import GraphRepresentation
from rnaglib.transforms import RNAFMTransform
from rnaglib.transforms import RfamTransform
from rnaglib.transforms import Compose
from rnaglib.transforms import SizeFilter

class TransformsTest(unittest.TestCase):

    def check_ndata(self, g, attribute: str):
        _, ndata = next(iter(g.nodes(data=True)))
        assert attribute in ndata

    def check_gdata(self, g, attribute: str):
        assert attribute in g.graph

    @classmethod
    def setUpClass(self):
        self.dataset = RNADataset(debug=True)

    def test_RNAFMTransform(self):
        tr = RNAFMTransform()
        tr(self.dataset[0])
        tr(self.dataset)
        pass

    def test_filter(self):
        f = SizeFilter(max_size=50)
        new_dset = list(f(self.dataset))
        assert len(new_dset) < len(self.dataset)

    def test_simple_compose(self):
        g = self.dataset[0]
        tr_1 = RNAFMTransform()
        tr_2 = RfamTransform()
        t = Compose([tr_1, tr_2])
        t(self.dataset[0])
        self.check_gdata(g['rna'], 'rfam')
        self.check_ndata(g['rna'], 'rnafm')

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

