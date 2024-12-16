import unittest

import networkx as nx

from rnaglib.data_loading import RNADataset
from rnaglib.transforms import GraphRepresentation
from rnaglib.transforms import SequenceRepresentation
from rnaglib.transforms import ChainSplitTransform
from rnaglib.transforms import FeaturesComputer


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.default_dataset = RNADataset(debug=True)
        pass

    def test_add_representation(self):
        self.default_dataset.add_representation(GraphRepresentation())
        pass

    def test_sequence_representation(self):
        rep = SequenceRepresentation(framework="torch")
        splitter = ChainSplitTransform()
        feat = FeaturesComputer(nt_features="nt_code")
        g_chain = list(splitter(self.default_dataset[0]))[0]
        seq_g = rep(g_chain["rna"], feat(g_chain))
        pass


if __name__ == "__main__":
    unittest.main()
