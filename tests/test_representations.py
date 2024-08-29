import unittest

import networkx as nx

from rnaglib.data_loading import RNADataset
from rnaglib.representations import GraphRepresentation

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        rnas = [nx.Graph(name='rna1'), nx.Graph(name='rna2')]
        self.default_dataset = RNADataset(rnas=rnas)
        pass

    def test_add_representation(self):
        self.default_dataset.add_representation(GraphRepresentation())
        pass

if __name__ == "__main__":
    unittest.main()
