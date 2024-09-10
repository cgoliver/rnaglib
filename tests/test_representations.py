import unittest

import networkx as nx

from rnaglib.data_loading import RNADataset
from rnaglib.transforms import GraphRepresentation

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.default_dataset = RNADataset(debug=True)
        pass

    def test_add_representation(self):
        self.default_dataset.add_representation(GraphRepresentation())
        pass

if __name__ == "__main__":
    unittest.main()
