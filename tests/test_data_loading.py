import unittest

import networkx as nx

from rnaglib.data_loading import RNADataset
from rnaglib.representations import GraphRepresentation

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.default_dataset = RNADataset()
        pass

    def test_rna_get(self):
        rna = self.default_dataset[0]
        assert 'rna' in rna

    def test_dataset_from_list(self):
        rnas = [nx.Graph(name='rna1'), nx.Graph(name='rna2')]
        da = RNADataset(rnas=rnas)
        assert len(da) == len(rnas)
        pass

    def test_dataset_from_pdbids(self):
        all_rnas = ['2pwt', '5v3f', '379d',
                    '5bjo', '4pqv', '430d',
                    '1nem', '1q8n', '1f1t',
                    '2juk', '4yaz', '364d',
                    '6ez0', '2tob', '1ddy',
                    '1fmn', '2mis', '4f8u'
                    ]

        da = RNADataset(all_rnas=all_rnas, redundancy='all')
        assert len(da) == len(all_rnas)

    def test_add_representation(self):
        self.default_dataset.add_representation(GraphRepresentation())
        pass

if __name__ == "__main__":
    unittest.main()
