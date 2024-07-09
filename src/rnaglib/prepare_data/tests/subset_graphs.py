import unittest
from ..subset_graphs import *

class subset_graphsTestCase(unittest.TestCase):

    def test_null(self):

        g = nx.complete_graph(100)
        subset = []

        g_subsetted, g_prime = slice_graph(g, subset)

        self.assertCountEqual(g, g_prime)

    def test_remove10(self):

        g = nx.complete_graph(100)
        subset = [i for i in range(10)]

        g_subgraph, g_prime = slice_graph(g, subset)

        self.assertCountEqual(list(g_subgraph.nodes), [i for i in range(10)])
