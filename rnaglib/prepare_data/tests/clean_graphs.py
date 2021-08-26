import sys
import os
import networkx as nx
import unittest

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join('..', '..', script_dir))

from tools.graph_utils import bfs_expand

class clean_graphsTestCase(unittest.TestCase):

    def test_bfs_expand(self):

        g = nx.Graph()
        g.add_nodes_from([  (1, (1,1)),
                            (2, (1,1)),
                            (3, (1,1)),
                            (4, (1,1))])

        g.add_edges_from([((1, (1, 1)), (2, (1, 1))),
                        ((1, (1, 1)), (3, (1, 1))),
                        ((3, (1, 1)), (4, (1, 1)))])

        # Calling expand on just 1 should return the whole graph
        new_nodes = bfs_expand(g, [(1, (1,1))])

        self.assertEqual(list(g.nodes), new_nodes)
