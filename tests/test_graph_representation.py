"""Residue and edge ordering of the PyG graph representation, for graphs with and without label_seq_id (the released
2.0.2 graphs have no label_seq_id)."""

import unittest

import networkx as nx
import torch

from rnaglib.transforms import GraphRepresentation
from rnaglib.transforms.represent.graph import residue_order_key


def make_graph(nodes, edges):
    """nodes: {node id: label_seq_id or None (attribute omitted)}; edges: [(u, v, LW)]."""
    g = nx.DiGraph()
    for n, lsi in nodes.items():
        attrs = {"chain_id": n.split(".")[1]}
        if lsi is not None:
            attrs["label_seq_id"] = lsi
        g.add_node(n, **attrs)
    for u, v, lw in edges:
        g.add_edge(u, v, LW=lw)
    return g


def to_pyg(g):
    rep = GraphRepresentation(framework="pyg")
    order = sorted(g.nodes(), key=residue_order_key(g))
    feats = {"nt_features": {n: torch.tensor([float(order.index(n))]) for n in g.nodes()}}
    return rep.to_pyg(g, feats), order, rep.edge_map


class TestResidueOrder(unittest.TestCase):
    def check_pyg(self, g, expected_order):
        data, order, edge_map = to_pyg(g)
        self.assertEqual(order, expected_order)
        # node features follow the residue order
        self.assertEqual(data.x.flatten().tolist(), [float(i) for i in range(len(order))])
        # edges are sorted by (source index, target index) and keep their own label
        pairs = [tuple(c) for c in data.edge_index.T.tolist()]
        self.assertEqual(pairs, sorted(pairs))
        expected = {(order.index(u), order.index(v)): edge_map[d["LW"]] for u, v, d in g.edges(data=True)}
        self.assertEqual(dict(zip(pairs, data.edge_attr.tolist())), expected)
        self.assertEqual(len(pairs), g.number_of_edges())

    def test_without_label_seq_id(self):
        # released 2.0.2 graphs: author numbering from the node id, numeric (2 < 10), negative numbers, insertion codes
        nodes = ["1abc.B.1", "1abc.A.10", "1abc.A.2", "1abc.A.1A", "1abc.A.1", "1abc.A.-1"]
        g = make_graph({n: None for n in nodes},
                       [("1abc.A.-1", "1abc.A.1", "B35"), ("1abc.A.1", "1abc.A.-1", "B53"),
                        ("1abc.A.1", "1abc.A.1A", "B35"), ("1abc.A.1A", "1abc.A.1", "B53"),
                        ("1abc.A.2", "1abc.A.10", "B35"), ("1abc.A.10", "1abc.A.2", "B53"),
                        ("1abc.A.10", "1abc.B.1", "cWW"), ("1abc.B.1", "1abc.A.10", "cWW")])
        self.check_pyg(g, ["1abc.A.-1", "1abc.A.1", "1abc.A.1A", "1abc.A.2", "1abc.A.10", "1abc.B.1"])

    def test_with_label_seq_id(self):
        # sequence order differs from author numbering (inserted segment numbered 1000)
        g = make_graph({"1abc.A.1": 1, "1abc.A.1000": 2, "1abc.A.2": 3, "1abc.B.5": 1},
                       [("1abc.A.1", "1abc.A.1000", "B35"), ("1abc.A.1000", "1abc.A.1", "B53"),
                        ("1abc.A.1000", "1abc.A.2", "B35"), ("1abc.A.2", "1abc.A.1000", "B53"),
                        ("1abc.A.2", "1abc.B.5", "tWH")])
        self.check_pyg(g, ["1abc.A.1", "1abc.A.1000", "1abc.A.2", "1abc.B.5"])

    def test_partial_label_seq_id_falls_back_to_author_order(self):
        g = make_graph({"1abc.A.3": 1, "1abc.A.2": None, "1abc.A.1": 3}, [])
        self.assertEqual(sorted(g.nodes(), key=residue_order_key(g)), ["1abc.A.1", "1abc.A.2", "1abc.A.3"])


if __name__ == "__main__":
    unittest.main()
