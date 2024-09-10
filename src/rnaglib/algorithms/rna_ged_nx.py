import os
import time
import pickle
import numpy as np

from networkx import graph_edit_distance, optimize_graph_edit_distance
from rnaglib.config.build_iso_mat import iso_mat as iso_matrix
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL

e_key = GRAPH_KEYS["bp_type"][TOOL]
indel_vector = GRAPH_KEYS["indel_vector"][TOOL]
edge_map = GRAPH_KEYS["edge_map"][TOOL]
sub_matrix = np.ones_like(iso_matrix) - iso_matrix


def e_sub(e1_attr, e2_attr, label="LW"):
    return sub_matrix[edge_map[e1_attr[label]]][edge_map[e2_attr[label]]]


def e_ins(e_attr, label="LW"):
    return indel_vector[edge_map[e_attr[label]]]
    pass


def e_del(e_attr, label="LW"):
    return indel_vector[edge_map[e_attr[label]]]
    pass


def n_ins(arg):
    return 0


def n_del(arg):
    return 0


def ged_approx(g1, g2, upper_bound=None):
    """
    Compute a faster version of the ged on RNA graphs

    :param g1: A networkx graph to compare
    :param g2: A networkx graph to compare
    :param upper_bound: Maximum edit distance to consider.
    :return: The GED value
    """
    return optimize_graph_edit_distance(
        g1,
        g2,
        edge_subst_cost=e_sub,
        edge_del_cost=e_del,
        edge_ins_cost=e_ins,
        node_ins_cost=n_ins,
        node_del_cost=n_del,
        upper_bound=upper_bound,
    )


def ged(g1, g2, roots=None, upper_bound=None, timeout=None):
    """
    Compute the graph edit distance on RNA graphs (default weighting scheme is adapted to RNA)

    :param g1: A networkx graph to compare
    :param g2: A networkx graph to compare
    :param roots: Whether to match rooted subgraphs (forced pairing betweeen these nodes)
    :param upper_bound: Maximum edit distance to consider.
    :param timeout: Time after which we want to stop
    :return: The GED value

    """
    return graph_edit_distance(
        g1,
        g2,
        edge_subst_cost=e_sub,
        edge_del_cost=e_del,
        edge_ins_cost=e_ins,
        node_ins_cost=n_ins,
        node_del_cost=n_del,
        roots=roots,
        upper_bound=upper_bound,
        timeout=timeout,
    )


if __name__ == "__main__":

    def random_node(G, depth=2):
        from tools.graph_utils import bfs_expand
        import random

        random.seed(0)
        # node = random.choice(list(G.nodes()))
        node = list(G.nodes())[0]
        subG = G.subgraph(list(bfs_expand(G, [node], depth=depth)) + [node]).copy()
        return node, subG

    graph_path = os.path.join("..", "data", "annotated", "whole_v3")
    graphs = os.listdir(graph_path)
    G = pickle.load(open(os.path.join(graph_path, graphs[0]), "rb"))["graph"]
    H = pickle.load(open(os.path.join(graph_path, graphs[0]), "rb"))["graph"]
    root_g, g = random_node(G)
    root_h, h = random_node(H)

    root_h = list(h.nodes())[1]

    start = time.perf_counter()
    roots = (root_g, root_h)
    print(roots)
    # roots = None
    d = graph_edit_distance(
        g,
        h,
        edge_subst_cost=e_sub,
        edge_del_cost=e_del,
        edge_ins_cost=e_ins,
        roots=roots,
    )
    print(d, time.perf_counter() - start)
