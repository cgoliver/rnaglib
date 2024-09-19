"""
Functions to take a nx object and return (nx, dict_tree, dict_rings)
"""
import sys
import os
import argparse

import pickle
from collections import defaultdict, Counter, OrderedDict
import multiprocessing as mlt

import networkx as nx
from tqdm import tqdm

from rnaglib.algorithms import build_hash_table, Hasher
from rnaglib.algorithms import extract_graphlet
from rnaglib.utils import load_json
from rnaglib.config import GRAPH_KEYS, TOOL


def node_2_unordered_rings(G, node, depth=5, hasher=None, hash_table=None):
    """
    Return rings centered at `node` up to depth `depth`.

    Return dict of dicts. One dict for each type of ring.
    Each inner dict is keyed by node id and its value is a list of lists.
    A ring is a list of lists with one list per depth ring.

    :param G: Networkx graph
    :param node: A node from G
    :param depth: The depth or number of hops starting from node to include in the ring annotation
    :param hasher: A hasher object to use for encoding the graphlets
    :param hash_table: A hash table to fill with the annotations

    :return: {'node_annots': list, 'edge_annots': list, 'graphlet_annots': list} each of the list is of length depth
    and contains lists of the nodes in the ring at each depth.

    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1,2, {'LW': 'A'}),\
                          (1, 3, {'LW': 'B'}),\
                          (2, 3, {'LW': 'C'}),\
                          (3, 4, {'LW': 'A'})])
    >>> rings = node_2_unordered_rings(G, 1, depth=2)
    >>> rings['edge']
    [[None], ['A', 'B'], ['C', 'A']]
    """
    do_hash = not hasher is None

    if do_hash:
        g_hash = hasher.hash(extract_graphlet(G, node))
        assert g_hash in hash_table
        graphlet_rings = [[g_hash]]
    else:
        graphlet_rings = [[extract_graphlet(G, node)]]

    node_rings = [[node]]
    edge_rings = [[None]]
    visited = set()
    visited.add(node)
    visited_edges = set()
    for k in range(depth):
        ring_k = []
        edge_ring_k = []
        ring_k_graphlet = []
        for node in node_rings[k]:
            children = []
            e_labels = []
            children_graphlet = []
            for nei in G.neighbors(node):
                if nei not in visited:
                    visited.add(nei)
                    children.append(nei)
                # check if we've seen this edge.
                e_set = frozenset([node, nei])
                if e_set not in visited_edges:
                    if do_hash:
                        nei_h = hasher.hash(extract_graphlet(G, nei))
                        assert nei_h in hash_table
                        children_graphlet.append(nei_h)
                    else:
                        children_graphlet.append(extract_graphlet(G, nei))
                    e_labels.append(G[node][nei][GRAPH_KEYS['bp_type'][TOOL]])
                    visited_edges.add(e_set)
            ring_k.extend(children)
            edge_ring_k.extend(e_labels)
            if do_hash:
                ring_k_graphlet.extend(children_graphlet)
        node_rings.append(ring_k)
        edge_rings.append(edge_ring_k)
        graphlet_rings.append(ring_k_graphlet)
    # uncomment to draw root node
    # from tools.drawing import rna_draw
    # rna_draw(G, node_colors=['blue' if n == v else 'grey' for n in G.nodes()], show=True)
    return {'node': node_rings, 'edge': edge_rings, 'graphlet': graphlet_rings}


def build_ring_tree_from_graph(graph, depth=5, hasher=None, hash_table=None):
    """
    This function mostly loops over nodes and calls the annotation function.
    It then puts the annotated data into the graph.

    :param graph: nx graph
    :param depth: The depth or number of hops starting from node to include in the ring annotation
    :param hasher: A hasher object to use for encoding the graphlets
    :param hash_table: A hash table to fill with the annotations

    :return: dict (ring_level: node: ring)
    """
    dict_ring = defaultdict(dict)
    for node in sorted(graph.nodes()):
        rings = node_2_unordered_rings(graph,
                                       node,
                                       depth=depth,
                                       hasher=hasher,
                                       hash_table=hash_table)
        dict_ring['node'][node] = rings['node']
        dict_ring['edge'][node] = rings['edge']
        dict_ring['graphlet'][node] = rings['graphlet']
    return dict_ring


def annotate_one(args):
    """
    To be called by map

    :param args: ( g (name of the graph),

    :return:
    """
    g, graph_path, dump_path, hasher, re_annotate, hash_table = args
    try:
        dump_name = os.path.basename(g).split('.')[0] + "_annot.p"
        dump_full = os.path.join(dump_path, dump_name)
        for processed in os.listdir(dump_path):
            if processed.startswith(dump_name):
                return 0, 0
        if re_annotate:
            graph = pickle.load(open(os.path.join(graph_path, g), 'rb'))['graph']
        else:
            graph = load_json(os.path.join(graph_path, g))
        rings = build_ring_tree_from_graph(graph,
                                           depth=5,
                                           hasher=hasher,
                                           hash_table=hash_table)

        if dump_path:
            pickle.dump({'graph': graph,
                         'rings': rings},
                        open(dump_full, 'wb'))
        return 0, g
    except Exception as e:
        print(e)
        return 1, g


def annotate_all(dump_path='../data/annotated/sample_v2',
                 graph_path='../data/chunks_nx',
                 parallel=True,
                 do_hash=True,
                 wl_hops=3,
                 graphlet_size=1,
                 re_annotate=False):
    """
    Routine for all files in a folder

    :param dump_path:
    :param graph_path:
    :param parallel:

    :return:
    """
    try:
        os.mkdir(dump_path)
    except:
        pass

    if do_hash:
        print(">>> hashing graphlets.")
        hasher = Hasher(wl_hops=wl_hops)
        hash_table = build_hash_table(graph_path,
                                      hasher,
                                      graphlet_size=graphlet_size
                                      )
        print(f">>> found {len(hash_table)} graphlets.")
        name = os.path.basename(dump_path)
        pickle.dump((hasher.__dict__, hash_table),
                    open(os.path.join(dump_path + "_hash.p"), 'wb'))
    else:
        hasher = None
        hash_table = None

    graphs = os.listdir(graph_path)
    failed = 0
    print(">>> annotating all.")
    if parallel:
        pool = mlt.Pool()
        print(">>> going parallel")
        arguments = [(g, graph_path, dump_path, hasher, re_annotate, hash_table) for g in graphs]
        for res in tqdm(pool.imap_unordered(annotate_one, arguments), total=len(graphs)):
            if res[0]:
                failed += 1
                print(f'failed on {res[1]}, this is the {failed}-th one on {len(graphs)}')
        print(f'failed on {(failed)} on {len(graphs)}')
        return failed
    for graph in tqdm(graphs, total=len(graphs)):
        res = annotate_one((graph, graph_path, dump_path, hasher, re_annotate, hash_table))
        if res[0]:
            failed += 1
            print(f'failed on {graph}, this is the {failed}-th one on {len(graphs)}')
    pass


if __name__ == '__main__':
    # import doctest
    # doctest.testmod()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_path", default=os.path.join(script_dir, '../data/samples_v2_chunks')
                        , type=str, help="The path of the graphs directory you want to annotate.")
    parser.add_argument("-a", "--annot_path", default=os.path.join(script_dir, '../data/annotated/samples'),
                        type=str, help="The path where we want our annotated graphs to be created.")
    parser.add_argument("-ha", "--do_hash", default=False, action='store_true', help="Hash graphlets.")
    parser.add_argument("-p", "--parallel", default=False, action='store_true', help='Multiprocess annotations.')
    parser.add_argument("-re", "--re_annotate", default=False, action='store_true',
                        help='Read already annotated graphs.')
    args, _ = parser.parse_known_args()
    annotate_all(graph_path=args.graph_path, dump_path=args.annot_path, do_hash=args.do_hash,
                 parallel=args.parallel, re_annotate=args.re_annotate)
