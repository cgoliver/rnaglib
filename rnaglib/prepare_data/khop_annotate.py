"""
Functions to take a nx object and return (nx, dict_tree, dict_rings)
"""
import sys
import os
import argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import pickle
from collections import defaultdict, Counter, OrderedDict
import multiprocessing as mlt

import networkx as nx
from tqdm import tqdm

from rnaglib.utils.graphlet_hash import extract_graphlet, build_hash_table, Hasher
from rnaglib.config.graph_keys import GRAPH_KEYS

TOOL = 'RGLIB'

def cline():
    """
    annotate_all(graph_path="../data/ref_graph", dump_path='../data/annotated/ref_graph', do_hash=True, parallel=False)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_path", default=os.path.join(script_dir, '../data/samples_v2_chunks'))
    parser.add_argument("-a", "--annot_id", default='samples', type=str, help="Annotated data ID.")
    parser.add_argument("-ha", "--do_hash", default=False, action='store_true', help="Hash graphlets.")
    parser.add_argument("-p", "--parallel", default=False, action='store_true', help='Multiprocess annotations.')
    parser.add_argument("-re", "--re_annotate", default=False, action='store_true',
                        help='Read alreadya annotated graphs.')
    args, _ = parser.parse_known_args()
    return args


def caller(graph_path=os.path.join(script_dir, '../data/samples_v2'), annot_id='samples', do_hash=False,
           parallel=False, re_annotate=False):
    annotate_all(graph_path=graph_path,
                 dump_path=os.path.join(os.path.join(script_dir, '..', 'data', 'annotated', annot_id)),
                 do_hash=do_hash, parallel=parallel,
                 re_annotate=re_annotate)
    pass


def node_2_unordered_rings(G, v, depth=5, hasher=None, hash_table=None):
    """
    Return rings centered at `v` up to depth `depth`.

    Return dict of dicts. One dict for each type of ring.
    Each inner dict is keyed by node id and its value is a list of lists.
    A ring is a list of lists with one list per depth ring.

    :param G:
    :param v:
    :param depth:
    :param: include_degrees: Whether to add a list of neighbor degrees to ring.
    :return:

    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1,2, {'label': 'A'}),\
                          (1, 3, {'label': 'B'}),\
                          (2, 3, {'label': 'C'}),\
                          (3, 4, {'label': 'A'})])
    >>> rings = node_2_unordered_rings(G, 1, depth=2)
    >>> rings['edge']
    [[None], ['A', 'B'], ['C', 'A']]

    """
    do_hash = not hasher is None

    if do_hash:
        g_hash = hasher.hash(extract_graphlet(G, v))
        assert g_hash in hash_table
        graphlet_rings = [[g_hash]]
    else:
        graphlet_rings = [[extract_graphlet(G, v)]]

    node_rings = [[v]]
    edge_rings = [[None]]
    visited = set()
    visited.add(v)
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
    :param graph: nx
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
            graph = nx.read_gpickle(os.path.join(graph_path, g))
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
                                      hasher, annot=re_annotate,
                                      graphlet_size=graphlet_size
                                      )
        print(f">>> found {len(hash_table)} graphlets.")
        name = os.path.basename(dump_path)
        pickle.dump((hasher.__dict__, hash_table),
                    open(os.path.join(dump_path + "_hash.p"), 'wb'))
    else:
        hasher = None

    graphs = os.listdir(graph_path)
    failed = 0
    print(">>> annotating all.")
    pool = mlt.Pool()
    if parallel:
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
    import doctest

    doctest.testmod()

    args = cline()
    caller(**vars(args))
