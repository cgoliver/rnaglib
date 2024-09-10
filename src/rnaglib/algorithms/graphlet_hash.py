"""
    Hash function for graphlets.

    Input: Graphlet (nx)
    Output: hash code
"""

import os

from tqdm import tqdm
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl
import numpy as np

from rnaglib.utils import load_json
from rnaglib.config import iso_mat as iso_matrix
from rnaglib.config import GRAPH_KEYS, TOOL
from .rna_ged_nx import ged
from .graph_algos import extract_graphlet

e_key = GRAPH_KEYS["bp_type"][TOOL]
indel_vector = GRAPH_KEYS["indel_vector"][TOOL]
edge_map = GRAPH_KEYS["edge_map"][TOOL]
sub_matrix = np.ones_like(iso_matrix) - iso_matrix


class Hasher:
    def __init__(
        self,
        method="WL",
        string_hash_size=8,
        graphlet_hash_size=16,
        symmetric_edges=True,
        wl_hops=2,
        label="LW",
        directed=True,
    ):
        """
        The hasher object. Once created, it will hash new graphs and optionnaly, one can run it onto
        a whole graph dir to create the hashtable once and for all.

        :param method: for now only WL hashing is supported
        :param string_hash_size: The length of the hash, longer ones will yields less collision but more processing time
        :param graphlet_hash_size: Same thing for hashes of graphlets
        :param symmetric_edges: Whether to use symetric weights for the edges.
        :param wl_hops: The depth to consider for hashing
        :param label: How the data for the surrounding edges is encoded in the nodes.
        :param directed: To use directed graphs instead of undirected ones
        """
        self.method = method
        self.string_hash_size = string_hash_size
        self.graphlet_hash_size = graphlet_hash_size
        self.symmetric_edges = symmetric_edges
        self.wl_hops = wl_hops
        self.label = label
        self.hash_table = None
        self.directed = directed

    def hash(self, graph):
        """
        WL hash of a graph.

        :param graph: nx graph to hash
        :return: hash

        >>> import networkx as nx
        >>> G1 = nx.Graph()
        >>> G1.add_edges_from([(('4v6m_76.nx', ('B8', 627)), ('4v6m_76.nx', ('B8', 626)), {'LW': 'B53'}),\
                               (('4v6m_76.nx', ('B8', 627)), ('4v6m_76.nx', ('B8', 628)), {'LW': 'B53'})])
        >>> G2 = nx.Graph()
        >>> G2.add_edges_from([(('4v6m_76.nx', ('B8', 654)), ('4v6m_76.nx', ('B8', 655)), {'LW': 'B53'}),\
                               (('4v6m_76.nx', ('B8', 655)), ('4v6m_76.nx', ('B8', 656)), {'LW': 'B53'})])

        >>> hasher = Hasher()
        >>> hasher.hash(G1) == hasher.hash(G2)
        True
        """
        if self.symmetric_edges:
            for u, v in graph.edges():
                label = graph[u][v][self.label]
                if label != "B53":
                    prefix, suffix = label[0], label[1:]
                    graph[u][v][self.label] = prefix + "".join(sorted(suffix))
        return wl(graph, edge_attr=self.label, iterations=self.wl_hops)

    def get_hash_table(self, graph_dir, max_graphs=0):
        self.hash_table = build_hash_table(
            graph_dir,
            hasher=self,
            max_graphs=max_graphs,
            graphlet_size=self.wl_hops,
            mode="count",
            label=self.label,
            directed=self.directed,
        )

    def get_node_hash(self, graph, n):
        """
        Get the correct node hashing from a node and a graph

        :param graph:
        :param n:
        :return:
        """
        return self.hash(
            extract_graphlet(graph, n, size=self.wl_hops, label=self.label)
        )


'''
def nei_agg(graph, node, label='LW'):
    x = tuple(sorted([graph.nodes()[node][label]] + [graph.nodes()[n][label] for n in graph.neighbors(node)]))
    return x


def nei_agg_edges(graph, node, node_labels, edge_labels='LW'):
    x = [node_labels[node]]
    for nei in graph.neighbors(node):
        x.append(graph[node][nei][edge_labels] + node_labels[nei])
    return ''.join(sorted(x))


def WL_step(graph, label='LW'):
    new_labels = {n: nei_agg(graph, n) for n in graph.nodes()}
    nx.set_node_attributes(graph, new_labels, label)
    pass


def WL_step_edges(G, labels):
    """
        Aggregate neighbor labels and edge label.
    """
    new_labels = dict()
    for n in G.nodes():
        new_labels[n] = nei_agg_edges(G, n, labels)
    return new_labels
'''


def build_hash_table(
    graph_dir,
    hasher,
    graphlets=True,
    max_graphs=0,
    graphlet_size=1,
    mode="count",
    label="LW",
    directed=True,
):
    """

    Iterates over nodes of the graphs in graph dir and fill a hash table with their graphlets hashes

    :param graph_dir:
    :param hasher:
    :param graphlets:
    :param max_graphs:
    :param graphlet_size:
    :param mode:
    :param label:
    :return:
    """
    hash_table = {}
    graphlist = os.listdir(graph_dir)
    if max_graphs:
        graphlist = graphlist[:max_graphs]
    for g in tqdm(graphlist):
        print(f"getting hashes : doing graph {g}")
        G = load_json(os.path.join(graph_dir, g))
        if not directed:
            G = G.to_undirected()
        if graphlets:
            todo = [
                extract_graphlet(G, n, size=graphlet_size, label=label)
                for n in G.nodes()
            ]
        else:
            todo = [G]
        for i, n in enumerate(todo):
            h = hasher.hash(n)
            if h not in hash_table:
                if mode == "append":
                    hash_table[h] = {"graphs": [n]}
                else:
                    hash_table[h] = {"graph": n, "count": 1}
            else:
                # see if collision
                if mode == "append":
                    hash_table[h]["graphs"].append(n)
                else:
                    hash_table[h]["count"] += 1
    return hash_table


def get_ged_hashtable(
    h_G,
    h_H,
    GED_table,
    graphlet_table,
    normed=True,
    beta=0.50,
    timeout=60,
    similarity=False,
):
    """
    Get the GED between two hashes.

    Then update a hash table that contains pairwise GEDs between graphs.
    {h_i:{h_j: d(G_i, G_j)}}

    :param h_G: hash of first graphlet
    :param h_H:  hash of second graphlet
    :param GED_table: The resulting object, it is modified in place
    :param graphlet_table: a map graphlet_hash : graph
    :param timeout: ged parameter
    :param similarity: Whether to use a similarity measure instead, by taking exp(-x/beta)
    :param normed: Whether to normalize the resulting ged. It returns 1 - similarity
    :param beta: If we normalize or turn into a similarity, the temperature factor to apply to the gaussian
    :return: The ged value
    """

    try:
        return GED_table[h_G][h_H]
    except:
        pass
    try:
        return GED_table[h_H][h_G]
    except:
        pass

    G = graphlet_table[h_G]["graph"]
    H = graphlet_table[h_H]["graph"]

    distance = ged(G, H, timeout=timeout)

    if similarity:
        similarity = np.exp(-beta * distance)
        GED_table[h_G][h_H] = similarity
        return similarity

    elif normed:
        # d /= (max(sub_matrix) +  max(indel_vector) * (max_edges - 1))
        distance = 1 - np.exp(-beta * distance)

    # estimated_value=[0, d])
    GED_table[h_G][h_H] = distance
    return distance


'''
def hash_analyze(annot_dir):
    """Check hashing properties on given graphs (deprecated).

    :param annot_dir: path containing annotated graphs
    """
    from itertools import product
    import pandas as pd

    features = ['e_types', 'betweenness', 'degree', 'core']
    results = defaultdict(list)

    for args in product([True, False], repeat=4):
        if args == [False] * 3:
            continue
        else:
            arg_dic = {f: v for f, v in zip(features, args)}
            start = time.time()
            hash_table, _ = build_hash_table(annot_dir, graphlet_size=1, **arg_dic)
            tot = time.time() - start
            if not hash_table is None:
                n, k, lf = load_factor(hash_table)
            else:
                n, k, lf = (None, None, None)
                tot = None
            print(arg_dic, lf)
            for key, v in arg_dic.items():
                results[key].append(v)
            results['k'].append(k)
            results['n'].append(n)
            results['load_factor'].append(lf)
            results['time'].append(tot)
    df = pd.DataFrame.from_dict(results)
    print(df.to_latex(index=False))
    pass



def graphlet_distribution(hash_table):
    """
        Plot counts for graphlets.
        Hash table should have a counts attribute.

        :param hash_table: hash table on RNA data
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # flat = flatten_table(hash_table)
    flat = hash_table
    table_sort = sorted(flat.values(), key=lambda x: x['count'])
    counts_sort = [d['count'] for d in table_sort]
    hist, bins = np.histogram(counts_sort, bins=100000)
    print(hist, bins)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.plot(bins[:-1], hist, linestyle='', marker='o', alpha=.8)
    plt.gca().set_aspect('equal')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.xlabel("Number of occurences")
    plt.ylabel("Number of graphlets")
    plt.show()

    return

    counts = [g['count'] for g in flat.values()]
    ax = sns.distplot(counts, norm_hist=False)
    ax.set_yscale('log')
    plt.show()
    print(table_sort[1]['count'])
    rna_draw(table_sort[0]['graph'], show=True)
    print(table_sort[-1]['count'])
    rna_draw(table_sort[-2]['graph'], show=True)

    pass
'''

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # table = build_hash_table("../data/annotated/whole_v3", Hasher(), mode='append', max_graphs=10)
    # check hashtable visually
    # from drawing import rna_draw

    # for h, data in table.items():
    # print(h)
    # for g in data['graphs']:
    # rna_draw(g, show=True)
