"""
    Hash function for graphlets.

    Input: Graphlet (nx)
    Output: hash code
"""

import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import pickle
import time
import random
from itertools import product
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from tqdm import tqdm
from hashlib import blake2b

import networkx as nx
import numpy as np
from networkx.algorithms import betweenness_centrality
from networkx.algorithms import core_number

from struct import *
# from tools.drawing import rna_draw, rna_draw_pair
from tools.graph_utils import bfs_expand
# from tools.rna_ged import ged
from tools.rna_ged_nx import ged

iso_matrix = pickle.load(open(os.path.join(script_dir, '../data/iso_mat.p'), 'rb'))
sub_matrix = np.ones_like(iso_matrix) - iso_matrix
# iso_matrix = iso_matrix[1:, 1:]


edge_map = {'B53': 0, 'CHH': 1, 'CHS': 2, 'CHW': 3, 'CSH': 2, 'CSS': 4, 'CSW': 5, 'CWH': 3, 'CWS': 5, 'CWW': 6,
            'THH': 7, 'THS': 8, 'THW': 9, 'TSH': 8, 'TSS': 10, 'TSW': 11, 'TWH': 9, 'TWS': 11, 'TWW': 12}

indel_vector = [1 if e == 'B53' else 2 if e == 'CWW' else 3 for e in sorted(edge_map.keys())]

faces = ['W', 'S', 'H']
orientations = ['C', 'T']
labels = {orient + e1 + e2 for e1, e2 in product(faces, faces) for orient in orientations}
labels.add('B53')


class Hasher:
    def __init__(self, method='WL',
            string_hash_size=8,
            graphlet_hash_size=16,
            symmetric_edges=True,
            wl_hops=2):
        self.method = method
        self.string_hash_size = string_hash_size
        self.graphlet_hash_size = graphlet_hash_size
        self.symmetric_edges = symmetric_edges
        self.wl_hops = wl_hops

    def hash(self, G):
        """
        WL hash of a graph.

        >>> import networkx as nx
        >>> G1 = nx.Graph()
        >>> G1.add_edges_from([(('4v6m_76.nx', ('B8', 627)), ('4v6m_76.nx', ('B8', 626)), {'label': 'B53'}),\
                               (('4v6m_76.nx', ('B8', 627)), ('4v6m_76.nx', ('B8', 628)), {'label': 'B53'})])
        >>> G2 = nx.Graph()
        >>> G2.add_edges_from([(('4v6m_76.nx', ('B8', 654)), ('4v6m_76.nx', ('B8', 655)), {'label': 'B53'}),\
                               (('4v6m_76.nx', ('B8', 655)), ('4v6m_76.nx', ('B8', 656)), {'label': 'B53'})])

        >>> hasher = Hasher()
        >>> hasher.hash(G1) == hasher.hash(G2)
        True
        """
        if self.symmetric_edges:
            for u,v in G.edges():
                label = G[u][v]['label']
                if label != 'B53':
                    prefix, suffix = label[0],label[1:]
                    G[u][v]['label'] = prefix + "".join(sorted(suffix))
        items = []
        node_labels = {n: '' for n in G.nodes()}
        for k in range(self.wl_hops):
            node_labels = WL_step_edges(G, node_labels)
            c = Counter()
            # count node labels
            for node, d in node_labels.items():
                h = blake2b(digest_size=self.string_hash_size)
                h.update(d.encode('ascii'))
                c.update([h.hexdigest()])
            # sort the counter, extend total counts
            items.extend(sorted(c.items(), key=lambda x: x[0]))

        # hash the final counter
        h = blake2b(digest_size=self.graphlet_hash_size)
        h.update(str(tuple(items)).encode('ascii'))
        return h.hexdigest()

def nei_agg(G, n):
    x = tuple(sorted([G.nodes()[n]['label']] + [G.nodes()[n]['label'] for n in G.neighbors(n)]))
    return x

def nei_agg_edges(G, n, node_labels):
    x = [node_labels[n]]
    for nei in G.neighbors(n):
        x.append(G[n][nei]['label'] + node_labels[nei])
    return ''.join(sorted(x))


def WL_step(G):
    new_labels = {n: nei_agg(G, n) for n in G.nodes()}
    nx.set_node_attributes(G, new_labels, 'label')
    pass


def WL_step_edges(G, labels):
    """
        Aggregate neighbor labels and edge label.
    """
    new_labels = dict()
    for n in G.nodes():
        new_labels[n] = nei_agg_edges(G, n, labels)
    return new_labels

def extract_graphlet(G, n, size=1):
    return G.subgraph(bfs_expand(G, [n], depth=size)).copy()
    pass


def build_hash_table(graph_dir, hasher, graphlets=True,
                     max_graphs=0,
                     graphlet_size=1,
                     mode='count',
                     annot=False):
    hash_table = {}
    graphlist = os.listdir(graph_dir)
    if max_graphs:
        graphlist = graphlist[:max_graphs]
    random.seed(0)
    random.shuffle(graphlist)
    start = time.time()
    for g in tqdm(graphlist):
        if annot:
            G = pickle.load(open(os.path.join(graph_dir, g), 'rb'))['graph']
        else:
            G = pickle.load(open(os.path.join(graph_dir, g), 'rb'))


        if graphlets:
            todo = (extract_graphlet(G, n, size=graphlet_size) for n in G.nodes())
        else:
            todo = [G]
        for n in todo:
            h = hasher.hash(n)
            if h not in hash_table:
                if mode == 'append':
                    hash_table[h] = {'graphs': [n]}
                else:
                    hash_table[h] = {'graph': n, 'count': 1}
            else:
                # see if collision
                if mode == 'append':
                    hash_table[h]['graphs'].append(n)
                else:
                    hash_table[h]['count'] += 1
    return hash_table


def GED_hashtable_hashed(h_G, h_H, GED_table, graphlet_table, normed=True,
                         max_edges=7,
                         beta=.50,
                         timeout=60,
                         similarity=False):
    """
        Produce a hash table that contains pairwise GEDs between graphs.
        {h_i:{h_j: d(G_i, G_j)}}
        Collisions are resolved with an extra entry in the hash digest that gives the 
        index of the graph in the bucket.
    """

    try:
        return GED_table[h_G][h_H]
    except:
        pass
    try:
        return GED_table[h_H][h_G]
    except:
        pass

    G = graphlet_table[h_G]['graph']
    H = graphlet_table[h_H]['graph']


    distance = ged(G,H, timeout=timeout)
    # d = ged(G, H, sub_matrix=sub_matrix,
            # edge_map=edge_map,
            # indel=indel_vector)
    # distance = d.cost

    # This is added by vincent, to use to be closer from the rest of the framework riso
    if similarity:
        similarity = np.exp(-beta * distance)
        GED_table[h_G][h_H] = similarity
        return similarity

    elif normed:
        # d /= (max(sub_matrix) +  max(indel_vector) * (max_edges - 1))
        distance = 1 - np.exp(-beta * distance)

    # rna_draw_pair([G, H],
    # estimated_value=[0, d])
    GED_table[h_G][h_H] = distance
    return distance



def hash_analyze(annot_dir):
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
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    table = build_hash_table("../data/annotated/whole_v3", Hasher(), mode='append', max_graphs=10)
    #check hashtable visually
    from tools.drawing import rna_draw
    for h,data in table.items():
        print(h)
        for g in data['graphs']:
            rna_draw(g, show=True)

