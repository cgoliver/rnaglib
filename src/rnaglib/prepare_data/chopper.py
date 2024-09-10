"""
Chops graphs built by rglib into subgraphs based on the
coordinates of each residue orthogonal to main PCA axis.
"""
import sys
import os

from joblib import Parallel, delayed
import os.path as osp
import multiprocessing as mlt

import numpy as np
from sklearn.decomposition import PCA
import networkx as nx

from rnaglib.algorithms import dangle_trim, gap_fill
from rnaglib.utils import load_json, dump_json

def block_pca(residues):
    """
    Get PCA of coordinates in block of residues.

    :param residues: list of tuples (node_id, coordinate)
    :return: PCA coordinates for each residue
    """

    coords = np.array([coord for _, coord in residues])
    pca = PCA()
    return pca.fit_transform(coords)

def pca_chop(residues):
    """
    Return chopped structure using PCA axes.
    All residues with negative first coords are assigned to one
    half of the list. This is not valid for very
    skewed distributions of points

    :param residues: list of tuples (node_id, coords)
    """
    proj = block_pca(residues)
    s1, s2 = [], []
    for i, p in enumerate(proj):
        if p[0] > 0:
            s1.append(residues[i])
        else:
            s2.append(residues[i])
    # print(f"sum check {len(s1) + len(s2)} == {len(residues)}, {len(proj)}")
    return s1, s2

def chop(residues, max_size=50):
    """
    Perform recursive chopping.

    :param residues: list of tuples (node_id, coord)
    :param max_size: stop chopping when `max_size` residues are left in a
                     chop.
    """
    if len(residues) > max_size:
        # do pca on the current residues
        res_1, res_2 = pca_chop(residues)
        yield from chop(res_1)
        yield from chop(res_2)
    else:
        yield residues


def graph_filter(G, max_nodes=10):
    """
    Check if a graph is valid : Small enough and with at least one non canonical

    :param G: An nx graph
    :param max_nodes : The max number of nodes
    :return: boolean
    """
    if len(G.nodes()) < max_nodes:
        return False
    for _, _, d in G.edges(data=True):
        if d['LW'] not in ['CWW', 'B35', 'B53']:
            return True
    return False

def graph_clean(G, subG, thresh=8):
    """
    Do post-cleanup on graph.
    Fill in backbones, remove islands, remove dangles.
    E.g. remove single nodes.

    :param G: An nx graph
    :param thresh: The threshold under which to discard small connected components
    """
    subG = gap_fill(G, subG)

    dangle_trim(subG)
    assert sum([1 if subG.degree(n) == 1 else 0 for n in subG.nodes()]) == 0

    for cc in nx.connected_components(subG.to_undirected()):
        if len(cc) < thresh:
            subG.remove_nodes_from(cc)
            # print('removed chunk')

    return subG


def chop_one_rna(G):
    """
    Returns subgraphs of a given rglib graph by following a chopping
    procedure.

    :param G: networkx graph built by rnaglib.
    :return: list of subgraphs
    """
    residues = []
    missing_coords = 0
    for n, d in sorted(G.nodes(data=True)):
        try:
            residues.append((n, d['C5prime_xyz']))
        except KeyError:
            missing_coords += 1
            continue
    print(f">>> Graph {G.graph['pdbid']} has {missing_coords} residues with missing coords.")

    # glib node format: 3iab.R.83 <pdbid>.<chain>.<pos>
    # residues = [r for r in structure.get_residues() if r.id[0] == ' ' and
    # r.get_resname() in RNA]

    try:
        chops = chop(residues)
        subgraphs = []
        for j, this_chop in enumerate(chops):
            subgraph = G.subgraph((n for n,_ in this_chop)).copy()
            subgraph = graph_clean(G, subgraph)
            if graph_filter(subgraph):
                subgraphs.append(subgraph)
            else:
                pass
        print(f"RNA with {len(residues)} bases chopped to {len(subgraphs)} chops.")
        return subgraphs
    except:
        print("chopping error")
        return None

def chop_all(graph_path, dest, n_jobs=4, parallel=True):
    """
    Chop and dump all the rglib graphs in the dataset.

    :param graph_path: path to graphs for chopping
    :param dest: path where chopped graphs will be dumped
    :n_jobs: number of workers to use
    :paralle: whether to use multiprocessing
    """

    try:
        os.mkdir(dest)
    except FileExistsError:
        pass

    graphs = (load_json(os.path.join(graph_path, g)) for g in os.listdir(graph_path))
    failed = 0
    subgraphs = Parallel(n_jobs=n_jobs)(delayed(chop_one_rna)(G) for G in graphs)
    # dump the chops
    for chopped_rna in subgraphs:
        if chopped_rna is None:
            continue
        for i, this_chop in enumerate(chopped_rna):
            dump_json(os.path.join(dest, f"{this_chop.graph['pdbid'][0]}_{i}.json"), this_chop)
    pass

if __name__ == "__main__":
    chop_all('db/graphs/all_graphs',
             "db/graphs_chopped",
             parallel=False
             )
    pass
