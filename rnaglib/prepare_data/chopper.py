"""
Chop the PDBs and extract graphs with PCA trick.
"""
import sys
import os
import traceback

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import os.path as osp
import multiprocessing as mlt

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import networkx as nx
from Bio.PDB import *

from rnaglib.utils.graph_utils import dangle_trim, gap_fill
from rnaglib.utils.graph_io import graph_from_pdbid
from rnaglib.config.graph_keys import GRAPH_KEYS

MM_of_Elements = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.0107, 'N': 14.0067,
                  'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, 'Na': 22.98976928, 'Mg': 24.305, 'Al': 26.9815386,
                  'Si': 28.0855, 'P': 30.973762, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.0983, 'Ca': 40.078,
                  'Sc': 44.955912, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938045,
                  'Fe': 55.845, 'Co': 58.933195, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.409, 'Ga': 69.723, 'Ge': 72.64,
                  'As': 74.9216, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.798, 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585,
                  'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.9063, 'Ru': 101.07, 'Rh': 102.9055, 'Pd': 106.42,
                  'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.760, 'Te': 127.6,
                  'I': 126.90447, 'Xe': 131.293, 'Cs': 132.9054519, 'Ba': 137.327, 'La': 138.90547, 'Ce': 140.116,
                  'Pr': 140.90465, 'Nd': 144.242, 'Pm': 146.9151, 'Sm': 150.36, 'Eu': 151.964, 'Gd': 157.25,
                  'Tb': 158.92535, 'Dy': 162.5, 'Ho': 164.93032, 'Er': 167.259, 'Tm': 168.93421, 'Yb': 173.04,
                  'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
                  'Pt': 195.084, 'Au': 196.966569, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804,
                  'Po': 208.9824, 'At': 209.9871, 'Rn': 222.0176, 'Fr': 223.0197, 'Ra': 226.0254, 'Ac': 227.0278,
                  'Th': 232.03806, 'Pa': 231.03588, 'U': 238.02891, 'Np': 237.0482, 'Pu': 244.0642, 'Am': 243.0614,
                  'Cm': 247.0703, 'Bk': 247.0703, 'Cf': 251.0796, 'Es': 252.0829, 'Fm': 257.0951, 'Md': 258.0951,
                  'No': 259.1009, 'Lr': 262, 'Rf': 267, 'Db': 268, 'Sg': 271, 'Bh': 270, 'Hs': 269, 'Mt': 278,
                  'Ds': 281, 'Rg': 281, 'Cn': 285, 'Nh': 284, 'Fl': 289, 'Mc': 289, 'Lv': 292, 'Ts': 294, 'Og': 294,
                  'ZERO': 0}


def residue_from_node(structure, node_id):
    """Fetch a residue with given node id.

    :param structure: Biopython structure
    :param node_id: id of node to fetch
    """
    _, chain, pos = node_id.split(".")
    for r in structure.get_residues():
        if r.get_parent().id == chain and str(r.id[1]) == pos:
            return r


def block_pca(residues):
    """
        Get PCA of coordinates in block of residues.

        :param residues: list of biopython residue objects

        :return: PCA coordinates for each residue
    """

    def element_name(atom_name):
        if atom_name[0].isupper():
            return atom_name[0]
        return atom_name

    def center_of_mass(residues, atom=False):
        """
        Compute center of mass of a residue.
        """
        M = 0
        atoms = []
        if not atom:
            for r in residues:
                atoms.extend(r.get_atoms())
        else:
            atoms = residues.get_atoms()
        for atom in atoms:
            name = element_name(atom.get_name())
            M += MM_of_Elements[name]
            pass
        center_vec = np.zeros(3)
        for atom in atoms:
            name = element_name(atom.get_name())
            mass = MM_of_Elements[name]
            center_vec += mass * atom.coord

        return (center_vec / M)

    def get_coords(residues):
        """
            Return atomic coordinates of a block.
        """
        return np.array([center_of_mass([r]) for r in residues])

    coords = get_coords(residues)
    pca = PCA()
    return pca.fit_transform(coords)
    # return pca.fit(coords).components_


def pca_chop(residues):
    """
        Return chopped structure using PCA axes.
        All residues with negative first coords are assigned to one
        half of the list. This is not valid for very
        skewed distributions of points

        :param residues: list of biopython residues
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

        :param residues: list of Biopython residues
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


def blob_to_graph(residues, graph):
    """
    Convert a list of residues back to the subgraph they correspond to

    :param residues: list of selected residues
    :param graph: nx graph they were extracted from
    :return: nx subgraph
    """

    def find_node(residue):
        """
            Gets node matching PDB residue.
        """
        for n, d in graph.nodes(data=True):
            chain = str(residue.get_parent().id)
            pos = str(residue.id[1])
            g_chain = str(d[GRAPH_KEYS['chain'][TOOL]])
            g_pos = str(d[GRAPH_KEYS['nt_position'][TOOL]])
            if (chain, pos) == (g_chain, g_pos):
                return n
        else:
            return None

    def res_to_carna_node(r):
        return find_node(r)

    nodes = list(map(res_to_carna_node, residues))
    sg = graph.subgraph(nodes).copy()
    return sg


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
        if d[GRAPH_KEYS['bp_type'][TOOL]] not in ['CWW', 'B53']:
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
    # filled_in_nodes  = bfs_expand(G, subG.nodes(), depth=1)

    # print(len(filled_in_nodes), len(subG.nodes()))
    # subG = G.subgraph(filled_in_nodes).copy()

    subG = gap_fill(G, subG)

    dangle_trim(subG)
    assert sum([1 if subG.degree(n) == 1 else 0 for n in subG.nodes()]) == 0

    for cc in nx.connected_components(subG.to_undirected()):
        if len(cc) < thresh:
            subG.remove_nodes_from(cc)
            # print('removed chunk')

    return subG


def chop_one_rna(args):
    """
    To be used by a map process, chop an rna

    :param args: should contain (rna, pdb_path, graph_path,dest)
    :return:
    """
    (g_path, pdb_path, dest) = args
    parser = MMCIFParser(QUIET=True)
    RNA = {'A', 'U', 'C', 'G'}

    _, graph_format = os.path.splitext(g_path)
    graph_format = graph_format.lstrip('.')

    if not (graph_format in ['nx', 'json']):
        return 1
    try:
        pdbid = os.path.basename(g_path).split('.')[0]

        # Check if already computed
        for processed in os.listdir(dest):
            if processed.startswith(pdbid):
                return 0

        structure = parser.get_structure('', osp.join(pdb_path, pdbid.lower() + ".cif"))[0]
        G = graph_from_pdbid(pdbid,
                             os.path.dirname(g_path),
                             graph_format=graph_format
                             )

        residues = []
        for nid in G.nodes():
            resi = residue_from_node(structure, nid)
            if resi is None:
                continue
            else:
                residues.append(resi)

        # glib node format: 3iab.R.83 <pdbid>.<chain>.<pos>

        # residues = [r for r in structure.get_residues() if r.id[0] == ' ' and
        # r.get_resname() in RNA]

        chops = chop(residues)
        for j, c in enumerate(chops):
            subgraph = blob_to_graph(c, G)
            subgraph = graph_clean(G, subgraph)
            if graph_filter(subgraph):
                nx.write_gpickle(subgraph, osp.join(dest, f"{pdbid}_{j}.nx"))
            else:
                pass
                # print(f"Graph {pdbid}_{j} failed graph filter. Had {len(subgraph.nodes())} nodes. " +
                # f"{set([d['label'] for _, _, d in subgraph.edges(data=True)])}")
        return 0
    except Exception as e:
        traceback.print_exc()
        print(e)
        return 1


def chop_all(graph_path='../data/carnaval',
             pdb_path='../data/all_rna_pdb',
             dest="../data/graphs/whole",
             parallel=True,
             graph_format='RGLIB'):
    """
        Chop all the RNAs in the dataset.
        Each graph in `graph_path` should be in the format `<pdb ID>.[nx|json]`
    """

    global TOOL
    TOOL = graph_format

    try:
        os.mkdir(dest)
    except FileExistsError:
        pass

    g_paths = [os.path.join(graph_path, g) for g in os.listdir(graph_path)]
    failed = 0
    pool = mlt.Pool()
    if parallel:
        arguments = [(rna, pdb_path, dest) for rna in g_paths]
        for res in tqdm(pool.imap_unordered(chop_one_rna, arguments), total=len(g_paths)):
            failed += res
        print(f'failed on {(failed)} on {len(g_paths)}')
        return failed
    for i, rna in tqdm(enumerate(g_paths), total=len(g_paths)):
        failed_rna = chop_one_rna((rna, pdb_path, dest))
        if failed_rna:
            failed += failed_rna
            print(f'failed on {rna}, this is the {failed}-th one on {len(g_paths)}')
    pass


if __name__ == "__main__":
    # all_rna_process(graph_path='../data/samples_graphs', pdb_path='../data/samples_pdb', dest="../data/graphs/samples")
    # all_rna_process(graph_path='../data/samples_graphs', pdb_path='../data/samples_pdb', dest="../data/graphs/samples",
    #                parallel=False)
    # all_rna_process(graph_path='../data/carnaval', pdb_path='../data/all_rna_pdb', dest="../data/graphs/test",
    # parallel=False)
    chop_all(graph_path='../data/carnaval_2', pdb_path='../data/all_rna_pdb', dest="../data/chunks_nx_2",
             parallel=False)
    pass
