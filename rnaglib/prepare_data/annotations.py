"""
Package installs:
conda install -c salilab dssp
"""
import os
import sys

import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.DSSP import DSSP
import csv
from collections import defaultdict
import json
import networkx as nx
from tqdm import tqdm
from time import perf_counter

script_dir = os.path.join(os.path.realpath(__file__), '..')
sys.path.append(os.path.join(script_dir, '..'))

IONS = ["3CO", "ACT", "AG", "AL", "ALF", "AU", "AU3", "BA", "BEF", "BO4", "BR", "CA", "CAC", "CD", "CL", "CO",
        "CON", "CS", "CU", "EU3", "F", "FE", "FE2", "FLC", "HG", "IOD", "IR", "IR3", "IRI", "IUM", "K", "LI",
        "LU", "MG", "MLI", "MMC", "MN", "NA", "NCO", "NH4", "NI", "NO3", "OH", "OHX", "OS", "PB", "PO4", "PT",
        "PT4", "RB", "RHD", "RU", "SE4", "SM", "SO4", "SR", "TB", "TL", "VO4", "ZN"]

ALLOWED_ATOMS = ['C', 'H', 'N', 'O', 'Br', 'Cl', 'F', 'P', 'Si', 'B', 'Se']
ALLOWED_ATOMS += [atom_name.upper() for atom_name in ALLOWED_ATOMS]
ALLOWED_ATOMS = set(ALLOWED_ATOMS)


def is_dna(res):
    """
    Returns true if the input residue is a DNA molecule

    :param res: biopython residue object
    """
    if res.id[0] != ' ':
        return False
    if is_aa(res):
        return False
    # resnames of DNA are DA, DC, DG, DT
    if 'D' in res.get_resname():
        return True
    else:
        return False


def hariboss_filter(lig, cif_dict, mass_lower_limit=160, mass_upper_limit=1000):
    """
    Sorts ligands into ion / ligand / None
     Returns ions for a specific list of ions, ligands if the hetatm has the right atoms and mass and None otherwise
    :param lig: A biopython ligand residue object
    :param cif_dict: The output of the biopython MMCIF2DICT object
    :param mass_lower_limit:
    :param mass_upper_limit:
    :return:
    """
    lig_name = lig.id[0][2:]
    if lig_name == 'HOH':
        return None

    if lig_name in IONS:
        return 'ion'

    lig_mass = float(cif_dict['_chem_comp.formula_weight'][cif_dict['_chem_comp.id'].index(lig_name)])

    if lig_mass < mass_lower_limit or lig_mass > mass_upper_limit:
        return None
    ligand_atoms = set([atom.element for atom in lig.get_atoms()])
    if 'C' not in ligand_atoms:
        return None
    if any([atom not in ALLOWED_ATOMS for atom in ligand_atoms]):
        return None
    return 'ligand'


def get_mmcif_graph_level(mmcif_dict):
    """
    Parse an mmCIF dict and return some metadata.

    :param cif: output of the Biopython MMCIF2Dict function
    :return: dictionary of mmcif metadata (for now only resolution terms)
    """
    try:
        resolution_lo = mmcif_dict['_reflns.d_resolution_low']
        resolution_hi = mmcif_dict['_reflns.d_resolution_high']
    except KeyError:
        resolution_lo, resolution_hi = (None, None)
    return {'resolution_low': resolution_lo,
            'resolution_high': resolution_hi
            }


# def get_hetatm(cif_dict):
#     all_hetatm = set(cif_dict.get('_pdbx_nonpoly_scheme.mon_id', []))
#     return all_hetatm

def get_small_partners(cif, mmcif_dict=None, radius=6, mass_lower_limit=160, mass_upper_limit=1000):
    """
    Returns all the relevant small partners in the form of a dict of list of dicts:
    {'ligands': [
                    {'ligand_id': ('H_ARG', 47, ' '),
                     'ligand_name': 'ARG'
                     'rna_neighs': ['1aju.A.21', '1aju.A.22', ... '1aju.A.41']},
                  ],
     'ions': [
                    {'ion_id': ('H_ZN', 56, ' '),
                     'ion_name': 'ZN'],
                     'rna_neighs': ['x', y ,z]}
                     }

    :param cif: path to a mmcif file
    :param mmcif_dict: if it got computed already
    :return:
    """
    structure_id = cif[-8:-4]
    print(f'Parsing structure {structure_id}...')

    mmcif_dict = MMCIF2Dict(cif) if mmcif_dict is None else mmcif_dict
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, cif)

    atom_list = unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)

    all_interactions = {'ligands': [], 'ions': []}

    model = structure[0]
    for res_1 in model.get_residues():
        # Only look around het_flag
        het_flag = res_1.id[0]
        if 'H' in het_flag:
            # hariboss select the right heteroatoms and look around ions and ligands
            selected = hariboss_filter(res_1, mmcif_dict,
                                       mass_lower_limit=mass_lower_limit,
                                       mass_upper_limit=mass_upper_limit)
            if selected is not None:  # ion or ligand
                interaction_dict = {f'{selected}_id': res_1.id, f'{selected}_name': res_1.id[0][2:]}
                found_rna_neighbors = set()
                for atom in res_1:
                    # print(atom)
                    for res_2 in neighbors.search(atom.get_coord(), radius=radius, level='R'):
                        # Select for interactions with RNA
                        if not (is_aa(res_2) or is_dna(res_2) or 'H' in res_2.id[0]):
                            # We found a hit
                            rglib_resname = '.'.join([structure_id, str(res_2.get_parent().id), str(res_2.id[1])])
                            found_rna_neighbors.add(rglib_resname)
                if len(found_rna_neighbors) > 0:
                    found_rna_neighbors = sorted(list(found_rna_neighbors))
                    interaction_dict['rna_neighs'] = found_rna_neighbors
                    all_interactions[f"{selected}s"].append(interaction_dict)
    return all_interactions


def add_graph_annotations(g, cif):
    """
    Adds information at the graph level and on the small molecules partner of an RNA molecule

    :param g: the nx graph created from dssr output
    :param cif: the path to a .mmcif file
    :return: the annotated graph, actually the graph is mutated in place
    """
    mmcif_dict = MMCIF2Dict(cif)
    # Add graph level like resolution
    graph_level_annots = get_mmcif_graph_level(mmcif_dict=mmcif_dict)
    g.graph.update(graph_level_annots)

    # Fetch interactions with small molecules and ions
    all_interactions = get_small_partners(cif, mmcif_dict=mmcif_dict)
    g.graph.update(all_interactions)

    # First fill relevant nodes
    for interaction_dict in all_interactions['ligands']:
        ligand_id = interaction_dict['ligand_id']
        for rna_neigh in interaction_dict['rna_neighs']:
            g.nodes[rna_neigh]['binding_small-molecule'] = ligand_id
    for interaction_dict in all_interactions['ions']:
        ligand_id = interaction_dict['ligand_id']
        for rna_neigh in interaction_dict['rna_neighs']:
            g.nodes[rna_neigh]['binding_ion'] = ligand_id

    # Then add a None field in all other nodes
    for node, node_data in g.nodes(data=True):
        if 'binding_ion' not in node_data:
            node_data['binding_ion'] = None
        if 'binding_small-molecule' not in node_data:
            node_data['binding_small-molecule'] = None
    return g


def dangle_trim(G):
    """
    Recursively remove dangling nodes from graph.

    :param G: Networkx graph
    :type G: networkx.Graph
    :return: Trimmed networkx graph
    :rtype: networkx.Graph
    """
    dangles = lambda G: [n for n in G.nodes() if G.degree(n) < 2]
    while dangles(G):
        G.remove_nodes_from(dangles(G))


def reorder_nodes(g):
    """
    Reorder nodes in graph

    :param g: Pass a graph for node reordering.
    :type g: networkx.DiGraph

    :return h: (nx DiGraph)
    """

    h = nx.DiGraph()
    h.add_nodes_from(sorted(g.nodes.data()))
    h.add_edges_from(g.edges.data())
    for key, value in g.graph.items():
        h.graph[key] = value

    return h


def annotate_proteinSSE(g, structure, pdb_file):
    """
    Annotate protein_binding node attributes with the relative SSE
    if available from DSSP

    :param g: (nx graph)
    :param structure: (PDB structure)

    :return g: (nx graph)
    """

    model = structure[0]
    tic = perf_counter()
    dssp = DSSP(model, pdb_file, dssp='mkdssp', file_type='DSSP')
    toc = perf_counter()

    print(dssp.keys())

    a_key = list(dssp.keys())[2]

    print(dssp[a_key])

    print(f'runtime = {tic - toc:0.7f} seconds')

    return g


def load_dssr_graph(json_file):
    """
    load DSSR graph from JSON

    :param json_file: path to json containing DSSR output

    :return: graph from parsed json data
    :rtype: networkx.DiGraph
    """
    pbid = json_file[-9:-5]
    with open(json_file, 'r') as f:
        d = json.load(f)

    g = nx.readwrite.json_graph.node_link_graph(d)

    return g


def write_graph(g, json_file):
    """
    Utility function to write networkx graph to JSON

    :param g: graph to dump
    :type g: networkx.Graph
    :param json_file: path to dump json
    :type json_file: str
    """
    d = nx.readwrite.json_graph.node_link_data(g)
    with open(json_file, 'w') as f:
        json.dump(d, f)

    return


def annotate_graph(g, annots):
    """
    Add node annotations to graph from annots
    nodes without a value receive None type

    :param g: RNA graph to add x3dna data annotations to.
    :type g: networkx.DiGraph
    :param annots: parsed output from x3dna
    :type annots: dict
    :return: graph with updated node and edge data
    :rtype: networkx.Graph
    """

    labels = {'binding_ion': 'ion',
              'binding_small-molecule': 'ligand'}

    for node in g.nodes():
        for label, typ in labels.items():
            try:
                annot = annots[node][typ]
            except KeyError:
                annot = None
            g.nodes[node][label] = annot

    return g


def parse_interfaces(interfaces,
                     types=['ion', 'ligand']):
    """
    Parse output from get_interfaces into a dictionary

    :param interfaces: output from dssr interface annotation
    :param types: which type of molecule to consider in the interface

    :return: dictionary containing interface annotations
    """
    annotations = defaultdict(dict)

    for pbid, chain, typ, target, PDB_pos in interfaces:
        if types:
            if typ not in types: continue
        annotations[str(pbid) + '.' + str(chain) + '.' + str(PDB_pos)][typ] = target

    return annotations


def load_csv_annot(csv_file, pbids=None, types=None):
    """
    Get annotations from csv file, parse into a dictionary

    :param csv_file: csv to read annotations from
    :type csv_file: path-like
    :param pdbids: list of PDBIs to process, if None, all are processed.
    :type pdbids: list
    :param types: only consider annotations for give molecule types ('ion', 'ligand')
    :type types: list

    :return: annotation dictionary
    """
    annotations = defaultdict(dict)
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = True
        for pbid, _, chain, typ, target, PDB_pos in reader:
            if header:
                header = False
                continue
            if pbids:
                if pbid not in pbids: continue
            if types:
                if typ not in types: continue
            annotations[pbid + '.' + chain + '.' + PDB_pos][typ] = target

    return annotations


def annotate_graphs(graph_dir, csv_file, output_dir,
                    ):
    """
    Add annotations from csv_file to all graphs in graph_dir

    :param graph_dir: where to read RNA graphs from
    :type graph_dir: path-like
    :param csv_file: csv containing annotations
    :type graph_dir: path-like
    :param output_dir: where to dump the annotated graphs
    :type output_dir: path-like
    """
    annotations = load_csv_annot(csv_file)

    for graph in os.listdir(graph_dir):
        path = os.path.join(graph_dir, graph)
        g, pbid = load_dssr_graph(path)
        h = annotate_graph(g, annotations)
        write_graph(h, os.path.join(output_dir, graph))


def main():
    # annotate_graphs('../examples/',
    # '../data/interface_list_1aju.csv',
    # '../data/graphs/DSSR/annotated')
    g = load_dssr_graph('../examples/2du5.json')
    # pdb_file = '../data/structures/4gkk.cif'
    # parser = MMCIFParser()
    # structure = parser.get_structure('4GKK', pdb_file)

    # annotate_proteinSSE(g, structure, '../data/structures/4gkk.dssp')

    # h = reorder_nodes(g)
    #
    # print('after reordered:\n', '\n'.join(h.nodes()))
    # print(h.nodes.data())

    return


if __name__ == '__main__':
    main()
