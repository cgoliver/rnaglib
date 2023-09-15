"""

Build 2.5D graphs using [x3dna DSSR](http://docs.x3dna.org/dssr-manual.pdf).
Requires a x3dna-dssr executable to be in $PATH.

"""
import os
import traceback
from pathlib import Path

from collections import defaultdict
from Bio.PDB import *
import json
import networkx as nx
import subprocess
from subprocess import check_output
from loguru import logger
import barnaba as bb

from rnaglib.utils import dump_json
from rnaglib.utils import reorder_nodes 
from rnaglib.prepare_data import add_graph_annotations
from rnaglib.prepare_data import filter_dot_edges
from rnaglib.config import GRAPH_KEYS
from rnaglib.config import EDGE_MAP_RGLIB

def snap_exec(cif):
    """Execute x3dna in SNAP mode to analyze protein interfaces.

    :param cif: path to mmCIF

    :return: plaintext output
    """
    try:
        rpb_dict = check_output(["x3dna-dssr", "snap", f"-i={cif}"], stderr=subprocess.DEVNULL)
    except Exception as e:
        print(e)
        return 1, None
    return 0, rpb_dict.decode("utf-8")


def snap_parse(snap_out):
    """
    SNAP output is raw text so we have to parse it.

    :param snap_out: raw output from SNAP

    :return: dictionary of data for each residue in interface

    """
    import re

    lines = iter(snap_out.split("\n"))

    # sometimes header is missing so we have to do this
    header = ["id", "nt-aa", "nt", "aa", "Tdst", "Rdst", "Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]

    # regex for base-amino acid interaction
    base_aa = re.compile("[AUCG]{1}-[a-z]{3}\s")
    interface_nts = dict()
    for i, l in enumerate(lines):
        # get rid of first two columns
        if base_aa.search(l):
            l = l.split()[2:]
            nt_id = l[1]
            interface_nts[nt_id] = dict(zip(header, l))

    return interface_nts


def find_nt(dssr_dict_nt, nt_id):
    """Find a nucleotide ID in DSSR dictionary.

    :param dssr_dict_nt: dict of annotated nucleotide objects
    :param nt_id: nucleotide ID we seek.
    """
    for nt in dssr_dict_nt:
        if nt['nt_id'] == nt_id:
            return nt


def rna_only_nts(dssr_dict):
    """
    Filter DSSR output to only keep RNA.

    :param: DSSR dictionary

    :return: filtered dictionay
    """
    return filter(lambda x: x['nt_type'] == 'RNA', dssr_dict['nts'])


def rna_only_pairs(dssr_dict):
    """
    Only keep pairs between RNAs.

    :param dssr_dict: dssr output dictionary

    :return: filtered dssr output dictionary
    """
    return filter(lambda x: find_nt(dssr_dict['nts'], x['nt1'])['nt_type'] == 'RNA' and \
                            find_nt(dssr_dict['nts'], x['nt2'])['nt_type'] == 'RNA', \
                  dssr_dict['pairs'])


def get_backbones(nts):
    """ Get backbone pairs.

    :param nts: DSSR nucleotide info.

    :return: list of tuples (5' base, 3' base)
    """
    bb = []
    for i, three_p in enumerate(nts):
        if i == 0:
            continue
        five_p = nts[i - 1]
        if five_p['chain_name'] != three_p['chain_name']:
            continue
        if three_p['nt_type'] != 'RNA' or five_p['nt_type'] != 'RNA':
            continue
        if 'break' not in three_p['summary']:
            bb.append((five_p, three_p))
    return bb


def add_sses(g, dssr_dict_nts):
    """
    Return dict of nodes that are in an sse as a list of annotations.

    :param g: networkx graph
    :param dssr_dict_nts: dssr dictionary

    :return: dictionary containing annotations with SSE info.
    """
    sse_nt_dict = dict()
    sse_types = ['hairpins', 'junctions', 'bulges', 'internal']
    for sse in sse_types:
        try:
            elements = dssr_dict_nts[sse]
        except KeyError:
            continue
        for elem in elements:
            for nt in elem['nts_long'].split(','):
                if nt in g.nodes():
                    sse_nt_dict[nt] = {'sse': f'{sse[:-1]}_{elem["index"]}'}
    return sse_nt_dict


def base_pair_swap(pairs):
    """Swap the order of the entries in a pair dict for bidirectional edges.

    For now not swapping 'Saenger' and 'DSSR'.

    :param pairs: list of pair of edges
    """
    new_pairs = []
    for pair in pairs:
        new_dict = dict(pair)
        new_dict['nt1'] = pair['nt2']
        new_dict['nt2'] = pair['nt1']
        new_dict['bp'] = pair['bp'][2] + pair['bp'][1] + pair['bp'][0]
        new_dict['LW'] = pair['LW'][0] + pair['LW'][:0:-1]
        new_pairs.append(new_dict)

    return pairs + new_pairs


def get_graph_level_infos(dssr_dict):
    """ Fetch graph-level data

    :param dssr_dict: Dssr output dictionary
    """

    def recursive_dd():
        return defaultdict(recursive_dd)

    g_data = {'dbn': recursive_dd()}

    for chain, info in dssr_dict['dbn'].items():
        if chain == 'all_chains':
            g_data['dbn']['all_chains'] = info
        else:
            g_data['dbn']['single_chains'][chain] = info
        pass
    return g_data


def dssr_dict_2_graph(dssr_dict, rbp_dict, pdbid):
    """
    DSSR Annotation JSON Keys:

        dict_keys(['num_pairs', 'pairs', 'num_helices', 'helices',
        'num_stems', 'stems', 'num_coaxStacks', 'coaxStacks', 'num_stacks',
        'stacks', 'nonStack', 'num_atom2bases', 'atom2bases', 'num_hairpins',
        'hairpins', 'num_bulges', 'bulges', 'num_splayUnits', 'splayUnits',
        'dbn', 'chains', 'num_nts', 'nts', 'num_hbonds', 'hbonds',
        'refCoords', 'metadata']

    :param dssr_dict: dictionary from dssr
    :param rbp_annt: interface dicitonary

    :return: graph containing all annotations
    """

    # First, include the graph level dbn annotations from dssr
    G = nx.DiGraph()
    graph_level_infos = get_graph_level_infos(dssr_dict)
    G.graph.update(graph_level_infos)
    nt_dict = rna_only_nts(dssr_dict)

    # add nucleotides
    G.add_nodes_from(((d['nt_id'], d) for d in nt_dict))

    # add backbones
    bbs = get_backbones(dssr_dict['nts'])
    G.add_edges_from(((five_p['nt_id'], three_p['nt_id'], {'LW': 'B53', 'backbone': True}) for five_p, three_p in bbs))
    G.add_edges_from(((three_p['nt_id'], five_p['nt_id'], {'LW': 'B35', 'backbone': True}) for five_p, three_p in bbs))

    # add base pairs
    try:
        rna_pairs = rna_only_pairs(dssr_dict)
        rna_pairs = base_pair_swap(list(rna_pairs))
    except Exception as e:
        # print(e)
        # traceback.print_exc()
        # print(f">>> No base pairs found for {pdbid}")
        return

    G.add_edges_from(((pair['nt1'], pair['nt2'], pair) for pair in rna_pairs))

    # add SSE data
    sse_nodes = add_sses(G, dssr_dict)
    for node in G.nodes():
        try:
            G.nodes[node]['sse'] = sse_nodes[node]
        except KeyError:
            G.nodes[node]['sse'] = {'sse': None}

    new_labels = {n: ".".join([pdbid, str(d['chain_name']), str(d['nt_resnum'])]) for n, d in G.nodes(data=True)}
    G = nx.relabel_nodes(G, new_labels)

    # Relabel the dict to include it at both the node and the graph level
    rbp_dict_relabeled = {}
    for node, interaction in rbp_dict.items():
        try:
            rbp_dict_relabeled[new_labels[node]] = interaction
        except KeyError:
            pass
    # add RNA-Protein interface data in the nodes
    for node in G.nodes():
        try:
            G.nodes[node]['binding_protein'] = rbp_dict_relabeled[node]
        except KeyError:
            G.nodes[node]['binding_protein'] = None
    # add RNA-Protein interface data in the graph
    G.graph['proteins'] = list(rbp_dict_relabeled.keys())
    return G


def one_rna_from_cif(cif):
    """
    Build 2.5d graph for one cif using dssr

    :param cif: path to mmCIF

    :return: 2.5d graph
    """
    exit_code, dssr_dict = dssr_exec(cif)
    if exit_code == 1:
        return None
    rbp_exit_code, rbp_out = snap_exec(cif)
    try:
        rbp_dict = snap_parse(rbp_out)
    except:
        rbp_dict = {}
    pdbid = os.path.basename(cif).split(".")[0]
    G = dssr_dict_2_graph(dssr_dict, rbp_dict, pdbid)
    return G

def cif_to_graph(cif, output_dir=None, min_nodes=20, return_graph=False):
    """
    Build DDSR graphs for one RNA. 

    :param cif: path to CIF
    :param output_dir: where to dump
    :param min_nodes: smallest RNA (number of residue nodes)
    :param annotator: which annotation tool to use ('barnaba' or 'fr3d')
    :param return_graph: Boolean to include the graph in the output

    :return: networkx graph of structure.
    """

    if '.cif' not in cif:
        # print("Incorrect format")
        return os.path.basename(cif), 'format'
    pdbid = cif[-8:-4]
    # print('Computing Graph for ', pdbid)

    # Build graph with DSSR
    error_type = 'OK'
    try:
        dssr_failed = False
        g = one_rna_from_cif(cif)
        dssr_failed = g is None
        filter_dot_edges(g)
    except Exception as e:
        print("ERROR: Could not construct DSSR graph for ", cif)
        print(e)
        if dssr_failed:
            # print("Annotation using x3dna-dssr failed, please ensure you have the executable in your PATH")
            # print("This requires a license.")
            error_type = 'DSSR_error'
        else:
            # print(traceback.print_exc())
            error_type = 'Filtering error after DSSR building'
        return pdbid, error_type

    if len(g.nodes()) < min_nodes:
        # print(f'Excluding {pdbid} from output, less than 20 nodes')
        error_type = 'tooSmall'
        return pdbid, error_type
    if len(g.edges()) < len(g.nodes()) - 3:
        # print(f'Excluding {pdbid} from output, edges < nodes -3')
        error_type = 'edges<nodes-3'
        return pdbid, error_type

    # Find ligand and ion annotations from the PDB cif
    try:
        add_graph_annotations(g=g, cif=cif)
    except Exception as e:
        print('ERROR: Could not compute interfaces for ', cif)
        print(e)
        print(traceback.print_exc())
        error_type = 'interfaces_error'
    # Order the nodes
    g = reorder_nodes(g)

    # Write graph to outputdir in JSON format
    if output_dir is not None:
        dump_json(os.path.join(output_dir, 'graphs', pdbid + '.json'), g)
    if return_graph:
        return pdbid, error_type, g
    return pdbid, error_type

def barnaba_nuc_id(raw_label, pdbid):
    """ Map a raw barnaba nucleotide ID to a glib format one

    :param raw_label: raw residue label from barnaba
    :param pdbid: pdbid of rna containing of nucleotide
    :returns str: new string with the format <pdbid>.<chain>.<pos>
     """
    # DT_0_T -> pdbid.chain_id.position
    logger.debug(raw_label)
    nuc_type, pos, chain = raw_label.split("_")
    return f"{pdbid.lower()}.{chain}.{pos}"

def barnaba_pair_type(raw_label):
    """ Map a raw barnaba nucleotide ID to a glib format one
    :param pdbid: pdbid of rna containing of nucleotide
    :param raw_label: raw residue label from barnaba
    :returns str: new string with the format <pdbid>.<chain>.<pos>
     """
    # DT_0_T -> pdbid.chain_id.position
    nuc_type, pos, chain = raw_label.split()
    return f"{pdbid.lower()}.{chain}.{pos}"


def barnaba_pair_id(raw_label):
    return label

def barnaba_bb(res):
    """ Get the backbone edges 
    """
    bb = []
    for i, three_p in enumerate(res):
        if i == 0:
            continue
        five_p = res[i - 1]
        five_p_chain, five_p_pos = five_p.split(".")[1:]
        three_p_chain, three_p_pos = five_p.split(".")[1:]
        # different chains
        if five_p_chain != three_p_chain:
            continue
        if int(five_p_pos) != (int(three_p_pos) + 1):
            bb.append((five_p, three_p, {'LW': 'B53'}))
            bb.append((three_p, five_p, {'LW': 'B35'}))
    return bb

def barnaba_to_graph(rna_path, output_dir=None, return_graph=False,):
    """ Use barnaba to generate networkx annotation graph.

    :param rna_path: path to a PDB of the RNA structure
    :returns nx.Graph: networkx graph with annotations
    """
    pdbid = rna_path.stem
    stackings, pairings, res = bb.annotate(rna_path)
    nts = [r.split("_")[0] for r in res]
    res = [barnaba_nuc_id(r, pdbid) for r in res]
    logger.debug(res)

    basepairs, edge_labels = pairings[0]
    logger.debug(edge_labels)

    G = nx.DiGraph()

    # add coords with biopython
    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure("", rna_path)[0]

    G.add_nodes_from([(r, {'nt_code': GRAPH_KEYS['nt_code']['barnaba'][nts[i]]}) for i, r in  enumerate(res)])
    logger.debug(G.nodes(data=True))

    try:
        coord_dict = {}
        for node in G.nodes():
            logger.info(node)
            chain, pos = node.split(".")[1:]
            r = structure[chain][int(pos)]
            phos_coord = list(map(float, r['P'].get_coord()))
            coord_dict[node] = {'xyz_P': phos_coord}

        nx.set_node_attributes(G, coord_dict)
    except Exception as e:
        logger.error(f"Failed to get coordinates for {pdbid}")
        return pdbid, "PDB Coordinate error"

    for (base_1, base_2), label in zip(basepairs, edge_labels):
        logger.debug(f"{res[base_1]} {res[base_2]} {label}")
        elabel = label[-1] + label[0:-1]
        elabel_flip = elabel[0] + elabel[2]  + elabel[1]
        if elabel not in EDGE_MAP_RGLIB:
            continue
        G.add_edge(res[base_1], res[base_2], LW=elabel)
        G.add_edge(res[base_2], res[base_1], LW=elabel_flip)

    bbs = barnaba_bb(res)
    logger.debug(bbs)
    G.add_edges_from(bbs)

    G.graph['pdbid'] = pdbid
    
    logger.debug(G.edges(data=True))

    if output_dir is not None:
        dump_json(os.path.join(output_dir, 'graphs', pdbid + '.json'), G)
    if return_graph:
        return pdbid, error_type, G
    return pdbid, None


if __name__ == "__main__":
    # doc example with multiloop
    # build_one("../data/1aju.cif")
    # multi chain
    build_one("../data/structures/1fmn.cif")
