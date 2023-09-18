"""

Build 2.5D graphs using [x3dna DSSR](http://docs.x3dna.org/dssr-manual.pdf).
Requires a x3dna-dssr executable to be in $PATH.

"""
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from collections import defaultdict
from Bio.PDB import *
import json
import networkx as nx
import subprocess
from subprocess import check_output
from loguru import logger
import barnaba as bb

from rnaglib.utils import dump_json
from rnaglib.config import GRAPH_KEYS
from rnaglib.config import EDGE_MAP_RGLIB


logger.add(sys.stderr, level='TRACE')

def barnaba_nuc_id(raw_label, pdbid):
    """ Map a raw barnaba nucleotide ID to a glib format one

    :param raw_label: raw residue label from barnaba
    :param pdbid: pdbid of rna containing of nucleotide
    :returns str: new string with the format <pdbid>.<chain>.<pos>
     """
    # DT_0_T -> pdbid.chain_id.position
    logger.trace(raw_label)
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
    try:
        pdbid = rna_path.stem
        stackings, pairings, res = bb.annotate(rna_path)
        logger.trace(res)
        nts = [r.split("_")[0] for r in res]
        res = [barnaba_nuc_id(r, pdbid) for r in res]
        logger.trace(res)
    except Exception as e:
        logger.error("Opening structure")
        return pdbid, "Loading structure fail"

    basepairs, edge_labels = pairings[0]
    logger.trace(edge_labels)

    G = nx.DiGraph()

    # add coords with biopython
    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure("", rna_path)[0]

    for i, r in enumerate(res):
        try: 
            nuc = GRAPH_KEYS['modified']['barnaba'][nts[i]]
            G.add_node(r, nt_code=nuc, is_modified=True, modification=nts[i])
        except KeyError:
           G.add_node(r, nt_code=GRAPH_KEYS['nt_code']['barnaba'][nts[i]], is_modified=False, modification=None)
    logger.trace(G.nodes(data=True))

    try:
        coord_dict = {}
        for node in G.nodes():
            logger.trace(node)
            chain, pos = node.split(".")[1:]
            r = structure[chain][int(pos)]
            try:
                phos_coord = list(map(float, r['P'].get_coord()))
            except KeyError:
                phos_coord = np.mean([a.get_coord() for a in r], axis=0)
                logger.warning(f"Couldn't find phosphate atom, taking center of atoms in residue instead for {pdbid}.{chain}.{pos} is at {phos_coord}.")
            logger.trace(phos_coord)
            coord_dict[node] = {'xyz_P': list(map(float, phos_coord))}

        nx.set_node_attributes(G, coord_dict)
    except Exception as e:
        logger.exception(f"Failed to get coordinates for {pdbid}, {e}")
        return pdbid, "PDB Coordinate error"

    for (base_1, base_2), label in zip(basepairs, edge_labels):
        logger.trace(f"{res[base_1]} {res[base_2]} {label}")
        elabel = label[-1] + label[0:-1]
        elabel_flip = elabel[0] + elabel[2]  + elabel[1]
        if elabel not in EDGE_MAP_RGLIB:
            continue
        G.add_edge(res[base_1], res[base_2], LW=elabel)
        G.add_edge(res[base_2], res[base_1], LW=elabel_flip)

    bbs = barnaba_bb(res)
    logger.trace(bbs)
    G.add_edges_from(bbs)

    G.graph['pdbid'] = pdbid
    
    logger.trace(G.edges(data=True))

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
