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

from fr3d.classifiers.NA_pairwise_interactions import generatePairwiseAnnotation_import

from rnaglib.utils import dump_json
from rnaglib.config import GRAPH_KEYS
from rnaglib.config import EDGE_MAP_RGLIB


logger.remove()
logger.add(sys.stderr, level='INFO')

def get_rna_chains(mmcif_dict):
    """ Return the list of RNA Chain IDs.
    """
    rna_chains = [chain for chain, chain_type in zip(mmcif_dict['_entity_poly.pdbx_strand_id'], mmcif_dict['_entity_poly.type']) if chain_type == 'polyribonucleotide']
    cleaned = []
    for r in rna_chains:
        sub = r.split(",")
        cleaned.extend([s.upper() for s in sub])
    return cleaned

def nuc_id(raw_label):
    """ Map a raw fr3d nucleotide ID to a glib format one

    :param raw_label: raw residue label from fr3d 
    :param pdbid: pdbid of rna containing of nucleotide
    :returns str: new string with the format <pdbid>.<chain>.<pos>
     """
    # 3OX0|1|A|C|70 -> 3ox0.A.70
    logger.trace(raw_label)
    pdbid,_, chain, _, pos = raw_label.split("|")
    return f"{pdbid.lower()}.{chain}.{pos}"

def get_residue_list(structure, chain):
    return sorted([r for r in chain if r.id[0] == ' '], key=lambda x: x.id[1])

def get_bb(structure, rna_chains, pdbid=''):
    """ Get the backbone edges 
    """
    bb = []
    nt_types = {}
    print(f"Using {rna_chains}")
    for chain in structure.get_chains():
        if chain.id not in rna_chains:
            continue 
        reslist = get_residue_list(structure, chain)
        logger.debug(reslist)

        for i, five_p in enumerate(reslist):
            if i == 0:
                continue
            three_p = reslist[i - 1]
            if int(five_p.id[1]) == (int(three_p.id[1]) + 1):
                bb.append((f"{pdbid}.{chain.id}.{five_p.id[1]}", f"{pdbid}.{chain.id}.{three_p.id[1]}", {'LW': 'B53'}))
                bb.append((f"{pdbid}.{chain.id}.{three_p.id[1]}", f"{pdbid}.{chain.id}.{five_p.id[1]}", {'LW': 'B35'}))

                nt_types[f"{pdbid}.{chain.id}.{five_p.id[1]}"] = five_p.get_resname()
                nt_types[f"{pdbid}.{chain.id}.{three_p.id[1]}"] = three_p.get_resname()
    return bb, nt_types

def nt_to_rgl(nt, pdbid):
    _,_, chain, _, pos = nt.split("|")[:5]
    return f"{pdbid.lower()}.{chain}.{pos}"

def fr3d_to_graph(rna_path):
    """ Use fr3d to generate networkx annotation graph.

    :param rna_path: path to a PDB of the RNA structure
    :returns nx.Graph: networkx graph with annotations
    """
    rna_path = Path(rna_path)
    try:
        annot_df = generatePairwiseAnnotation_import(rna_path, category='basepair')
    except Exception as e:
        logger.exception(f"Fr3D error {rna_path}")
        return None

    pdbid = rna_path.stem.lower()
    try: 
        rna_chains = get_rna_chains(MMCIF2Dict.MMCIF2Dict(rna_path))
        logger.debug(f"RNA chains in {pdbid}: {rna_chains}")
    except KeyError:
        logger.error(f"Couldn't identify RNA chains in {pdbid}")
        return None


    # add coords with biopython
    parser = MMCIFParser()
    structure = parser.get_structure("", rna_path)[0]

    bbs, nt_types = get_bb(structure, rna_chains, pdbid=pdbid)
    logger.trace(bbs)
    G = nx.DiGraph()
    G.add_edges_from(bbs)

    nx.set_node_attributes(G, nt_types, 'nt')

    try:
        coord_dict = {}
        for node in G.nodes():
            chain, pos = node.split(".")[1:]
            r = structure[chain][int(pos)]
            try:
                phos_coord = list(map(float, r['P'].get_coord()))
            except KeyError:
                phos_coord = np.mean([a.get_coord() for a in r], axis=0)
                logger.warning(f"Couldn't find phosphate atom, taking center of atoms in residue instead for {pdbid}.{chain}.{pos} is at {phos_coord}.")
            logger.debug(f"{node} {phos_coord}")
            coord_dict[node] = {'xyz_P': list(map(float, phos_coord))}
        nx.set_node_attributes(G, coord_dict)
    except Exception as e:
        logger.exception(f"Failed to get coordinates for {pdbid}, {e}")
        return None

    for pair in annot_df.itertuples():
        # logger.trace(f"{pair.from} {pair.to} {pair.interaction}")
        elabel = pair.interaction 
        elabel_flip = elabel[0] + elabel[2]  + elabel[1]
        if elabel not in EDGE_MAP_RGLIB:
            continue
        nt1 = nt_to_rgl(pair.source, pdbid) 
        nt2 = nt_to_rgl(pair.target, pdbid) 
        G.add_edge(nt1,nt2 , LW=elabel)
        G.add_edge(nt2, nt1, LW=elabel_flip)

    G.graph['pdbid'] = pdbid
    
    return G


if __name__ == "__main__":
    # doc example with multiloop
    # build_one("../data/1aju.cif")
    # multi chain
    build_one("../data/structures/1fmn.cif")
