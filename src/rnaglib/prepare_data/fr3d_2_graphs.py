"""

Build 2.5D graphs using [fr3d-python].

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

try:
    from fr3d.classifiers.NA_pairwise_interactions import generatePairwiseAnnotation_import
except ImportError:
    print("Missing fr3d installation, pip install\
          git+https://github.com/cgoliver/fr3d-python.git")

from rnaglib.utils import dump_json
from rnaglib.config import GRAPH_KEYS
from rnaglib.config import EDGE_MAP_RGLIB

from rnaglib.config import get_modifications_cache

from .annotations import add_graph_annotations

logger.remove()
logger.add(sys.stderr, level="INFO")

modifications = get_modifications_cache()


def get_rna_chains(mmcif_dict):
    """Return the list of RNA Chain IDs."""
    rna_chains = [
        chain
        for chain, chain_type in zip(mmcif_dict["_entity_poly.pdbx_strand_id"], mmcif_dict["_entity_poly.type"])
        if chain_type == "polyribonucleotide"
    ]
    cleaned = []
    for r in rna_chains:
        sub = r.split(",")
        cleaned.extend([s for s in sub])
    return cleaned


def nuc_id(raw_label):
    """Map a raw fr3d nucleotide ID to a glib format one

    :param raw_label: raw residue label from fr3d
    :param pdbid: pdbid of rna containing of nucleotide
    :returns str: new string with the format <pdbid>.<chain>.<pos>
    """
    # 3OX0|1|A|C|70 -> 3ox0.A.70
    logger.trace(raw_label)
    pdbid, _, chain, _, pos = raw_label.split("|")
    return f"{pdbid.lower()}.{chain}.{pos}"


def get_residue_list(chain, XNA_linking):
    # return sorted([r for r in chain if r.id[0] == ' '], key=lambda x: x.id[1])
    return sorted(
        [r for r in chain.get_residues() if r.id[0] == " " or r.id[0][2:] in XNA_linking], key=lambda x: x.id[1]
    )


def rna_letters_3to1(three_letter_code: str) -> str:
    """Convert RNA nucleic acid `three_letter_code` to `one_letter_code`.

    Args:
        three_letter_code (str): Three letter code to check.
    Returns:
    str: one_letter_code of RNA nucleic acid, "N" if cannot be found.
    """
    return modifications["rna"].get(three_letter_code, "N")


def get_bb(structure, rna_chains, XNA_linking, pdbid=""):
    """Get the backbone edges"""
    bb = []
    nt_types = {}
    nt_types_full = {}
    for chain in structure.get_chains():
        if chain.id not in rna_chains:
            continue
        # reslist = get_residue_list(structure, chain)
        reslist = get_residue_list(chain, XNA_linking)
        logger.debug(reslist)

        for i, five_p in enumerate(reslist):
            if i == 0:
                continue
            three_p = reslist[i - 1]
            if int(five_p.id[1]) == (int(three_p.id[1]) + 1):
                bb.append((f"{pdbid}.{chain.id}.{five_p.id[1]}", f"{pdbid}.{chain.id}.{three_p.id[1]}", {"LW": "B53"}))
                bb.append((f"{pdbid}.{chain.id}.{three_p.id[1]}", f"{pdbid}.{chain.id}.{five_p.id[1]}", {"LW": "B35"}))

                nt_types[f"{pdbid}.{chain.id}.{five_p.id[1]}"] = rna_letters_3to1(five_p.get_resname())
                nt_types[f"{pdbid}.{chain.id}.{three_p.id[1]}"] = rna_letters_3to1(three_p.get_resname())

                nt_types_full[f"{pdbid}.{chain.id}.{five_p.id[1]}"] = five_p.get_resname()
                nt_types_full[f"{pdbid}.{chain.id}.{three_p.id[1]}"] = three_p.get_resname()
    return bb, nt_types, nt_types_full


def nt_to_rgl(nt, pdbid):
    _, _, chain, _, pos = nt.split("|")[:5]
    return f"{pdbid.lower()}.{chain}.{pos}"


def fr3d_to_graph(rna_path):
    """Use fr3d to generate networkx annotation graph.

    :param rna_path: path to a PDB of the RNA structure
    :returns nx.Graph: networkx graph with annotations
    """
    rna_path = Path(rna_path)
    try:
        annot_df = generatePairwiseAnnotation_import(rna_path, category="basepair")
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

    # load mmCIF structure
    struc_dict = MMCIF2Dict.MMCIF2Dict(rna_path)

    # find all XNA linking, including standard and non-standard
    chem_comp = {}
    chem_comp["chem_code"] = struc_dict["_chem_comp.id"]
    chem_comp["chem_type"] = struc_dict["_chem_comp.type"]
    XNA_linking = [
        chem_comp["chem_code"][idx]
        for idx, tp in enumerate(chem_comp["chem_type"])
        if tp == "RNA linking" or tp == "DNA linking"
    ]
    # XNA_linking = [chem_comp['chem_code'][idx] for idx, tp in enumerate(chem_comp['chem_type']) if tp =='RNA linking']

    # add coords with biopython
    parser = MMCIFParser()
    structure = parser.get_structure("", rna_path)[0]

    # bbs, nt_types = get_bb(structure, rna_chains, pdbid=pdbid)
    bbs, nt_types, nt_types_full = get_bb(structure, rna_chains, XNA_linking, pdbid=pdbid)
    # print(f"rna chain residues: {nt_types}")
    logger.trace(bbs)
    G = nx.MultiDiGraph()
    G.add_edges_from(bbs)

    nx.set_node_attributes(G, nt_types, "nt")
    nx.set_node_attributes(G, nt_types_full, "nt_full")

    for node in G.nodes():
        G.nodes[node]["nt_code"] = nt_types[node]
        G.nodes[node]["nt_full"] = nt_types_full[node]
        G.nodes[node]["chain_id"] = node.split(".")[1]
        G.nodes[node]["is_modified"] = len(nt_types_full[node]) != 1

    try:
        coord_dict = {}
        for node in G.nodes():
            chain, pos = node.split(".")[1:]
            try:
                r = structure[chain][int(pos)]  # in this index-way only standard residue got
            except KeyError:
                for res in structure[chain]:
                    if int(pos) == res.id[1]:  # got non-standard residue
                        r = res
            try:
                phos_coord = list(map(float, r["P"].get_coord()))
            except KeyError:
                phos_coord = np.mean([a.get_coord() for a in r], axis=0)
                logger.warning(
                    f"Couldn't find phosphate atom, taking center of atoms in residue instead for {pdbid}.{chain}.{pos} is at {phos_coord}."
                )
            logger.debug(f"{node} {phos_coord}")
            coord_dict[node] = {"xyz_P": list(map(float, phos_coord))}
        nx.set_node_attributes(G, coord_dict)
    except Exception as e:
        logger.exception(f"Failed to get coordinates for {pdbid}, {e}")
        return None

    for pair in annot_df.itertuples():
        # logger.trace(f"{pair.from} {pair.to} {pair.interaction}")
        elabel = pair.interaction
        elabel_flip = elabel[0] + elabel[2] + elabel[1]
        if elabel not in EDGE_MAP_RGLIB:
            continue
        nt1 = nt_to_rgl(pair.source, pdbid)
        nt2 = nt_to_rgl(pair.target, pdbid)
        # G.add_edge(nt1,nt2 , LW=elabel)
        # G.add_edge(nt2, nt1, LW=elabel_flip)
        # avoid getting nodes with no attributes
        if G.has_node(nt1) and G.has_node(nt2):
            G.add_edge(nt1, nt2, LW=elabel)
            G.add_edge(nt2, nt1, LW=elabel_flip)

    G.graph["name"] = pdbid
    G.graph["pdbid"] = pdbid

    return G


if __name__ == "__main__":
    # doc example with multiloop
    # build_one("../data/1aju.cif")
    # multi chain
    build_one("../data/structures/1fmn.cif")
