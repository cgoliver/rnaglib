"""

Build 2.5D graphs using [fr3d-python].

"""

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
from collections import defaultdict
from Bio.PDB import MMCIF2Dict, MMCIFParser
import networkx as nx
from loguru import logger

try:
    from fr3d.classifiers.NA_pairwise_interactions import generatePairwiseAnnotation_import
except ImportError:
    print("Missing fr3d installation, pip install\
          git+https://github.com/cgoliver/fr3d-python.git")

from rnaglib.utils import dump_json
from rnaglib.config import GRAPH_KEYS
from rnaglib.config import EDGE_MAP_RGLIB, EDGE_MAP_RGLIB_WITH_STACKING

from rnaglib.config import get_modifications_cache

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
        [r for r in chain.get_residues() if r.id[0] == " " or r.id[0][2:] in XNA_linking], key=lambda x: (x.id[1], x.id[2])
    )


def rna_letters_3to1(three_letter_code: str) -> str:
    """Convert RNA nucleic acid `three_letter_code` to `one_letter_code`.

    Args:
        three_letter_code (str): Three letter code to check.
    Returns:
    str: one_letter_code of RNA nucleic acid, "N" if cannot be found.
    """
    return modifications["rna"].get(three_letter_code, "N")


def _res_node_id(pdbid, chain_id, res):
    icode = res.id[2].strip()
    return f"{pdbid}.{chain_id}.{res.id[1]}{icode}"


def _sym_code(unit_id):
    parts = unit_id.split("|")
    return parts[8] if len(parts) > 8 else "1_555"


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
            fivep_name = _res_node_id(pdbid, chain.id, five_p) 
            threep_name = _res_node_id(pdbid, chain.id, three_p)

            # Use the label sequence ids (and not the author sequence ids) to compute backbone (for the purpose of sequence alignment, consecutive residues might have non-consecutive author seq ids)
            five_p_lsi = five_p.xtra.get("label_seq_id")
            three_p_lsi = three_p.xtra.get("label_seq_id")
            if five_p_lsi is None or three_p_lsi is None:
                continue
            if int(five_p_lsi) == (int(three_p_lsi) + 1):
                bb.append((fivep_name, threep_name, {"LW": "B53"}))
                bb.append((threep_name, fivep_name, {"LW": "B35"}))

                nt_types[fivep_name] = rna_letters_3to1(five_p.get_resname())
                nt_types[threep_name] = rna_letters_3to1(three_p.get_resname())

                nt_types_full[fivep_name] = five_p.get_resname()
                nt_types_full[threep_name] = three_p.get_resname()

    return bb, nt_types, nt_types_full


def nt_to_rgl(nt, pdbid):
    parts = nt.split("|")
    chain = parts[2]
    pos = parts[4]
    icode = parts[7].strip() if len(parts) > 7 else ""
    return f"{pdbid.lower()}.{chain}.{pos}{icode}"


_FORGI_HELPER = Path(__file__).parent / "_forgi_helper.py"


def _forgi_annotate_sse(G, rna_path, pdbid):
    """Annotate each node with its secondary structure element type using forgi.

    Requires the FORGI_PYTHON environment variable to point to a Python
    interpreter with forgi installed (a separate env with numpy<2).
    Sets 'secondary_element' to one of: 'stem', 'hairpin', 'internal',
    'junction', 'five_prime_end', 'three_prime_end', or None.
    """
    forgi_python = os.environ.get("FORGI_PYTHON")
    if forgi_python is None:
        logger.warning("FORGI_PYTHON not set, skipping secondary_element annotation")
        nx.set_node_attributes(G, None, "secondary_element")
        return

    try:
        proc = subprocess.run(
            [forgi_python, str(_FORGI_HELPER), str(rna_path), pdbid],
            capture_output=True,
            text=True,
            check=True,
        )
        node_types = json.loads(proc.stdout)
    except Exception as e:
        logger.warning(f"forgi annotation failed for {pdbid}: {e}")
        nx.set_node_attributes(G, None, "secondary_element")
        return

    secondary_element = {node: None for node in G.nodes()}
    for node, elem_type in node_types.items():
        if node in secondary_element:
            secondary_element[node] = elem_type
    nx.set_node_attributes(G, secondary_element, "secondary_element")


def fr3d_to_graph(rna_path, atom_coords_to_store=["P"], include_stacking=False):
    """Use fr3d to generate networkx annotation graph.

    :param rna_path: path to a PDB of the RNA structure
    :returns nx.Graph: networkx graph with annotations
    """
    rna_path = Path(rna_path)
    try:
        category = "basepair,stacking" if include_stacking else "basepair"
        annot_df = generatePairwiseAnnotation_import(rna_path, category="basepair,stacking")
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

    # Build an author id -> label id mapping
    auth_asym  = struc_dict["_atom_site.auth_asym_id"]
    auth_seq   = struc_dict["_atom_site.auth_seq_id"]
    label_seq  = struc_dict["_atom_site.label_seq_id"]
    ins_codes  = struc_dict.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(auth_seq))

    label_map = {}
    for ca, cs, ls, ic in zip(auth_asym, auth_seq, label_seq, ins_codes):
        icode = " " if ic in (".", "?") else ic
        key = (ca, int(cs), icode)
        label_map[key] = None if ls in (".", "?") else int(ls)
    
    # Attach label_seq_id to each residue's .xtra dict
    for chain in structure.get_chains():
        for residue in chain:
            _, resseq, icode = residue.id
            key = (chain.id, resseq, icode)
            print(f"label_map.keys()={label_map.keys()}")
            residue.xtra["label_seq_id"] = label_map[key]

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
            chain, pos_str = node.split(".")[1:]
            icode = pos_str[-1] if pos_str[-1].isalpha() else ""
            seqnum = int(pos_str[:-1] if icode else pos_str)
            try:
                r = structure[chain][(" ", seqnum, icode if icode else " ")]
            except KeyError:
                for res in structure[chain]:
                    if res.id[1] == seqnum and res.id[2].strip() == icode:
                        r = res
            centroid_coord = np.mean([a.get_coord() for a in r], axis=0)
            coord_dict[node] = {f"xyz_centroid": list(map(float, centroid_coord))}
            if atom_coords_to_store == ["all_atom"]:
                heavy_atoms = {
                    a.get_name(): list(map(float, a.get_coord()))
                    for a in r
                    if a.element != "H" and a.element.strip() != "H"
                }
                coord_dict[node]["heavy_atoms"] = heavy_atoms
            else:
                for atom_type in atom_coords_to_store:
                    try:
                        atom_coord = list(map(float, r[atom_type].get_coord()))
                    except KeyError:
                        atom_coord = None
                        logger.warning(f"Couldn't find {atom_type} atom")
                    logger.debug(f"{node} {atom_coord}")
                    coord_dict[node][f"xyz_{atom_type}"] = atom_coord
            nx.set_node_attributes(G, coord_dict)

    except Exception as e:
        logger.exception(f"Failed to get coordinates for {pdbid}, {e}")
        return None

    for pair in annot_df.itertuples():

        # Discard pairs between residues having different symmetry codes
        if _sym_code(pair.source) != _sym_code(pair.target):
            continue

        elabel = pair.interaction
        if elabel not in EDGE_MAP_RGLIB_WITH_STACKING:
            continue
        
        nt1 = nt_to_rgl(pair.source, pdbid)
        nt2 = nt_to_rgl(pair.target, pdbid)

        if G.has_node(nt1) and G.has_node(nt2):
            G.add_edge(nt1, nt2, LW=elabel)


    _forgi_annotate_sse(G, rna_path, pdbid)

    G.graph["name"] = pdbid
    G.graph["pdbid"] = pdbid

    return G


if __name__ == "__main__":
    # doc example with multiloop
    # build_one("../data/1aju.cif")
    # multi chain
    build_one("../data/structures/1fmn.cif")
