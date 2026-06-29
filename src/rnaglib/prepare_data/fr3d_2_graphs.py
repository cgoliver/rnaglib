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


def _parse_oper_list(mmcif_dict):
    """Return {sym_name: (R, t)} from _pdbx_struct_oper_list.

    The PDB stores these operators in Cartesian (Angstrom) space, so R and t
    can be applied directly to BioPython coordinates without any cell conversion.
    """
    names = mmcif_dict.get("_pdbx_struct_oper_list.name", [])
    operators = {}
    for i, name in enumerate(names):
        R = np.array([
            [float(mmcif_dict[f"_pdbx_struct_oper_list.matrix[{r}][{c}]"][i]) for c in range(1, 4)]
            for r in range(1, 4)
        ])
        t = np.array([float(mmcif_dict[f"_pdbx_struct_oper_list.vector[{r}]"][i]) for r in range(1, 4)])
        operators[name] = (R, t)
    return operators


def _collect_sym_needs(mmcif_dict, rna_chains):
    """Return {sym_name: set(auth_chain_id)} from _pdbx_struct_assembly_gen.

    Uses the PDB biological assembly definition as the ground truth for which
    chains receive symmetry copies. The identity operator is skipped since those
    chains are already in the file. Only RNA chains are included.

    Note: _pdbx_struct_assembly_gen.oper_expression references
    _pdbx_struct_oper_list.id values (not operator names); asym_id_list uses
    label_asym_id which is mapped to auth_asym_id via the atom_site table.
    Structures where the relevant operator is absent from _pdbx_struct_oper_list
    (e.g. pure crystal-packing contacts) are silently skipped since no Cartesian
    matrix is available for them.
    """
    label_to_auth = {}
    for l, a in zip(
        mmcif_dict.get("_atom_site.label_asym_id", []),
        mmcif_dict.get("_atom_site.auth_asym_id", []),
    ):
        label_to_auth.setdefault(l, a)

    oper_ids   = mmcif_dict.get("_pdbx_struct_oper_list.id", [])
    oper_names = mmcif_dict.get("_pdbx_struct_oper_list.name", [])
    oper_types = mmcif_dict.get("_pdbx_struct_oper_list.type", [])
    id_to_name = {}
    identity_ids = set()
    for oid, oname, otype in zip(oper_ids, oper_names, oper_types):
        id_to_name[oid] = oname
        if otype == "identity operation":
            identity_ids.add(oid)

    needed = defaultdict(set)
    for oper_expr, asym_list in zip(
        mmcif_dict.get("_pdbx_struct_assembly_gen.oper_expression", []),
        mmcif_dict.get("_pdbx_struct_assembly_gen.asym_id_list", []),
    ):
        if "(" in oper_expr:
            # Product-notation expressions (e.g. "(1-60)(61)") encode point-group
            # assemblies; skip them as they are not relevant for crystal contacts.
            logger.debug(f"Skipping product oper_expression: {oper_expr}")
            continue
        ops = [o.strip() for o in oper_expr.split(",")]
        label_chains = [c.strip() for c in asym_list.split(",")]
        for op_id in ops:
            if op_id in identity_ids:
                continue
            sym_name = id_to_name.get(op_id)
            if sym_name is None:
                logger.warning(f"Operator id {op_id} not found in _pdbx_struct_oper_list")
                continue
            for label_chain in label_chains:
                auth_chain = label_to_auth.get(label_chain)
                if auth_chain in rna_chains:
                    needed[sym_name].add(auth_chain)

    return needed


def _add_sym_copy_chain(G, chain_id, sym_code, R, t):
    """Add a symmetry copy of chain_id to G, transforming all coordinates with (R, t).

    New node IDs use {pdbid}.{chain_id}_{sym_code}.{pos}. Backbone edges are
    replicated from the identity chain so that downstream code sees a complete chain.
    """
    sym_cid = f"{chain_id}_{sym_code}"
    orig_nodes = [n for n in G.nodes() if G.nodes[n].get("chain_id") == chain_id]

    if not orig_nodes:
        logger.warning(f"No nodes for chain {chain_id}, skipping sym copy {sym_code}")
        return

    for orig_node in orig_nodes:
        parts = orig_node.split(".")
        parts[1] = sym_cid
        sym_node = ".".join(parts)

        attrs = dict(G.nodes[orig_node])
        attrs["chain_id"] = sym_cid

        for key in list(attrs.keys()):
            val = attrs[key]
            if key.startswith("xyz_") and val is not None:
                attrs[key] = (R @ np.array(val) + t).tolist()

        if "heavy_atoms" in attrs and attrs["heavy_atoms"] is not None:
            attrs["heavy_atoms"] = {
                atom: (R @ np.array(coord) + t).tolist()
                for atom, coord in attrs["heavy_atoms"].items()
            }

        G.add_node(sym_node, **attrs)

    # Replicate intra-chain backbone edges into the sym copy
    for u, v, data in list(G.edges(data=True)):
        if G.nodes[u].get("chain_id") == chain_id and G.nodes[v].get("chain_id") == chain_id:
            u_parts = u.split("."); u_parts[1] = sym_cid
            v_parts = v.split("."); v_parts[1] = sym_cid
            sym_u = ".".join(u_parts)
            sym_v = ".".join(v_parts)
            if G.has_node(sym_u) and G.has_node(sym_v):
                G.add_edge(sym_u, sym_v, **data)


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
    sym = _sym_code(nt)
    chain_key = f"{chain}_{sym}" if sym != "1_555" else chain
    return f"{pdbid.lower()}.{chain_key}.{pos}{icode}"


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

    # load mmCIF structure
    struc_dict = MMCIF2Dict.MMCIF2Dict(rna_path)

    try:
        rna_chains = get_rna_chains(struc_dict)
        logger.debug(f"RNA chains in {pdbid}: {rna_chains}")
    except KeyError:
        logger.error(f"Couldn't identify RNA chains in {pdbid}")
        return None

    # Parse Cartesian symmetry operators and identify which chains need copies.
    # Done before building the graph so we know the scope of sym copies upfront.
    operators = _parse_oper_list(struc_dict)
    sym_needs = _collect_sym_needs(struc_dict, rna_chains)

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

    # Build full symmetry-copy chains so that cross-symmetry base pairs can be
    # added as ordinary graph edges. Each sym copy is geometrically the image of
    # the original chain under the corresponding Cartesian operator from
    # _pdbx_struct_oper_list. Node IDs use the convention {pdbid}.{chain}_{sym}.{pos}.
    for sym_code, chains in sym_needs.items():
        if sym_code not in operators:
            logger.warning(f"{pdbid}: sym operator {sym_code} not in _pdbx_struct_oper_list, skipping")
            continue
        R, t = operators[sym_code]
        for chain_id in chains:
            _add_sym_copy_chain(G, chain_id, sym_code, R, t)

    for pair in annot_df.itertuples():
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
