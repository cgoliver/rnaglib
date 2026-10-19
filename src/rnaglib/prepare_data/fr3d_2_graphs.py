"""Build 2.5D RNA graphs (residues as nodes, backbone and base-pair interactions as edges) using fr3d-python."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from collections import defaultdict
from Bio.PDB import MMCIF2Dict, MMCIFParser
import networkx as nx
from loguru import logger

try:
    from fr3d.classifiers.NA_pairwise_interactions import generatePairwiseAnnotation_import
except ImportError:
    print("Missing fr3d installation: pip install git+https://github.com/cgoliver/fr3d-python.git")

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
        cleaned.extend(r.split(","))
    return cleaned


# Maximum O3'(i)-P(i+1) distance (Angstrom) for two label-consecutive residues to be joined by a backbone edge.
# Label adjacency alone is not enough: unmodelled segments are sometimes omitted from the entity sequence, so the
# residues on both sides of a physical chain break get consecutive label_seq_ids. A covalent O3'-P bond is ~1.6 A;
# 4 A tolerates poorly modelled linkages while a single missing nucleotide almost always spans more.
MAX_O3P_DIST = 4.0


def get_polymer_rna_residue_keys(mmcif_dict):
    """Return the set of (auth_asym_id, auth_seq_id, ins_code) of residues belonging to a polyribonucleotide entity.

    Uses the mmCIF polymer definition (label_seq_id set + polyribonucleotide entity) rather than the chem_comp type,
    so polymer residues typed e.g. 'non-polymer' (5' GTP) or 'RNA OH 3 prime terminus' are kept, while non-polymer
    ligands sharing the author chain (no label_seq_id) are excluded.
    """
    to_list = lambda x: x if isinstance(x, list) else [x]
    rna_entities = {
        entity
        for entity, entity_type in zip(to_list(mmcif_dict["_entity_poly.entity_id"]), to_list(mmcif_dict["_entity_poly.type"]))
        if entity_type == "polyribonucleotide"
    }
    auth_seq = mmcif_dict["_atom_site.auth_seq_id"]
    ins_codes = mmcif_dict.get("_atom_site.pdbx_PDB_ins_code", ["?"] * len(auth_seq))
    keys = set()
    for ca, cs, ic, entity, ls in zip(
        mmcif_dict["_atom_site.auth_asym_id"], auth_seq, ins_codes,
        mmcif_dict["_atom_site.label_entity_id"], mmcif_dict["_atom_site.label_seq_id"],
    ):
        if entity in rna_entities and ls not in (".", "?"):
            keys.add((ca, int(cs), " " if ic in (".", "?") else ic))
    return keys


def get_residue_list(chain, poly_keys):
    """Polymer RNA residues of `chain`, sorted in sequence (label_seq_id) order.

    Sorting by author numbering misplaces residues whenever it does not follow the sequence (numbering restarts,
    inserted segments numbered 1000+, plain numbers placed after their insertion-code variants, reversed numbering).
    """
    return sorted(
        [
            r for r in chain.get_residues()
            if (chain.id, r.id[1], r.id[2]) in poly_keys and r.xtra.get("label_seq_id") is not None
        ],
        key=lambda x: (x.xtra["label_seq_id"], x.id[1], x.id[2]),
    )


def _o3p_distance(res_i, res_j):
    """O3'(res_i)-P(res_j) distance in Angstrom, None if either atom is missing."""
    if "O3'" not in res_i or "P" not in res_j:
        return None
    return round(float(res_i["O3'"] - res_j["P"]), 3)


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

    New node IDs use {pdbid}.{chain_id}-{sym_code}.{pos}. Backbone edges are replicated from the identity chain
    so that downstream code sees a complete chain.
    """
    sym_cid = f"{chain_id}-{sym_code.replace('_', '-')}"
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


def get_bb(structure, rna_chains, poly_keys, pdbid=""):
    """Get the polymer RNA residues and the backbone edges joining them.

    Two residues are joined if they are consecutive in the entity sequence (label_seq_id) and their O3'-P distance
    is at most MAX_O3P_DIST (kept on label adjacency alone if either atom is missing). The distance is stored on
    both edges as 'o3p_dist' (None if unmeasurable). Residues without any backbone edge (isolated by chain breaks
    on both sides) are still returned.

    :returns: (backbone edges, {node: one-letter code}, {node: residue name}, {node: label_seq_id},
        {node: Bio.PDB residue})
    """
    bb = []
    nt_types = {}
    nt_types_full = {}
    lsi_dict = {}
    residues = {}
    for chain in structure.get_chains():
        if chain.id not in rna_chains:
            continue
        reslist = get_residue_list(chain, poly_keys)
        logger.debug(reslist)

        for res in reslist:
            name = _res_node_id(pdbid, chain.id, res)
            residues[name] = res
            nt_types[name] = rna_letters_3to1(res.get_resname())
            nt_types_full[name] = res.get_resname()
            lsi_dict[name] = res.xtra["label_seq_id"]

        for three_p, five_p in zip(reslist, reslist[1:]):
            if five_p.xtra["label_seq_id"] != three_p.xtra["label_seq_id"] + 1:
                continue
            fivep_name = _res_node_id(pdbid, chain.id, five_p)
            threep_name = _res_node_id(pdbid, chain.id, three_p)
            o3p_dist = _o3p_distance(three_p, five_p)
            if o3p_dist is not None and o3p_dist > MAX_O3P_DIST:
                logger.debug(f"{threep_name}->{fivep_name}: O3'-P {o3p_dist} A > {MAX_O3P_DIST}, no backbone edge")
                continue
            bb.append((fivep_name, threep_name, {"LW": "B53", "o3p_dist": o3p_dist}))
            bb.append((threep_name, fivep_name, {"LW": "B35", "o3p_dist": o3p_dist}))

    return bb, nt_types, nt_types_full, lsi_dict, residues


def nt_to_rgl(nt, pdbid):
    parts = nt.split("|")
    chain = parts[2]
    pos = parts[4]
    icode = parts[7].strip() if len(parts) > 7 else ""
    sym = _sym_code(nt)
    chain_key = f"{chain}-{sym.replace('_', '-')}" if sym != "1_555" else chain
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
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        stderr = getattr(e, "stderr", None)
        logger.warning(f"forgi annotation failed for {pdbid}: {e}" + (f"\n{stderr}" if stderr else ""))
        nx.set_node_attributes(G, None, "secondary_element")
        return

    secondary_element = {node: None for node in G.nodes()}
    for node, elem_type in node_types.items():
        if node in secondary_element:
            secondary_element[node] = elem_type
    nx.set_node_attributes(G, secondary_element, "secondary_element")


def fr3d_to_graph(rna_path, atom_coords_to_store=["P"], include_stacking=False):
    """Use fr3d to generate networkx annotation graph.

    For multi-model entries (NMR ensembles), only the first model is used, both for coordinates and for
    base-pair annotations.

    :param rna_path: path to the mmCIF file of the RNA structure
    :param atom_coords_to_store: atom names whose coordinates are stored as 'xyz_<atom>' node attributes, or
        ["all_atom"] to store every heavy atom in a 'heavy_atoms' node attribute
    :param include_stacking: also add fr3d stacking interactions as edges
    :returns nx.MultiDiGraph: networkx graph with annotations, or None if fr3d or chain parsing fails
    """
    rna_path = Path(rna_path)
    try:
        category = "basepair,stacking" if include_stacking else "basepair"
        annot_df = generatePairwiseAnnotation_import(rna_path, category=category)
    except Exception:
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

    # Cartesian symmetry operators, and the RNA chains needing a symmetry copy in the biological assembly
    operators = _parse_oper_list(struc_dict)
    sym_needs = _collect_sym_needs(struc_dict, rna_chains)

    poly_keys = get_polymer_rna_residue_keys(struc_dict)

    parser = MMCIFParser()
    structure = parser.get_structure("", rna_path)[0]
    model_num = str(structure.serial_num)

    # author residue id -> label_seq_id
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

    bbs, nt_types, nt_types_full, lsi_dict, residues = get_bb(structure, rna_chains, poly_keys, pdbid=pdbid)
    logger.trace(bbs)
    G = nx.MultiDiGraph()
    G.add_nodes_from(residues)
    G.add_edges_from(bbs)

    nx.set_node_attributes(G, nt_types, "nt")
    nx.set_node_attributes(G, nt_types_full, "nt_full")
    nx.set_node_attributes(G, lsi_dict, "label_seq_id")

    for node in G.nodes():
        G.nodes[node]["nt_code"] = nt_types[node]
        G.nodes[node]["chain_id"] = node.split(".")[1]
        G.nodes[node]["is_modified"] = len(nt_types_full[node]) != 1

    coord_dict = {}
    for node, r in residues.items():
        centroid_coord = np.mean([a.get_coord() for a in r], axis=0)
        coord_dict[node] = {"xyz_centroid": list(map(float, centroid_coord))}
        if atom_coords_to_store == ["all_atom"]:
            coord_dict[node]["heavy_atoms"] = {
                a.get_name(): list(map(float, a.get_coord())) for a in r if a.element.strip() != "H"
            }
        else:
            for atom_type in atom_coords_to_store:
                if atom_type in r:
                    atom_coord = list(map(float, r[atom_type].get_coord()))
                else:
                    atom_coord = None
                    logger.warning(f"{node}: no {atom_type} atom")
                coord_dict[node][f"xyz_{atom_type}"] = atom_coord
    nx.set_node_attributes(G, coord_dict)

    # Symmetry copies of the chains of the biological assembly, so that base pairs across symmetry mates are ordinary
    # edges. Node IDs use the convention {pdbid}.{chain}-{sym}.{pos}.
    for sym_code, chains in sym_needs.items():
        if sym_code not in operators:
            logger.warning(f"{pdbid}: sym operator {sym_code} not in _pdbx_struct_oper_list, skipping")
            continue
        R, t = operators[sym_code]
        for chain_id in chains:
            _add_sym_copy_chain(G, chain_id, sym_code, R, t)

    edge_map = EDGE_MAP_RGLIB_WITH_STACKING if include_stacking else EDGE_MAP_RGLIB
    for pair in annot_df.itertuples():
        elabel = pair.interaction
        if elabel not in edge_map:
            continue
        # fr3d annotates every model of an NMR ensemble, and node IDs do not carry the model number
        if pair.source.split("|")[1] != model_num:
            continue

        nt1 = nt_to_rgl(pair.source, pdbid)
        nt2 = nt_to_rgl(pair.target, pdbid)

        if G.has_node(nt1) and G.has_node(nt2):
            G.add_edge(nt1, nt2, LW=elabel)

    _forgi_annotate_sse(G, rna_path, pdbid)

    G.graph["name"] = pdbid
    G.graph["pdbid"] = pdbid

    return G
