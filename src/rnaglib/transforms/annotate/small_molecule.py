import os
import sys
from typing import Union
from pathlib import Path
import requests

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.DSSP import DSSP
from time import perf_counter

from rnaglib.transforms import Transform, AnnotationTransform
from rnaglib.transforms.annotate.rbp import protein_residues

IONS = [
    "3CO",
    "ACT",
    "AG",
    "AL",
    "ALF",
    "AU",
    "AU3",
    "BA",
    "BEF",
    "BO4",
    "BR",
    "CA",
    "CAC",
    "CD",
    "CL",
    "CO",
    "CON",
    "CS",
    "CU",
    "EU3",
    "F",
    "FE",
    "FE2",
    "FLC",
    "HG",
    "IOD",
    "IR",
    "IR3",
    "IRI",
    "IUM",
    "K",
    "LI",
    "LU",
    "MG",
    "MLI",
    "MMC",
    "MN",
    "NA",
    "NCO",
    "NH4",
    "NI",
    "NO3",
    "OH",
    "OHX",
    "OS",
    "PB",
    "PO4",
    "PT",
    "PT4",
    "RB",
    "RHD",
    "RU",
    "SE4",
    "SM",
    "SO4",
    "SR",
    "TB",
    "TL",
    "VO4",
    "ZN",
]

# The sets below are offered to callers as excluded_ligands, none of them is a
# default: HARIBOSS keeps all of these deliberately, weak binders being hits for
# fragment based design, and dropping them here would put them out of reach of
# every task downstream. Codes from IONS are absent on purpose, they are sorted
# as ions before excluded_ligands is checked.

# HARIBOSS reports polyamines as common RNA binders. Non specific, but they do
# sit in grooves: median pocket of 88 RNA atoms, against 127 for specific ligands
# and 60 for buffers. Kept apart from the enhancers so a task can drop one
# without the other.
POLYAMINES = {"SPD", "SPK", "SPM"}

# Grouped with the enhancers rather than with the polyamines despite being long
# polar molecules too: their pockets (64) match a buffer's, not a groove.
PEG_FRAGMENTS = {
    "12P", "15P", "1PE", "2PE", "7PE", "DIO", "M2M", "P33", "P6G", "PE3",
    "PE4", "PE5", "PE8", "PEG", "PG4", "PG5", "PG6", "PGE", "TOE", "XPE",
}

# What HARIBOSS calls compounds "not likely to be specific ligands"
BUFFERS_AND_ENHANCERS = {
    # polyols and cryoprotectants
    "BU3", "EDO", "GOL", "MPD", "MRD", "PDO", "PGO",
    # alcohols
    "EOH", "IPA", "MOH", "POH",
    # buffers
    "BTB", "CIT", "CXS", "EPE", "IMD", "MES", "MPO", "NHE", "TAM", "TAR", "TRS",
    # detergents
    "BNG", "BOG", "C8E", "HTG", "LDA", "LMT", "SDS",
    # reducing agents and small organics
    "ACY", "BME", "DTT", "DTV", "FMT", "OXL",
}

CRYSTALLIZATION_ADDITIVES = PEG_FRAGMENTS | BUFFERS_AND_ENHANCERS

SMILES_CACHE = {}


def is_dna(res):
    """
    Returns true if the input residue is a DNA molecule

    :param res: biopython residue object
    """
    if res.id[0] != " ":
        return False
    if is_aa(res):
        return False
    # resnames of DNA are DA, DC, DG, DT
    if "D" in res.get_resname():
        return True
    else:
        return False


# backbone bonds joining a residue to the rest of its chain, as (own, partner)
# atom names: phosphodiester either way round, then peptide either way round,
# then the 5'-5' triphosphate cap bridge either way round. A cap analog (M7G
# capping a transcript) is not joined through its ribose O3' like a normal
# residue: its own terminal bridging phosphate oxygen bonds directly to the
# next residue's alpha phosphate, e.g. m7G(5')ppp(5')N. That bridging oxygen
# is chemically one of three equivalent non-bridging positions on the beta
# phosphate (O1B/O2B/O3B) and which label the deposition assigns it to is
# arbitrary - confirmed on 5f98's six copies, which split 3xO3B/2xO1B/1xO2B
# for the identical bond (~1.5-1.6A), plus 6vu1/6vvj (both O3B).
# Last comes the aminoacyl ester of a charged tRNA, which joins the amino acid
# through its carboxyl carbon to the ribose of the 3' terminal adenosine rather
# than through a peptide bond, so none of the linkages above sees it: fMet on
# tRNA-fMet is the third most frequent "ligand" in the database without it,
# 44 structures, confirmed at 1.56A C-O3' on 5afi. Deposited as either the 2'
# or the 3' isomer, both of which occur, hence the two ribose oxygens.
CHAIN_LINKAGES = (
    ("O3'", "P"), ("P", "O3'"),
    ("C", "N"), ("N", "C"),
    ("O1B", "P"), ("P", "O1B"),
    ("O2B", "P"), ("P", "O2B"),
    ("O3B", "P"), ("P", "O3B"),
    ("C", "O3'"), ("O3'", "C"),
    ("C", "O2'"), ("O2'", "C"),
)


def is_chain_integrated(lig, neighbors, covalent_cutoff=2.0):
    """
    Returns true if the input residue is covalently linked into a polymer chain

    _chem_comp.type only says what a component is in isolation, which is wrong in
    both directions: GTP and M7G are "non-polymer" but are routinely built as the
    5' residue of a transcript, while ARG and CIR are "L-peptide linking" but are
    genuine ligands of the argininamide and citrulline aptamers.

    Only backbone bonds count. Any close contact would do were coordinates ideal,
    but NMR models refined against hydrogen bond restraints put base pair donors
    and acceptors under 2, which would reject bound ligands.

    :param lig: biopython residue object
    :param neighbors: NeighborSearch over the atoms of the same model as lig
    :param covalent_cutoff: a P-O bond is around 1.6 long, a peptide C-N 1.3
    """
    for own, partner in CHAIN_LINKAGES:
        if own not in lig:
            continue
        for other in neighbors.search(lig[own].get_coord(), radius=covalent_cutoff, level="A"):
            if other.get_parent() is not lig and other.get_id() == partner:
                return True
    return False


def is_rna_bound(lig, neighbors, rna_nodes, contact_cutoff=4.0, min_rna=2):
    """
    Returns true if the ligand binds the RNA rather than a protein partner: it
    contacts at least min_rna RNA residues and at least as many RNA as protein.

    The component averaged protein_content annotation never crosses its filter
    threshold, so a ligand sitting in the protein with the RNA a few angstrom
    away still passes. This is the local counterpart: the arginine substrate of
    a synthetase is rejected while the argininamide aptamer ligand is kept, both
    of them ARG.

    RNA is the graph node set, so modified and locked nucleotides deposited as
    heteroatoms (LCC, PSU, ...) count; protein is the residue set from rbp.
    Contacts are distinct residues with a heavy atom within contact_cutoff.

    :param lig: biopython residue object of the candidate ligand
    :param neighbors: NeighborSearch over the atoms of the same model as lig
    :param rna_nodes: set of (chain_id, resnum_str) of the RNA graph nodes
    :param contact_cutoff: heavy atom distance for a contact (default 4.0)
    :param min_rna: fewest RNA contacts for a genuine site (default 2)
    """
    rna, protein = set(), set()
    for atom in lig:
        for res in neighbors.search(atom.get_coord(), radius=contact_cutoff, level="R"):
            if res is lig:
                continue
            key = (res.get_parent().id, str(res.id[1]))
            if key in rna_nodes:
                rna.add(key)
            elif res.get_resname() in protein_residues:
                protein.add(key)
    return len(rna) >= min_rna and len(rna) >= len(protein)


def hariboss_filter(lig, cif_dict, mass_lower_limit=160, mass_upper_limit=1000,
                    verbose=False, additional_atoms=None, disallowed_atoms=None,
                    excluded_ligands=None):
    """
    Sorts ligands into ion / ligand / None
     Returns ions for a specific list of ions, ligands if the hetatm has the right atoms and mass and None otherwise

    :param lig: A biopython ligand residue object
    :param cif_dict: The output of the biopython MMCIF2DICT object
    :param mass_lower_limit:
    :param mass_upper_limit:
    :param excluded_ligands: chem comp codes to reject on identity alone, e.g.
        CRYSTALLIZATION_ADDITIVES. None rejects nothing.

    """
    if excluded_ligands is None:
        excluded_ligands = ()

    allowed_atoms = ["C", "H", "N", "O", "Br", "Cl", "F", "P", "Si", "B", "Se", "S"]
    if not additional_atoms is None:
        allowed_atoms += additional_atoms
    if not disallowed_atoms is None:
        allowed_atoms = [a for a in allowed_atoms if a not in
                              disallowed_atoms]
    allowed_atoms += [atom_name.upper() for atom_name in
                           allowed_atoms]
    allowed_atoms= set(allowed_atoms)

    try:
        lig_name = lig.id[0][2:]
        if lig_name == "HOH":
            return None

        # whether a component is a link in a chain or a free ligand is decided on
        # the coordinates by is_chain_integrated, not on _chem_comp.type: ppGpp is
        # typed "RNA linking" and argininamide "L-peptide linking", yet both are
        # ligands of the riboswitch and the aptamer that bind them

        if lig_name in IONS:
            # if verbose: print("ION")
            return "ion"

        if lig_name in excluded_ligands:
            if verbose: print(f"{lig_name} additive")
            return None

        lig_mass = float(cif_dict["_chem_comp.formula_weight"][cif_dict["_chem_comp.id"].index(lig_name)])

        if lig_mass < mass_lower_limit or lig_mass > mass_upper_limit:
            if verbose: print(f"mass {lig_name}: {lig_mass} fail.")
            return None
        ligand_atoms = set([atom.element for atom in lig.get_atoms()])
        if "C" not in ligand_atoms:
            if verbose: print(f"{lig_name} no C atom")
            return None
        if any([atom not in allowed_atoms for atom in ligand_atoms]):
            if verbose: print(f"{lig_name} Disallowed atoms {ligand_atoms}.")
            return None
        return "ligand"
    except ValueError:
        return None


def get_smiles_from_rcsb(ligand_code):
    """
    Query the RCSB PDB API for a ligand code and return its SMILES string.

    Parameters:
    - ligand_code (str): The 3-letter code of the ligand.

    Returns:
    - str: The SMILES string of the ligand, or None if not found.
    """
    try:
        return SMILES_CACHE[ligand_code]
    except KeyError:
        base_url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{ligand_code.upper()}"
        try:
            response = requests.get(base_url)
            response.raise_for_status()  # Raise an error for HTTP issues
            data = response.json()
            # Extract SMILES string
            smiles = data.get("rcsb_chem_comp_descriptor", {}).get("smiles")
            SMILES_CACHE[ligand_code] = smiles
            return smiles
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except KeyError:
            print(f"SMILES not found for ligand: {ligand_code}")
            return None


def get_small_partners(cif, mmcif_dict=None, radius=6, mass_lower_limit=160,
                       mass_upper_limit=1000, verbose=False,
                       additional_atoms=None,
                       disallowed_atoms=None,
                       excluded_ligands=None,
                       covalent_cutoff=2.0,
                       rna_nodes=None,
                       protein_contact_cutoff=4.0,
                       min_rna_contacts=2):
    """
    Returns all the relevant small partners in the form of a dict of list of dicts:
    {'ligands': [
                    {'id': ('H_ARG', 47, ' '),
                     'name': 'ARG'
                     'rna_neighs': ['1aju.A.21', '1aju.A.22', ... '1aju.A.41']},
                  ],
     'ions': [
                    {'id': ('H_ZN', 56, ' '),
                     'name': 'ZN',
                     'rna_neighs': ['x', y , z]}
                     }

    :param cif: path to a mmcif file
    :param mmcif_dict: if it got computed already
    :return:
    """
    if verbose: print("Searching neibhors at {radius} cutoff.")
    structure_id = cif[-8:-4]
    # print(f'Parsing structure {structure_id}...')

    mmcif_dict = MMCIF2Dict(cif) if mmcif_dict is None else mmcif_dict
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, cif)

    atom_list = unfold_entities(structure, "A")
    neighbors = NeighborSearch(atom_list)

    all_interactions = {"ligands": [], "ions": []}

    model = structure[0]
    # scoped to one model, the copies of a residue in the other models of an NMR
    # ensemble sit within bonding distance of the ligand
    model_neighbors = NeighborSearch(unfold_entities(model, "A"))
    for res_1 in model.get_residues():
        # Only look around het_flag
        het_flag = res_1.id[0]
        if "H" in het_flag:
            # if verbose:
                # print(f"Processing {structure_id}: {het_flag}")
            # hariboss select the right heteroatoms and look around ions and ligands
            selected = hariboss_filter(
                res_1, mmcif_dict, mass_lower_limit=mass_lower_limit,
                mass_upper_limit=mass_upper_limit, verbose=verbose,
                additional_atoms=additional_atoms,
                disallowed_atoms=disallowed_atoms,
                excluded_ligands=excluded_ligands
            )
            # if selected is None and verbose:
                # print("Failed HARIBOSS filter.")

            # ions coordinate at around 2.0, only ligands are bond checked
            if selected == "ligand" and is_chain_integrated(res_1, model_neighbors, covalent_cutoff):
                if verbose: print(f"{res_1.id[0][2:]} chain integrated")
                selected = None

            # the RNA node set is only known when called from the transform
            if selected == "ligand" and rna_nodes is not None and not is_rna_bound(
                    res_1, model_neighbors, rna_nodes, protein_contact_cutoff, min_rna_contacts):
                if verbose: print(f"{res_1.id[0][2:]} protein bound")
                selected = None

            if selected is not None:  # ion or ligand
                name = res_1.id[0][2:]
                smiles = get_smiles_from_rcsb(name)
                # the chain leads the id: a biopython residue id carries none, so two copies of one ligand sitting at
                # equivalent positions in two chains share an id, and every consumer grouping residues by it welds
                # their two sites into a single pocket. Measured on the v4 build, a quarter of the annotated sites
                # spanned more than one chain and a third of those were geometrically impossible, up to 260A across
                interaction_dict = {"id": (res_1.get_parent().id,) + tuple(res_1.id), "name": name, "smiles": smiles}
                found_rna_neighbors = set()
                for atom in res_1:
                    # print(atom)
                    for res_2 in neighbors.search(atom.get_coord(), radius=radius, level="R"):
                        # Select for interactions with RNA
                        if not (is_aa(res_2) or is_dna(res_2) or "H" in res_2.id[0]):
                            # We found a hit
                            rglib_resname = ".".join([structure_id, str(res_2.get_parent().id), str(res_2.id[1])])
                            found_rna_neighbors.add(rglib_resname)
                if len(found_rna_neighbors) > 0:
                    found_rna_neighbors = sorted(list(found_rna_neighbors))
                    interaction_dict["rna_neighs"] = found_rna_neighbors
                    all_interactions[f"{selected}s"].append(interaction_dict)
    return all_interactions


class SmallMoleculeBindingTransform(AnnotationTransform):
    """Annotate RNAs with small molecule binding information.

    :param structures_dir: path to directory containing the mmCIFs to annotate
    :param cutoff: distance threshold (Angstroms) to use for including small molecule ligands (default= 6).
    """

    def __init__(self, structures_dir: Union[os.PathLike, str], cutoffs=[4.0,
                                                                         6.0,
                                                                         8.0],
                 mass_lower_limit=160,
                 mass_upper_limit=1000,
                 verbose=False,
                 additional_atoms=None,
                 disallowed_atoms=None,
                 excluded_ligands=None,
                 covalent_cutoff=2.0,
                 protein_contact_cutoff=4.0,
                 min_rna_contacts=2
                 ):
        self.structures_dir = structures_dir
        self.cutoffs = sorted(cutoffs)
        self.verbose = verbose
        self.mass_lower_limit = mass_lower_limit
        self.mass_upper_limit = mass_upper_limit
        self.additional_atoms = additional_atoms
        self.disallowed_atoms = disallowed_atoms
        self.excluded_ligands = excluded_ligands
        self.covalent_cutoff = covalent_cutoff
        self.protein_contact_cutoff = protein_contact_cutoff
        self.min_rna_contacts = min_rna_contacts


    def forward(self, rna_dict: dict) -> dict:
        """
        Adds information at the graph level and on the small molecules partner of an RNA molecule

        :param g: the nx graph created from dssr output
        :param cif: the path to a .mmcif file
        :return: the annotated graph, actually the graph is mutated in place
        """
        g = rna_dict["rna"]
        cif = str(Path(self.structures_dir) / f"{g.graph['pdbid'].lower()}.cif")
        mmcif_dict = MMCIF2Dict(cif)

        rna_nodes = set(tuple(node.split(".")[1:]) for node in g.nodes())

        lig_to_smiles = {}
        for cutoff in self.cutoffs:
            # Reset both fields for every node before recomputing. A previous
            # pass over this same graph (e.g. a stale copy from before this
            # ligand was reclassified as chain-integrated) may have left a
            # stale non-None value here; only filling in currently-absent
            # keys would let that stale value survive undetected.
            for node, node_data in g.nodes(data=True):
                node_data[f"binding_ion_{cutoff}A"] = None
                node_data[f"binding_small-molecule-{cutoff}A"] = None

            # Fetch interactions with small molecules and ions
            all_interactions = get_small_partners(cif, mmcif_dict=mmcif_dict,
                                                  radius=cutoff,
                                                  verbose=self.verbose,
                                                  mass_lower_limit=self.mass_lower_limit,
                                                  mass_upper_limit=self.mass_upper_limit,
                                                  additional_atoms=self.additional_atoms,
                                                  disallowed_atoms=self.disallowed_atoms,
                                                  excluded_ligands=self.excluded_ligands,
                                                  covalent_cutoff=self.covalent_cutoff,
                                                  rna_nodes=rna_nodes,
                                                  protein_contact_cutoff=self.protein_contact_cutoff,
                                                  min_rna_contacts=self.min_rna_contacts)

            for interaction_dict in all_interactions["ligands"]:
                for rna_neigh in interaction_dict["rna_neighs"]:
                    if rna_neigh in g.nodes:
                        g.nodes[rna_neigh][f"binding_small-molecule-{cutoff}A"] = {
                            "name": interaction_dict["name"],
                            "id": interaction_dict["id"],
                        }
                        lig_to_smiles[interaction_dict["name"]] = interaction_dict["smiles"]
            for interaction_dict in all_interactions["ions"]:
                ion_id = interaction_dict["id"]
                for rna_neigh in interaction_dict["rna_neighs"]:
                    # In some rare cases, dssr removes a residue from the cif, in which case it can be fou
                    # in the interaction dict but not in graph...
                    if rna_neigh in g.nodes:
                        g.nodes[rna_neigh][f"binding_ion_{cutoff}A"] = ion_id
            rna_dict["rna"].graph["ligand_to_smiles"] = lig_to_smiles
        return rna_dict
