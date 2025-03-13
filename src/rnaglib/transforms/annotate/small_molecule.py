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


def hariboss_filter(lig, cif_dict, mass_lower_limit=160, mass_upper_limit=1000,
                    verbose=False, additional_atoms=None, disallowed_atoms=None):
    """
    Sorts ligands into ion / ligand / None
     Returns ions for a specific list of ions, ligands if the hetatm has the right atoms and mass and None otherwise

    :param lig: A biopython ligand residue object
    :param cif_dict: The output of the biopython MMCIF2DICT object
    :param mass_lower_limit:
    :param mass_upper_limit:

    """

    allowed_atoms = ["C", "H", "N", "O", "Br", "Cl", "F", "P", "Si", "B", "Se"]
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

        if cif_dict["_chem_comp.type"][cif_dict["_chem_comp.id"].index(lig_name)] in ["RNA linking", "DNA linking"]:
            if verbose: print(f"{lig_name} Covalent")
            return None

        if lig_name in IONS:
            # if verbose: print("ION")
            return "ion"

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
                       disallowed_atoms=None):
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
                disallowed_atoms=disallowed_atoms
            )
            # if selected is None and verbose:
                # print("Failed HARIBOSS filter.")

            if selected is not None:  # ion or ligand
                name = res_1.id[0][2:]
                smiles = get_smiles_from_rcsb(name)
                interaction_dict = {"id": tuple(res_1.id), "name": name, "smiles": smiles}
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
                 disallowed_atoms=None
                 ):
        self.structures_dir = structures_dir
        self.cutoffs = sorted(cutoffs)
        self.verbose = verbose
        self.mass_lower_limit = mass_lower_limit
        self.mass_upper_limit = mass_upper_limit
        self.additional_atoms = additional_atoms
        self.disallowed_atoms = disallowed_atoms


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

        lig_to_smiles = {}
        for cutoff in self.cutoffs:
            # Fetch interactions with small molecules and ions
            all_interactions = get_small_partners(cif, mmcif_dict=mmcif_dict,
                                                  radius=cutoff,
                                                  verbose=self.verbose,
                                                  mass_lower_limit=self.mass_lower_limit,
                                                  mass_upper_limit=self.mass_upper_limit,
                                                  additional_atoms=self.additional_atoms,
                                                  disallowed_atoms=self.disallowed_atoms)

            # First fill relevant nodes
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
            # Then add a None field in all other nodes
            for node, node_data in g.nodes(data=True):
                if f"binding_ion_{cutoff}A" not in node_data:
                    node_data[f"binding_ion_{cutoff}A"] = None
                if f"binding_small-molecule-{cutoff}A" not in node_data:
                    node_data[f"binding_small-molecule-{cutoff}A"] = None
            rna_dict["rna"].graph["ligand_to_smiles"] = lig_to_smiles
        return rna_dict
