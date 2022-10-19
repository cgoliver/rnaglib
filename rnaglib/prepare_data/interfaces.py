import os
import numpy as np
from Bio.PDB import *
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import csv
import sys
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from prepare_data.retrieve_structures import load_csv


def mmcif_data(cif):
    """Parse an mmCIF return some metadata.

    :param cif: path to an mmCIF

    :return: dictionary of mmcif metadata (for now only resolution terms)
    """
    mmcif_dict = MMCIF2Dict.MMCIF2Dict(cif)
    try:
        resolution_lo = mmcif_dict['_reflns.d_resolution_low']
        resolution_hi = mmcif_dict['_reflns.d_resolution_high']
    except KeyError:
        resolution_lo, resolution_hi = (None, None)
    return {'resolution_low': resolution_lo,
            'resolution_high': resolution_hi
            }


# useless ?
def find_ligand_annotations(cif_path, ligands):
    """
    Returns a list of ligand annotations in from a PDB structures cif file
    if they exist

    :param cif_path: path to PDB structure in mmCIF format
    :param ligans: list of ligands

    :return known_interfaces: list of tuples of known interfaces
                                [(pbid, position, chain, type), ...]
    """
    known_interfaces = []
    mmcif_dict = MMCIF2Dict(cif_path)
    structure_id = cif_path[-8:-4]
    ligands = set(ligands)

    try:
        binding_site_details = mmcif_dict['_struct_site.details']
        binding_site_ids = mmcif_dict['_struct_site.id']
    except KeyError:
        print('No interface annotations found for:\n', cif_path, '\n\n')
        return None

    # Find binding site ID of first ligand if it exists
    site_id = ''
    for site, detail in zip(binding_site_ids, binding_site_details):
        words = detail.split()
        for w in words:
            if w in ligands and len(w) > 1:
                site_id = site

    if site_id == '':
        print('No ligand annotations found for: \n', cif_path, '\n\n')
        return None

    print(site_id)

    # Find the residues of the binding site
    positions = mmcif_dict['_struct_site_gen.label_seq_id']
    chains = mmcif_dict['_struct_site_gen.label_asym_id']
    res_ids = mmcif_dict['_struct_site_gen.label_comp_id']
    sites = mmcif_dict['_struct_site_gen.site_id']

    for position, chain, res_id, site in zip(positions, chains, res_ids, sites):
        if site != site_id: continue
        if len(res_id) > 1 and res_id not in 'AUCG': continue
        known_interfaces.append((structure_id, position, chain, 'ligand'))

    if len(known_interfaces) == 0: return None

    return known_interfaces


def is_dna(res):
    """
    Returns true if the input residue is a DNA molecule

    :param res: biopython residue object
    """
    if res.id[0] != ' ':
        return False
    if is_aa(res):
        return False
    if 'D' in res.get_resname():
        return True
    else:
        return False


# useless ?
def get_offset_pos(res):
    """Get neighboring residues on chain.

    :param res: Biopython residue object.
    """
    pos = list(res.get_parent().get_residues())[0].id[1]
    if pos != 0:
        return res.id[1] - pos + 1
    else:
        return res.id[1] + 1


def get_interfaces(cif_path,
                   ligands_file=os.path.join(script_dir, 'ligand_list.txt'),
                   redundancy_filter=None,
                   pbid_filter=None,
                   cutoff=10,
                   skipWater=True):
    """Obtain RNA interface residues within a single structure of polymers. Uses
    KDTree data structure for vector search, by the biopython NeighborSearch module.

    :param cif_path: Path to structure to analyze (MMCif format)
    :param ligands: list of molecules to classify as small molecule interactions
    :param redundancy_filter: List of non redundancy RNA chains. Can be downloaded from
                        rna.bgsu.edu/rna3dhub/nrlist
    :param cutoff: Number of Angstroms to use as cutoff distance
        for interface calculation.
    :return interface_residues: List of tuples of the pbid, position, chain of RNA-RNA,
                                interaction type, interacting residue, pbid_position
    :return: BioPython Structure object
    """
    # Parse Ligands
    ligands = set()
    with open(ligands_file, 'r') as f:
        for line in f.readlines(): ligands.add(line.strip())

    # Parse Structure
    parser = MMCIFParser(QUIET=True)
    structure_id = cif_path[-8:-4]
    print(f'Parsing structure {structure_id}...')
    structure = parser.get_structure(structure_id, cif_path)

    # print(f'Finding RNA interfaces for structure: {structure_id}')
    # 3-D KD tree
    atom_list = Selection.unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)
    interface_residues = []

    for model in structure:
        model_residues = list(model.get_residues())
        for res_1 in model_residues:

            if is_aa(res_1) or is_dna(res_1) or 'H' in res_1.id[0]:
                continue

            for atom in res_1:
                for res_2 in neighbors.search(atom.get_coord(), cutoff, level='R'):
                    # skip interaction between different the same chain
                    if res_1.get_parent() == res_2.get_parent(): continue

                    # Select for interactions with heteroatoms
                    if 'H' not in res_2.id[0]:
                        continue

                    # get attrs
                    r1_pbid_position = res_1.id[1]
                    c1 = res_1.get_parent().id.strip()
                    r2 = res_2.get_resname().strip()
                    if r2 in ligands:
                        typ = 'ligand'
                    else:
                        typ = 'ion'

                    interface_residues.append((structure_id, c1, typ,
                                               r2, r1_pbid_position))

    # remove duplicates and sort by seqid
    interface_residues = list(set(interface_residues))
    interface_residues_sorted = sorted(interface_residues, key=lambda tup: tup[1])
    return interface_residues_sorted, structure


IONS = ["3CO", "ACT", "AG", "AL", "ALF", "AU", "AU3", "BA", "BEF", "BO4", "BR", "CA", "CAC", "CD", "CL", "CO",
        "CON", "CS", "CU", "EU3", "F", "FE", "FE2", "FLC", "HG", "IOD", "IR", "IR3", "IRI", "IUM", "K", "LI",
        "LU", "MG", "MLI", "MMC", "MN", "NA", "NCO", "NH4", "NI", "NO3", "OH", "OHX", "OS", "PB", "PO4", "PT",
        "PT4", "RB", "RHD", "RU", "SE4", "SM", "SO4", "SR", "TB", "TL", "VO4", "ZN"]

ALLOWED_ATOMS = ['C', 'H', 'N', 'O', 'Br', 'Cl', 'F', 'P', 'Si', 'B', 'Se']
ALLOWED_ATOMS += [atom_name.upper() for atom_name in ALLOWED_ATOMS]
ALLOWED_ATOMS = set(ALLOWED_ATOMS)


# def get_hetatm(cif_dict):
#     all_hetatm = set(cif_dict.get('_pdbx_nonpoly_scheme.mon_id', []))
#     return all_hetatm


def hariboss_filter(lig, cif_dict, mass_lower_limit=1, mass_upper_limit=1000):
    """
    Return if the hetatm is ion, ligand or discard
    :param cif_dict: 
    :param lig: 
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
    """Parse an mmCIF return some metadata.

    :param cif: path to an mmCIF

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


def get_small_partners(cif):
    """
    Returns all the relevant small partners in the form of a dict of list of dicts:
    {'ligands': [
                    {'ligand_id': ('H_ARG', 47, ' '),
                     'rna_neighs': ['1aju.A.21', '1aju.A.22', ... '1aju.A.41']},
                  ],
     'ion': []}

    :param cif:
    :return:
    """
    structure_id = cif[-8:-4]
    print(f'Parsing structure {structure_id}...')

    mmcif_dict = MMCIF2Dict(cif)
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, cif)

    atom_list = Selection.unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)

    all_interactions = {'ligands': [], 'ions': []}

    model = structure[0]
    for res_1 in model.get_residues():
        # Only look around het_flag
        het_flag = res_1.id[0]
        if 'H' in het_flag:
            # hariboss select the right heteroatoms and look around ions and ligands
            selected = hariboss_filter(res_1, mmcif_dict)
            if selected is not None:  # ion or ligand
                interaction_dict = {f'{selected}_id': res_1.id}
                found_rna_neighbors = set()
                for atom in res_1:
                    # print(atom)
                    for res_2 in neighbors.search(atom.get_coord(), 6, level='R'):
                        # Select for interactions with RNA
                        if not (is_aa(res_2) or is_dna(res_2) or 'H' in res_2.id[0]):
                            # We found a hit
                            rglib_resname = '.'.join([structure_id, str(res_2.get_parent().id), str(res_2.id[1])])
                            found_rna_neighbors.add(rglib_resname)
                if len(found_rna_neighbors) > 0:
                    found_rna_neighbors = sorted(list(found_rna_neighbors))
                    interaction_dict['rna_neighs'] = found_rna_neighbors
                    all_interactions[f"{selected}s"].append(interaction_dict)
    print(all_interactions)
