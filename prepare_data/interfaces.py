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

def get_nonRedundantChains(NR_csv_file):
    """
    Parse NR BGSU csv file for a list of non-redundant RNA chains
    list can be downloaded from:
        http://rna.bgsu.edu/rna3dhub/nrlist
    :param repr_set: Set of representative RNAs output (see load_csv())
    :return: set of non-redundant RNA chains (tuples of (structure, model, chain))
    """

    nonRedundantStrings = load_csv(NR_csv_file)
    nonRedundantChains = []

    # split into each IFE (Integrated Functional Element)
    for representative in nonRedundantStrings:
        items = representative.split('+')
        for entry in items:
            pbid, model, chain = entry.split('|')
            nonRedundantChains.append((pbid, model, chain))

    return set(nonRedundantChains)

def find_ligand_annotations(cif_path, ligands):
    """
    Returns a list of ligand annotations in from a PDB structures cif file
    if they exist
    :Param cif_path: path to PDB structure in mmCIF format
    :Param ligans: list of ligands
    :return known_interfaces: list of tuples of known interfaces
                                [(pbid, position, chain, type), ...]
    """
    known_interfaces=[]
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
    returns true if the input residue is a DNA molecule
    """
    if res.id[0] != ' ':
        return False
    if is_aa(res):
        return False
    if 'D' in res.get_resname():
        return True
    else:
        return False

def get_offset_pos(res):
        pos = list(res.get_parent().get_residues())[0].id[1]
        if pos != 0:
            return res.id[1] - pos + 1
        else:
            return res.id[1] + 1

def get_interfaces(cif_path, ligands,
                    redundancy_filter=None,
                    pbid_filter = None,
                    cutoff=10,
                    skipWater=True):
    """Obtain RNA interface residues within a single structure of polymers. Uses
    KDTree data structure for vector search, by the biopython NeighborSearch module.

    Args:
        `cif_path (str)`: Path to structure to analyze (MMCif format)
        `ligands `: list of molecules to classify as small molecule interactions
        `redundancy_filter`: List of non redundancy RNA chains. Can be downloaded from
                        rna.bgsu.edu/rna3dhub/nrlist
        `cutoff (float, int)`: Number of Angstroms to use as cutoff distance
        for interface calculation.
    Returns:
        `interface_residues`: List of tuples of the pbid, position, chain of RNA-RNA,
                                interaction type, interacting residue, pbid_position
        `Structure`: BioPython Structure object
    """
    parser = MMCIFParser(QUIET=True)
    structure_id = cif_path[-8:-4]
    print(f'Parsing structure {structure_id}...')
    try:
        structure = parser.get_structure(structure_id, cif_path)
    except KeyboardInterrupt:
        print('Execution stopped')
        raise Exception
    except:
        print(f'ERROR: file {cif_path} not found, trying to download from pdb...')
        pdbl = PDBList()
        struct_path = os.path.join(script_dir, '..', 'data', 'structures', '.downloads')
        pdbl.retrieve_pdb_file(structure_id, struct_path)
        cif_path = os.path.join(struct_path, structure_id + '.cif')
        try:
            structure = parser.get_structure(structure_id, cif_path)
        except ValueError:
            print('Could not parse new downloaded file either')
            return None

    if redundancy_filter:
        representative_set = get_nonRedundantChains(redundancy_filter)


    # print(f'Finding RNA interfaces for structure: {structure_id}')
    #3-D KD tree
    atom_list = Selection.unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)
    close_residues = neighbors.search_all(cutoff, level='R')
    interface_residues = []
    for res_1, res_2 in close_residues:

       # skip interactions with water
        if skipWater:
            if res_1.id[0] == 'W' or res_2.id[0] == 'W': continue

        # skip protein-protein pairs
        if is_aa(res_1) and is_aa(res_2): continue

        # skip interactions with DNA
        if is_dna(res_1) or is_dna(res_2): continue

        # skip interaction between different the same chain
        if res_1.get_parent() == res_2.get_parent(): continue

        # get position offset
        r1_position = get_offset_pos(res_1)
        r2_position = get_offset_pos(res_2)
        r1_pbid_position = res_1.id[1]
        r2_pbid_position = res_2.id[1]

        # get chain names and res names
        c1 = res_1.get_parent().id.strip()
        c2 = res_2.get_parent().id.strip()
        r1 = res_1.get_resname().strip()
        r2 = res_2.get_resname().strip()

        # Filter for redundancy
        if redundancy_filter:
            model1 = res_1.get_full_id()[1]
            full_id1 = (structure_id.upper(), str(model1 + 1), c1.upper())
            model2 = res_2.get_full_id()[1]
            full_id2 = (structure_id.upper(), str(model2 + 1), c2.upper())
            if full_id1 not in representative_set and full_id2 not in representative_set:
                continue

        # Determine interaction type and append to corresponding dataset
        # RNA-Protein 
        if is_aa(res_1):
            interface_residues.append((structure_id, r2_position, c2, 'protein',
                                        'True', r2_pbid_position))
        elif is_aa(res_2):
            interface_residues.append((structure_id, r1_position, c1, 'protein',
                                        'True', r1_pbid_position))
        # RNA-RNA 
        elif res_1.id[0] == ' ' and res_2.id[0] == ' ':
            interface_residues.append((structure_id, r1_position, c1, 'rna',
                                        'True', r1_pbid_position))
            interface_residues.append((structure_id, r2_position, c2, 'rna',
                                        'True', r2_pbid_position))
        # RNA-smallMolecule
        elif  r1 in ligands and 'H' in res_1.id[0] and ' ' in res_2.id[0]:
            interface_residues.append((structure_id, r2_position, c2, 'ligand',
                                        r1, r2_pbid_position))
        elif  r2 in ligands and 'H' in res_2.id[0] and ' ' in res_1.id[0]:
            interface_residues.append((structure_id, r1_position, c1, 'ligand',
                                        r2, r1_pbid_position))
        # RNA-Ion
        elif 'H' in res_1.id[0] and ' ' in res_2.id[0]:
            interface_residues.append((structure_id, r2_position, c2, 'ion',
                                        r1, r2_pbid_position))
        elif 'H' in res_2.id[0] and ' ' in res_1.id[0]:
            interface_residues.append((structure_id, r1_position, c2, 'ion',
                                        r2, r1_pbid_position))
        elif  'H' in res_2.id[0] and 'H' in res_1.id[0]:
            continue
        else:
            print('warning unmatched residue pair \t res_1.id:', res_1.id, 'res_2.id:', res_2.id)

    # remove duplicates and sort by seqid
    interface_residues = list(set(interface_residues))
    interface_residues_sorted = sorted(interface_residues, key=lambda tup: tup[2])
    interface_residues_sorted = sorted(interface_residues, key=lambda tup: tup[1])

    return interface_residues_sorted, structure

