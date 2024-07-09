"""
Package installs:
conda install -c salilab dssp
"""
import os
import sys

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.DSSP import DSSP
from time import perf_counter

IONS = ["3CO", "ACT", "AG", "AL", "ALF", "AU", "AU3", "BA", "BEF", "BO4", "BR", "CA", "CAC", "CD", "CL", "CO",
        "CON", "CS", "CU", "EU3", "F", "FE", "FE2", "FLC", "HG", "IOD", "IR", "IR3", "IRI", "IUM", "K", "LI",
        "LU", "MG", "MLI", "MMC", "MN", "NA", "NCO", "NH4", "NI", "NO3", "OH", "OHX", "OS", "PB", "PO4", "PT",
        "PT4", "RB", "RHD", "RU", "SE4", "SM", "SO4", "SR", "TB", "TL", "VO4", "ZN"]

ALLOWED_ATOMS = ['C', 'H', 'N', 'O', 'Br', 'Cl', 'F', 'P', 'Si', 'B', 'Se']
ALLOWED_ATOMS += [atom_name.upper() for atom_name in ALLOWED_ATOMS]
ALLOWED_ATOMS = set(ALLOWED_ATOMS)


def is_dna(res):
    """
    Returns true if the input residue is a DNA molecule

    :param res: biopython residue object
    """
    if res.id[0] != ' ':
        return False
    if is_aa(res):
        return False
    # resnames of DNA are DA, DC, DG, DT
    if 'D' in res.get_resname():
        return True
    else:
        return False


def hariboss_filter(lig, cif_dict, mass_lower_limit=160, mass_upper_limit=1000):
    """
    Sorts ligands into ion / ligand / None
     Returns ions for a specific list of ions, ligands if the hetatm has the right atoms and mass and None otherwise

    :param lig: A biopython ligand residue object
    :param cif_dict: The output of the biopython MMCIF2DICT object
    :param mass_lower_limit:
    :param mass_upper_limit:

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
    """
    Parse an mmCIF dict and return some metadata.

    :param cif: output of the Biopython MMCIF2Dict function
    :return: dictionary of mmcif metadata (for now only resolution terms)
    """
    keys = {'resolution_low': '_reflns.d_resolution_low',
            'resolution_high': '_reflns.d_resolution_high',
            'pdbid': '_pdbx_database_status.entry_id'
            }

    annots = {}
    for name, key in keys.items():
        try:
            annots[name] = mmcif_dict[key]
        except KeyError:
            pass
    return annots


# def get_hetatm(cif_dict):
#     all_hetatm = set(cif_dict.get('_pdbx_nonpoly_scheme.mon_id', []))
#     return all_hetatm

def get_small_partners(cif, mmcif_dict=None, radius=6, mass_lower_limit=160, mass_upper_limit=1000):
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
    structure_id = cif[-8:-4]
    # print(f'Parsing structure {structure_id}...')

    mmcif_dict = MMCIF2Dict(cif) if mmcif_dict is None else mmcif_dict
    parser = FastMMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, cif)

    atom_list = unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)

    all_interactions = {'ligands': [], 'ions': []}

    model = structure[0]
    for res_1 in model.get_residues():
        # Only look around het_flag
        het_flag = res_1.id[0]
        if 'H' in het_flag:
            # hariboss select the right heteroatoms and look around ions and ligands
            selected = hariboss_filter(res_1, mmcif_dict,
                                       mass_lower_limit=mass_lower_limit,
                                       mass_upper_limit=mass_upper_limit)
            if selected is not None:  # ion or ligand
                interaction_dict = {'id': tuple(res_1.id), 'name': res_1.id[0][2:]}
                found_rna_neighbors = set()
                for atom in res_1:
                    # print(atom)
                    for res_2 in neighbors.search(atom.get_coord(), radius=radius, level='R'):
                        # Select for interactions with RNA
                        if not (is_aa(res_2) or is_dna(res_2) or 'H' in res_2.id[0]):
                            # We found a hit
                            rglib_resname = '.'.join([structure_id, str(res_2.get_parent().id), str(res_2.id[1])])
                            found_rna_neighbors.add(rglib_resname)
                if len(found_rna_neighbors) > 0:
                    found_rna_neighbors = sorted(list(found_rna_neighbors))
                    interaction_dict['rna_neighs'] = found_rna_neighbors
                    all_interactions[f"{selected}s"].append(interaction_dict)
    return all_interactions


def add_graph_annotations(g, cif):
    """
    Adds information at the graph level and on the small molecules partner of an RNA molecule

    :param g: the nx graph created from dssr output
    :param cif: the path to a .mmcif file
    :return: the annotated graph, actually the graph is mutated in place
    """
    mmcif_dict = MMCIF2Dict(cif)
    # Add graph level like resolution
    graph_level_annots = get_mmcif_graph_level(mmcif_dict=mmcif_dict)
    g.graph.update(graph_level_annots)

    # Fetch interactions with small molecules and ions
    all_interactions = get_small_partners(cif, mmcif_dict=mmcif_dict)
    g.graph.update(all_interactions)

    # First fill relevant nodes
    for interaction_dict in all_interactions['ligands']:
        ligand_id = interaction_dict['id']
        for rna_neigh in interaction_dict['rna_neighs']:
            # In some rare cases, dssr removes a residue from the cif, in which case it can be fou
            # in the interaction dict but not in graph...
            if rna_neigh in g.nodes:
                g.nodes[rna_neigh]['binding_small-molecule'] = ligand_id
    for interaction_dict in all_interactions['ions']:
        ion_id = interaction_dict['id']
        for rna_neigh in interaction_dict['rna_neighs']:
            # In some rare cases, dssr removes a residue from the cif, in which case it can be fou
            # in the interaction dict but not in graph...
            if rna_neigh in g.nodes:
                g.nodes[rna_neigh]['binding_ion'] = ion_id
    # Then add a None field in all other nodes
    for node, node_data in g.nodes(data=True):
        if 'binding_ion' not in node_data:
            node_data['binding_ion'] = None
        if 'binding_small-molecule' not in node_data:
            node_data['binding_small-molecule'] = None
    return g


def annotate_proteinSSE(g, structure, pdb_file):
    """
    Annotate protein_binding node attributes with the relative SSE
    if available from DSSP

    :param g: (nx graph)
    :param structure: (PDB structure)

    :return g: (nx graph)
    """

    model = structure[0]
    tic = perf_counter()
    dssp = DSSP(model, pdb_file, dssp='mkdssp', file_type='DSSP')
    toc = perf_counter()

    print(dssp.keys())

    a_key = list(dssp.keys())[2]

    print(dssp[a_key])

    print(f'runtime = {tic - toc:0.7f} seconds')

    return g


if __name__ == '__main__':
    pass
    # pdb_file = '../data/structures/4gkk.cif'
    # parser = MMCIFParser()
    # structure = parser.get_structure('4GKK', pdb_file)
    # annotate_proteinSSE(g, structure, '../data/structures/4gkk.dssp')
