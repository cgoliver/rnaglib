import unittest
from Bio.PDB import *
from ..interfaces import *
import os

scriptdir = os.path.dirname(os.path.realpath(__file__))
rnadir = os.path.join(scriptdir, '..', '..',
                        'data', 'structures')
test_structures_dir = os.path.join(scriptdir, '..', '..',
                        'data', 'test_structures')
ligands_file = os.path.join(scriptdir, '..', '..',
                                'data', 'ligand_list.txt')
ligands = []
with open(ligands_file, 'r') as f:
    for line in f.readlines():
        ligands.append(line.strip())

parser = MMCIFParser(QUIET=True)



class is_dnaTestCase(unittest.TestCase):

    def test_is_dna(self):

        filtered_residues = {}
        unfiltered_residues = {}
        for int_type in ['rna', 'protein', 'dna']:

            # get all residues in all structures of the directory
            dna_files = os.listdir(os.path.join(test_structures_dir, int_type))
            residues = []
            for cif in dna_files:
                path = os.path.join(test_structures_dir, int_type, cif)
                structure = parser.get_structure(cif, path)
                residues = residues + list(structure.get_residues())

            # remove ions/water/ligands
            unfiltered_residues[int_type] = []
            for res in residues:
                if res.id[0] == ' ':
                    unfiltered_residues[int_type].append(res)

            # test function filtering out rna and protein from dna
            filtered_residues[int_type] = []
            for res in residues:
                if is_dna(res):
                    filtered_residues[int_type].append(res)

        self.assertEqual(unfiltered_residues['dna'], filtered_residues['dna'])
        self.assertEqual([], filtered_residues['rna'])
        self.assertEqual([], filtered_residues['protein'])
