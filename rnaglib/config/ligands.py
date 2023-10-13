import os

script_dir = os.path.dirname(os.path.realpath(__file__))

LIGAND_TO_SMILES = {}

with open(os.path.join(script_dir, "rna_smiles.txt"), "r") as ligs:
    for line in ligs:
        lig, smiles = line.split()
        LIGAND_TO_SMILES[lig] = smiles
