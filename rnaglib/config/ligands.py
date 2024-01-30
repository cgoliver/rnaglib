import os
from importlib import resources

pkg = resources.files("rnaglib")
smiles_file = pkg / "config" / "rna_smiles.txt"

script_dir = os.path.dirname(os.path.realpath(__file__))

LIGAND_TO_SMILES = {}

with smiles_file.open() as ligs:
    for line in ligs:
        lig, smiles = line.split()
        LIGAND_TO_SMILES[lig] = smiles
