import os
from typing import Union
from pathlib import Path

import networkx as nx
from Bio.PDB import PDBParser, NeighborSearch, Selection
from Bio.PDB.MMCIFParser import FastMMCIFParser

from rnaglib.transforms import AnnotationTransform


class RBPTransform(AnnotationTransform):

    def __init__(self, structures_dir: Union[os.PathLike, str], distance_threshold: float = 5.0):
        self.structures_dir = structures_dir
        self.distance_threshold = distance_threshold
        pass

    def forward(self, rna_dict: dict) -> dict:
        # Load the structure
        g = rna_dict["rna"]
        cif = str(Path(self.structures_dir) / f"{g.graph['pdbid'].lower()}.cif")
        parser = FastMMCIFParser(QUIET=True)
        structure = parser.get_structure("", cif)

        # Load the structure

        # Extract atoms for RNA and Protein
        rna_atoms = []
        protein_atoms = []
        rna_residues = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    # Classify based on residue name
                    res_name = residue.get_resname()
                    if res_name in ["A", "U", "G", "C"]:  # RNA residue codes
                        rna_atoms.extend(residue.get_atoms())
                        rna_residues.append(residue)
                    elif "CA" in residue or "N" in residue:  # Typical protein residues
                        protein_atoms.extend(residue.get_atoms())

        # Build a KDTree
        all_rna_atoms = list(rna_atoms)
        all_protein_atoms = list(protein_atoms)
        neighbor_search = NeighborSearch(all_rna_atoms)

        # Find RNA residues near protein residues
        distance_threshold = 5.0
        close_residues = set()

        for atom in all_protein_atoms:
            close_atoms = neighbor_search.search(atom.coord, distance_threshold)
            for close_atom in close_atoms:
                close_residue = close_atom.get_parent()
                if close_residue in rna_residues:
                    close_residues.add((close_residue.get_parent().id, close_residue.id[1]))

        # Output the results
        rbp_status = {}
        for node in g.nodes():
            chain, pos = node.split(".")[1:]
            rbp_status[node] = (chain, str(pos)) in close_residues

        nx.set_node_attributes(g, rbp_status, "protein_binding")
        return rna_dict
