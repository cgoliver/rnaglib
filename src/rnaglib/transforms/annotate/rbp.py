import os
from typing import Union
from pathlib import Path

import networkx as nx
from Bio.PDB import PDBParser, NeighborSearch, Selection
from Bio.PDB.MMCIFParser import FastMMCIFParser

from rnaglib.transforms import AnnotationTransform

protein_residues = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

phosphate_atoms = {"P", "OP1", "OP2"}


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

        # set of tuples (chain, pos) for RNA nodes
        rna_res_ids = set([tuple(n.split(".")[1:]) for n in g.nodes()])

        # Extract atoms for RNA and Protein
        rna_atoms = []
        protein_atoms = []
        rna_residues = []

        for chain in structure[0]:
            for residue in chain:
                res_name = residue.get_resname()
                if (chain.id, str(residue.id[1])) in rna_res_ids:
                    for atom in residue.get_atoms():
                        if atom.get_name() in phosphate_atoms:
                            rna_atoms.append(atom)
                            break
                    rna_residues.append(residue)
                if residue.get_resname() in protein_residues:
                    for atom in residue.get_atoms():
                        if atom.get_name() == "CA":
                            protein_atoms.append(atom)
                            break

        # Build a KDTree
        all_rna_atoms = list(rna_atoms)
        all_protein_atoms = list(protein_atoms)

        close_residues = set()

        if len(all_protein_atoms) > 1:
            neighbor_search = NeighborSearch(all_protein_atoms)

            # Find RNA residues near protein residues
            distance_threshold = self.distance_threshold

            for rna_atom in all_rna_atoms:
                close_atoms = neighbor_search.search(rna_atom.coord, distance_threshold)
                if len(close_atoms) > 0:
                    rna_residue = rna_atom.get_parent()
                    close_residues.add((rna_residue.get_parent().id, rna_residue.id[1]))

        # Output the results
        rbp_status = {}
        for node in g.nodes():
            chain, pos = node.split(".")[1:]
            rbp_status[node] = (chain, str(pos)) in close_residues

        nx.set_node_attributes(g, rbp_status, "protein_binding")
        return rna_dict
