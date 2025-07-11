import os
import sys

import torch
import numpy as np
from pathlib import Path
from typing import Union
import networkx as nx
from collections import defaultdict
from loguru import logger
from Bio.PDB.MMCIFParser import MMCIFParser

from rnaglib.transforms import AnnotationTransform
from rnaglib.algorithms import fix_buggy_edges, internal_coords, internal_vecs, rbf_expansion, positional_encoding, normed_vec, get_backbone_coords

logger.remove()
logger.add(sys.stderr, level="INFO")

class AtomCoordsAnnotator(AnnotationTransform):
    """Annotation transform adding to each node of the dataset the 3D coordinates of its atoms or heavy atoms

    :param Union[os.PathLike, str] structures_dir: directory in which the cif files of RNAs are stored
    :param bool heavy_only: if set to True, the coordinates of all heavy atoms are annotated, if set to False, the coordinates of hydrogen atoms are also computed
    :param List[str] atoms_list: list of names of atoms which coordinates to extract. If not set to None, it has the prioirty over heavy_only parameter
    """
    def __init__(self, structures_dir: Union[os.PathLike, str]=None, heavy_only=True, atoms_list=None):
        super().__init__()
        self.structures_dir = structures_dir
        self.heavy_only = heavy_only
        self.atoms_list = atoms_list

    def forward(self, rna_dict: dict) -> dict:
        """Application of the transform to an RNA dictionary object

        :param dict rna_dict: the RNA dictionary which has to be annotated with atom coordinates
        :return: the annotated version of rna_dict
        :rtype: dict
        """
        if self.structures_dir is None:
            dirname = os.path.join(os.path.expanduser("~"), ".rnaglib/")
            self.structures_dir = os.path.join(dirname, "structures")
        g = rna_dict["rna"]
        pdbid = g.graph['pdbid'].lower()
        rna_path = str(Path(self.structures_dir) / f"{pdbid}.cif")
        rna_path = Path(rna_path)

        # add coords with biopython
        parser = MMCIFParser()
        structure = parser.get_structure("", rna_path)[0]

        try:
            coord_dict = defaultdict(dict)
            atom_names = []
            for node in g.nodes():
                chain, pos = node.split(".")[1:]
                try:
                    r = structure[chain][int(pos)]  # in this index-way only standard residue got
                except KeyError:
                    for res in structure[chain]:
                        if int(pos) == res.id[1]:  # got non-standard residue
                            r = res
                for atom in r:
                    atom_name = atom.get_name()
                    if atom_name != "P" and not (atom_name.startswith("H") and self.heavy_only) and not(self.atoms_list is not None and atom not in self.atoms_list):
                        atom_names.append(atom_name)
                        atom_coord = list(map(float, atom.get_coord()))
                        coord_dict[node][f"xyz_{atom_name}"] = list(map(float, atom_coord))
                        logger.debug(f"{node} {atom_coord}")
            for node in g.nodes():
                for atom_name in atom_names:
                    if f"xyz_{atom_name}" not in coord_dict[node]:
                        coord_dict[node][f"xyz_{atom_name}"] = None
            nx.set_node_attributes(g, coord_dict)

        except Exception as e:
            logger.exception(f"Failed to get coordinates for {pdbid}, {e}")
            return None
        
        return rna_dict