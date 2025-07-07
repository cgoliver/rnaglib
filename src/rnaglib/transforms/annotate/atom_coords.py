import os
import sys

import numpy as np
from pathlib import Path
from typing import Union
import networkx as nx
from collections import defaultdict
from loguru import logger
from Bio.PDB.MMCIFParser import MMCIFParser

from rnaglib.transforms import AnnotationTransform

logger.remove()
logger.add(sys.stderr, level="INFO")

class AtomCoordsAnnotator(AnnotationTransform):
    """Annotation transform adding to each node of the dataset the 3D coordinates of all itss heavy atoms

    :param bool include_ions: if set to False, only small-molecule-binding RNA residues are considered part of a binding site. If set to True, ion-binding RNA residues are also considered part of a binding site
    :param float cutoff: the maximal distance (in Angstroms) between an RNA residue and any small molecule or ion atom such that the RNA residue is considered part of a binding site (either 4.0, 6.0 or 8.0, default 6.0)
    """
    def __init__(self, structures_dir: Union[os.PathLike, str]=None, heavy_only=True):
        super().__init__()
        self.structures_dir = structures_dir
        self.heavy_only = heavy_only

    def forward(self, rna_dict: dict) -> dict:
        """Application of the transform to an RNA dictionary object

        :param dict rna_dict: the RNA dictionary which has to be annotated with binding site information
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
            for node in g.nodes():
                chain, pos = node.split(".")[1:]
                try:
                    r = structure[chain][int(pos)]  # in this index-way only standard residue got
                except KeyError:
                    for res in structure[chain]:
                        if int(pos) == res.id[1]:  # got non-standard residue
                            r = res
                try:
                    phos_coord = list(map(float, r["P"].get_coord()))
                except KeyError:
                    phos_coord = np.mean([a.get_coord() for a in r], axis=0)
                    logger.warning(
                        f"Couldn't find phosphate atom, taking center of atoms in residue instead for {pdbid}.{chain}.{pos} is at {phos_coord}."
                    )
                for atom in r:
                    atom_name = atom.get_name()
                    if atom_name != "P" and not (atom_name.startswith("H") and self.heavy_only):
                        atom_coord = list(map(float, atom.get_coord()))
                        coord_dict[node][f"xyz_{atom_name}"] = list(map(float, atom_coord))
                        logger.debug(f"{node} {atom_coord}")

            nx.set_node_attributes(g, coord_dict)

        except Exception as e:
            logger.exception(f"Failed to get coordinates for {pdbid}, {e}")
            return None
        
        return rna_dict