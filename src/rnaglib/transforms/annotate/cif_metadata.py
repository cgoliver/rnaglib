import os
from pathlib import Path
from typing import Union

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from rnaglib.transforms import AnnotationTransform


class CifMetadata(AnnotationTransform):
    def __init__(self, structures_dir: Union[os.PathLike, str]):
        self.structures_dir = structures_dir

    def forward(self, rna_dict):
        """
        Parse an mmCIF dict and return some metadata.

        :param cif: output of the Biopython MMCIF2Dict function
        :return: dictionary of mmcif metadata (for now only resolution terms)
        """
        keys = {
            "resolution_low": "_reflns.d_resolution_low",
            "resolution_high": "_reflns.d_resolution_high",
        }
        g = rna_dict["rna"]
        cif = str(Path(self.structures_dir) / f"{g.graph['pdbid'].lower()}.cif")
        mmcif_dict = MMCIF2Dict(cif)
        annots = {}
        for name, key in keys.items():
            try:
                annots[name] = mmcif_dict[key][0]
            except KeyError:
                pass
        g.graph.update(annots)
        return rna_dict
