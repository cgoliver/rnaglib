import os
from typing import Union
from pathlib import Path

import forgi

from rnaglib.transforms import Transform


class SecondaryStructureTransform(Transform):
    """Compute secondary structure in dot-bracket notation
    for each chain in the RNA and store in a graph-level dictionary.
    Secondary structure assignments computed by forgi

    >>> from rnaglib.transforms import SecondaryStructureTransform
    >>> from rnaglib.data_loading import RNADataset
    >>> dset = RNADataset(debug=True)
    >>> T = SecondaryStructureTransform(dset.structures_path)
    >>> T(dset[0])
    >>> dset[0].graph['ss']
    """

    def __init__(self, structures_dir: Union[os.PathLike, str]):
        self.structures_dir = structures_dir
        pass

    def forward(self, rna_dict):
        pdbid = rna_dict["rna"].graph["pdbid"]
        ss_dict = {}
        seq_dict = {}
        try:
            out = forgi.load_rna(str(Path(self.structures_dir) / f"{pdbid}.cif"))
        except:
            pass
        else:
            for cg in out:
                (
                    pdb,
                    seq,
                    ss,
                ) = cg.to_fasta_string().split()
                chain_id = pdb.split("_")[-1].strip()
                ss_dict[chain_id] = ss
                seq_dict[chain_id] = seq
        rna_dict["rna"].graph["ss"] = ss_dict
        rna_dict["rna"].graph["seq"] = seq_dict
        return rna_dict

    pass
