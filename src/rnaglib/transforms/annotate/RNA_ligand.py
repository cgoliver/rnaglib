import os
import pandas as pd
from rnaglib.transforms import AnnotationTransform

class LigandAnnotator(AnnotationTransform):
    name = "ligand_code"

    def __init__(
            self,
            data,
            **kwargs
    ):
        self.data = data
        super().__init__(**kwargs)
        
    def forward(self, rna_dict: dict) -> dict:
        node = next(iter(rna_dict["rna"].nodes()))
        ligand_code = int(self.data.loc[self.data.nid == node, "label"].values[0])
        rna_dict["rna"].graph[self.name] = ligand_code
        return rna_dict