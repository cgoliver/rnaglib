import os
import pandas as pd
from rnaglib.transforms import AnnotationTransform

class AnnotatorFromDict(AnnotationTransform):

    def __init__(
            self,
            annotation_dict,
            name,
            **kwargs
    ):
        self.annotation_dict = annotation_dict
        self.name = name
        super().__init__(**kwargs)
        
    def forward(self, rna_dict: dict) -> dict:
        node = next(iter(rna_dict["rna"].nodes()))
        annotation = self.annotation_dict[node]
        rna_dict["rna"].graph[self.name] = annotation
        return rna_dict