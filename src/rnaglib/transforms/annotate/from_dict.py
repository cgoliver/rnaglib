import os
import pandas as pd
from rnaglib.transforms import AnnotationTransform

class AnnotatorFromDict(AnnotationTransform):
    """Generic annotator which enables to add node-level features to a dataset by only using a dictionary mapping the node names to the desired node features.
    Enables to store information to build annotations in a JSON dictionary for instance.

    :param dict annotation_dict: dictionary of type {node_id:node_feature}
    :param str name: name to give to the feature resulting from this annotation
    """
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
        """Application of the transform to an RNA dictionary object

        :param dict rna_dict: the RNA dictionary which has to be annotated
        :return: the annotated version of rna_dict
        :rtype: dict
        """
        node = next(iter(rna_dict["rna"].nodes()))
        annotation = self.annotation_dict[node]
        rna_dict["rna"].graph[self.name] = annotation
        return rna_dict