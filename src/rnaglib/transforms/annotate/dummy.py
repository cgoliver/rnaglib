"""Annotator for adding dummy values to RNA graphs."""

from networkx import set_node_attributes
from rnaglib.transforms import AnnotationTransform


class DummyAnnotator(AnnotationTransform):
    """Add a dummy attribute with value 1 to all nodes in an RNA graph."""

    def forward(self, rna_dict: dict) -> dict:
        dummy_values = {node: 1 for node in rna_dict["rna"].nodes()}
        set_node_attributes(rna_dict["rna"], dummy_values, "dummy")
        return rna_dict
