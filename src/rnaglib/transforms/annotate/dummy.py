"""Annotator for adding dummy values to RNA graphs."""

from networkx import set_node_attributes
from rnaglib.transforms import AnnotationTransform


class DummyAnnotator(AnnotationTransform):
    """Add a dummy attribute with value 1 to all nodes in an RNA graph."""

    def __init__(self, graph_level=False, **kwargs):
        self.graph_level = graph_level
        super().__init__(**kwargs)

    def forward(self, rna_dict: dict) -> dict:
        if self.graph_level:
            rna_dict["rna"].graph["dummy"] = 1
            return rna_dict
        dummy_values = {node: 1 for node in rna_dict["rna"].nodes()}
        set_node_attributes(rna_dict["rna"], dummy_values, "dummy")
        return rna_dict
