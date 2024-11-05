from rnaglib.transforms import AnnotationTransform
from networkx import set_node_attributes


class DummyAnnotator(AnnotationTransform):
    def forward(self, rna_dict: dict) -> dict:
        dummy_values = {node: 1 for node in rna_dict["rna"].nodes()}
        set_node_attributes(rna_dict["rna"], dummy_values, "dummy")
        return rna_dict
