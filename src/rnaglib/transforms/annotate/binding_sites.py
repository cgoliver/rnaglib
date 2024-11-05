from rnaglib.transforms import AnnotationTransform
from networkx import set_node_attributes


class BindingSiteAnnotator(AnnotationTransform):
    def forward(self, rna_dict: dict) -> dict:
        binding_sites = {
            node: self._has_binding_site(nodedata)
            for node, nodedata in rna_dict["rna"].nodes(data=True)
        }
        set_node_attributes(rna_dict["rna"], binding_sites, "binding_site")
        return rna_dict

    @staticmethod
    def _has_binding_site(nodedata: dict) -> bool:
        return any(
            nodedata.get(binding_type) is not None
            for binding_type in ("binding_small-molecule", "binding_ion")
        )
