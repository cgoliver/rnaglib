from rnaglib.transforms import AnnotationTransform
from networkx import set_node_attributes


class BindingSiteAnnotator(AnnotationTransform):
    def __init__(self, include_ions=False, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        self.include_ions = include_ions

        self.bind_types = [f"binding_small-molecule-{cutoff}A"]
        if self.include_ions:
            self.bind_types.append(f"binding_ion_{cutoff}A")

    def forward(self, rna_dict: dict) -> dict:
        binding_sites = {node: self._has_binding_site(nodedata) for node, nodedata in rna_dict["rna"].nodes(data=True)}
        set_node_attributes(rna_dict["rna"], binding_sites, "binding_site")
        return rna_dict

    def _has_binding_site(self, nodedata: dict) -> bool:
        return any(nodedata.get(binding_type) is not None for binding_type in self.bind_types)
