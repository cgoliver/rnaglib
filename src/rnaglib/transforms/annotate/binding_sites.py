from rnaglib.transforms import AnnotationTransform
from networkx import set_node_attributes


class BindingSiteAnnotator(AnnotationTransform):
    def __init__(self, include_ions=False, include_covalent=False, cutoff=6.0):
        self.cutoff = cutoff
        self.include_ions = include_ions
        self.include_covalent = include_covalent

        self.bind_types = [f"binding_small-molecule-{cutoff}A"]
        if self.include_ions:
            self.bind_types.append(f"binding_ion_{cutoff}A")


    def forward(self, rna_dict: dict) -> dict:
        binding_sites = {
            node: self._has_binding_site(nodedata, self.cutoff)
            for node, nodedata in rna_dict["rna"].nodes(data=True)
        }
        set_node_attributes(rna_dict["rna"], binding_sites, "binding_site")
        return rna_dict

    def _has_binding_site(self, nodedata: dict, cutoff: float) -> bool:
        ligs = any(
            nodedata.get(binding_type) is not None
            for binding_type in self.bind_types 
        )
        return ligs

        if self.include_covalent:
            covalent = nodedata["is_modified"]
            return ligs or covalent
