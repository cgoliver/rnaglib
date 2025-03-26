from rnaglib.transforms import AnnotationTransform
from networkx import set_node_attributes


class BindingSiteAnnotator(AnnotationTransform):
    """Annotation transform adding to each node of the dataset a binary node feature indicating whether it is part of a binding site

    :param bool include_ions: if set to False, only small-molecule-binding RNA residues are considered part of a binding site. If set to True, ion-binding RNA residues are also considered part of a binding site
    :param float cutoff: the maximal distance (in Angstroms) between an RNA residue and any small molecule or ion atom such that the RNA residue is considered part of a binding site (either 4.0, 6.0 or 8.0, default 6.0)
    """
    def __init__(self, include_ions=False, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        self.include_ions = include_ions

        self.bind_types = [f"binding_small-molecule-{cutoff}A"]
        if self.include_ions:
            self.bind_types.append(f"binding_ion_{cutoff}A")

    def forward(self, rna_dict: dict) -> dict:
        """Application of the transform to an RNA dictionary object

        :param dict rna_dict: the RNA dictionary which has to be annotated with binding site information
        :return: the annotated version of rna_dict
        :rtype: dict
        """
        binding_sites = {node: self._has_binding_site(nodedata) for node, nodedata in rna_dict["rna"].nodes(data=True)}
        set_node_attributes(rna_dict["rna"], binding_sites, "binding_site")
        return rna_dict

    def _has_binding_site(self, nodedata: dict) -> bool:
        """Returns whether an RNA residue is binding for at least one of the binding types taken into account by the present BindingSiteAnnotator object

        :param dict nodedata: the dictionary containing the data describing one RNA residue node containing annotations regarding its binding for all binding types taken into account by the BindingSiteAnnotator object
        :return: boolean indicating whether the node described by nodedata is part of the binding site according to one of the binding types described in self.bind_types
        :rtype: bool
        """
        return any(nodedata.get(binding_type) is not None for binding_type in self.bind_types)
