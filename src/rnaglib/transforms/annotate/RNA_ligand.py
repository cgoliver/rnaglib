from rnaglib.transforms import AnnotationTransform
from networkx import set_node_attributes


class LigandAnnotator(AnnotationTransform):
    def forward(self, rna_dict: dict) -> dict:
        ligand_codes = {
            node: int(self.data.loc[self.data.nid == node, "label"].values[0])
            for node, nodedata in rna_dict["rna"].nodes(data=True)
        }
        set_node_attributes(rna_dict["rna"], ligand_codes, "ligand_code")
        return rna_dict
