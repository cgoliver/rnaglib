from networkx import set_node_attributes

from rnaglib.transforms import AnnotationTransform
from rnaglib.config import NATURAL_RNA_MODIFICATIONS


class BiologicalModificationAnnotator(AnnotationTransform):
    """Annotate each residue with whether it is a natural, enzymatically installed RNA modification.

    The database-level ``is_modified`` flag is set from residue-name length and so also
    fires on synthetic analogs, DNA residues and ligands. This transform derives a stricter
    node feature, ``is_biological_modification``, by matching the raw residue code
    (``nt_full``) against a curated set of natural modifications, without any database rebuild.

    :param allowed_modifications: iterable of PDB residue codes to treat as positives.
        Defaults to :data:`rnaglib.config.NATURAL_RNA_MODIFICATIONS`. Pass a wider set to
        include, e.g., synthetic analogs for a different scope.
    """

    def __init__(self, allowed_modifications=None):
        super().__init__()
        self.allowed = (
            NATURAL_RNA_MODIFICATIONS
            if allowed_modifications is None
            else frozenset(allowed_modifications)
        )

    def forward(self, rna_dict: dict) -> dict:
        flags = {
            node: nodedata.get("nt_full") in self.allowed
            for node, nodedata in rna_dict["rna"].nodes(data=True)
        }
        set_node_attributes(rna_dict["rna"], flags, "is_biological_modification")
        return rna_dict
