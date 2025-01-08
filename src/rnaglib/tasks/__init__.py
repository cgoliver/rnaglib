from .task import Task
from .task import ClassificationTask, RNAClassificationTask, ResidueClassificationTask

from .RNA_GO.rna_go import RNAGo
from .RBP_Node.protein_binding_site import ProteinBindingSiteDetection
from .RNA_CM.chemical_modification import ChemicalModification
from .RNA_IF.inverse_folding import InverseFolding, gRNAde
from .RNA_Ligand.ligand_identity import LigandIdentification
from .RNA_Site.binding_site import BindingSiteDetection, BenchmarkBindingSiteDetection

__all__ = [
    "Task",
    "ResidueClassificationTask",
    "RNAClassificationTask",
    "BindingSiteDetection",
    "ProteinBindingSiteDetection",
    "ChemicalModification",
    "InverseFolding",
    "gRNAde",
    "LigandIdentification",
    "RNAGo",
]

classes = __all__
