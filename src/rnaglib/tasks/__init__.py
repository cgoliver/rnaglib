TASKS = ["rna_cm",
         "rna_go",
         "rna_if",
         "rna_if_bench",
         "rna_ligand",
         "rna_prot",
         "rna_site",
         "rna_site_bench" 
         ]


from .task import Task
from .task import ClassificationTask, RNAClassificationTask, ResidueClassificationTask

from .RNA_CM.chemical_modification import ChemicalModification
from .RNA_GO.rna_go import RNAGo
from .RNA_IF.inverse_folding import InverseFolding, gRNAde
from .RNA_Ligand.ligand_identity import LigandIdentification
from .RNA_Prot.protein_binding_site import ProteinBindingSite
from .RNA_Site.binding_site import BindingSite, BenchmarkBindingSite
# from .RNA_VS.task import VirtualScreening

from .utils import get_task

__all__ = [
    "Task",
    "RNAClassificationTask",
    "ResidueClassificationTask",
    "ChemicalModification",
    "RNAGo",
    "InverseFolding",
    "gRNAde",
    "LigandIdentification",
    "ProteinBindingSite",
    "BindingSite",
    "BenchmarkBindingSite",
    # "VirtualScreening",
]

classes = __all__


