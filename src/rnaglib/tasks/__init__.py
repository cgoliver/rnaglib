from .task import Task
from .task import ResidueClassificationTask, RNAClassificationTask

from .RNA_Family.rfam import RNAFamilyTask
from .RBP_Node.protein_binding_site import ProteinBindingSiteDetection
from .RNA_CM.chemical_modification import ChemicalModification
from .RNA_IF.gRNAde import gRNAde
from .RNA_IF.inverse_folding import InverseFolding
from .RNA_Ligand.ligand_identity import GMSM
from .RNA_Site.binding_site import BindingSiteDetection, BenchmarkBindingSiteDetection

__all__ = [
    'Task',
    'ResidueClassificationTask',
    'RNAClassificationTask',
    'BindingSiteDetection',
    'ProteinBindingSiteDetection',
    'ChemicalModification',
    'InverseFolding',
    'gRNAde',
    'GMSM',
    'RNAFamilyTask'
]

classes = __all__
