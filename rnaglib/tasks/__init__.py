from .task import Task
from .task import ResidueClassificationTask, RNAClassificationTask

from .RBP_Node.protein_binding_site import ProteinBindingSiteDetection
from .RBP_Graph.protein_binding import ProteinBindingDetection
from .RNA_CM.chemical_modification import ChemicalModification
from .RNA_Site.binding_site import BindingSiteDetection
from .RNA_Site.benchmark_binding_site import BenchmarkLigandBindingSiteDetection
from .RNA_Ligand.ligand_identity import GMSM
#from .RNA_VS todo
from .RNA_IF.inverse_folding import InverseFolding
from .RNA_IF.gRNAde import gRNAde



__all__ = [
           'Task',
           'ResidueClassificationTask',
           'RNAClassificationTask',
           'BindingSiteDetection',
           'ProteinBindingSiteDetection',
           'BenchmarkLigandBindingSiteDetection',
           'BindingDetection',
           'ProteinBindingDetection',
           'ChemicalModification',
           'InverseFolding',
           'gRNAde',
           'GMSM'
           ]


classes = __all__

