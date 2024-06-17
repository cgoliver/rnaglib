from .task import Task
from .task import ResidueClassificationTask, RNAClassificationTask
from .benchmark_binding_site import BenchmarkLigandBindingSiteDetection
from .binding_site import BindingSiteDetection, ProteinBindingSiteDetection, BindingDetection, ProteinBindingDetection, ChemicalModification
from .inverse_folding import InverseFolding, gRNAde

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
           'gRNAde'
           ]


classes = __all__

