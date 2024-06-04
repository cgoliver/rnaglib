from .task import Task
from .task import ResidueClassificationTask, RNAClassificationTask
from .benchmark_binding_site import BenchmarkLigandBindingSiteDetection
from .binding_site import BindingSiteDetection, ProteinBindingSiteDetection, BindingDetection, ProteinBindingDetection
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
           'InverseFolding',
           'gRNAde'
           ]


classes = __all__

