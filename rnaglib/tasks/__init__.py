from .task import Task
from .task import ResidueClassificationTask, RNAClassificationTask
from .binding_site import BindingSiteDetection, ProteinBindingSiteDetection, BindingDetection, ProteinBindingDetection
from .benchmark_binding_site import BenchmarkLigandBindingSiteDetection

__all__ = [
           'Task',
           'ResidueClassificationTask',
           'RNAClassificationTask',
           'BindingSiteDetection',
           'ProteinBindingSiteDetection',
           'BenchmarkLigandBindingSiteDetection',
           'BindingDetection',
           'ProteinBindingDetection'
           ]


classes = __all__

