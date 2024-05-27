from .task import Task
from .task import ResidueClassificationTask
from .binding_site import BindingSiteDetection, ProteinBindingSiteDetection
from .benchmark_binding_site import BenchmarkLigandBindingSiteDetection

__all__ = [
           'Task',
           'ResidueClassificationTask',
           'BindingSiteDetection',
           'ProteinBindingSiteDetection',
           'BenchmarkLigandBindingSiteDetection'
           ]


classes = __all__

