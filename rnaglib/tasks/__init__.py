from .task import Task
from .task import ResidueClassificationTask
from .binding_site import BindingSiteDetection
from .benchmark_binding_site import BenchmarkLigandBindingSiteDetection

__all__ = [
           'Task',
           'ResidueClassificationTask',
           'BindingSiteDetection',
           'BenchmarkLigandBindingSiteDetection'
           ]


classes = __all__

