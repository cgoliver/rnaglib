import importlib

from rnaglib.tasks import Task
from rnaglib.tasks import TASKS

TASK_MAP = {'rna_cm': 'ChemicalModification',
            'rna_prot': 'ProteinBindingSite',
            'rna_ligand': 'LigandIdentification',
            'rna_site': 'BindingSite',
            'rna_site_bench': 'BenchmarkBindingSite',
            'rna_if': 'InverseFolding',
            'rna_if_bench': 'gRNAde',
            'rna_go': 'RNAGo'
            }


def get_task(root=".", task_id="rna_cm", **task_kwargs) -> Task:
    """Load a task object using its unique string ID.

    You can get a list of all available task IDs as a list by importing ``rnaglib.tasks.TASKS``.

    :param root: Directory for holding the task data
    :param task_id: String ID of desired task (e.g., "rna_cm", "rna_go")
    :param task_kwargs: Additional keyword arguments to pass to the task constructor
    :return: Task object instance
    """
    try:
        cls = TASK_MAP[task_id]
    except KeyError:
        raise ValueError(f"Task id {task_id} not found. Use one of {TASKS}")
    else:
        module = importlib.import_module("rnaglib.tasks")
        task = getattr(module, cls)
        return task(root=root, **task_kwargs)
