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
    """  Load a task object using its unique string ID. You can get a list of
    all available task IDs as a list by importing ``rnaglib.tasks.TASKS``.

    :param root: directory for holding the task data.
    :param task_id: string ID of desired task. 

    :returns: a Task object.
    """
    try:
        cls = TASK_MAP[task_id]
    except KeyError:
        raise ValueError(f"Task id {task_id} not found. Use one of {TASKS}")
    else:
        module = importlib.import_module("rnaglib.tasks")
        task = getattr(module, cls)
        return task(root=root, **task_kwargs)
