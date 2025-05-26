Quickstart
~~~~~~~~~~~


Get the data
______________

Once you have :doc:`installed <../pages/install>` RNAglib, you can fetch a pre-built database of RNA structures using the command line::

    rnaglib_download


By default you will get non-redundant RNA structures saved to ``~/.rnaglib``.

To obtain different versions or larger sets of RNAs have a look at the command line options ``rnaglib_download --help``.

Load single RNA
__________________

Annotations for each RNA are accessed through networkx graph objects.
You can load one RNA using ``rna_from_pdbid()``

.. code-block:: python

    >>> from rnaglib.dataset import rna_from_pdbid

    >>> rna = rna_from_pdbid("1fmn")
    >>> rna['rna'].graph
    {'name': '1fmn',
    'pdbid': '1fmn',
    'ligand_to_smiles': {'FMN': 'Cc1cc2c(cc1C)N(C3=NC(=O)NC(=O)C3=N2)CC(C(C(COP(=O)(O)O)O)O)O'},
    'ss': {'A': '..(((((......(((....))).....)))))..'},
    'seq': {'A': 'GGCGUGUAGGAUAUGCUUCGGCAGAAGGACACGCC'}
    }


See the data :doc:`tutorial <../tutorials/tuto_2.5d>` for more on the data.

Load an RNA Dataset
______________________

For machine learning purposes, we often want a collection of data objects in one place.
For that we have the ``RNADataset`` object.::

   from rnaglib.dataset import RNADataset

   dataset = RNADataset()


This object holds the same objects as above but also supports ML functionalities such as converting the RNAs to different representations (graphs, point clouds, voxels) and to different frameworks (dgl, torch, pytorch geometric)
See the ML :doc:`tutorial <../tutorials/tuto_tasks>` for more on model training and tasks.

Train a model on an RNA Task
____________________________________

The :mod:`rnaglib.tasks` library contains all utilities necessary for loading predefined tasks with splits and evaluation functions.::

    from rnaglib.tasks import get_task
    from rnaglib.transforms import GraphRepresentation
    from rnaglib.learning.task_models import PygModel

    # Load task, representation, and get loaders 
    task = get_task(root="my_root",
    task_id="rna_cm")
    model = PygModel.from_task(task)
    pyg_rep = GraphRepresentation(framework="pyg")

    task.add_representation(pyg_rep)
    train_loader, val_loader, test_loader = task.get_split_loaders(batch_size=8)

    for batch in train_loader:
        batch = batch['graph'].to(model.device)
        output = model(batch)

    test_metrics = model.evaluate(task, split='test')

