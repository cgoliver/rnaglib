Quickstart
~~~~~~~~~~~


Get the data
______________

Once you have :doc:`installed <../pages/install>` RNAglib, you can fetch a pre-built dataset of RNA structures using the command line::

    rnaglib_download


By default you will get non-redundant RNA structures saved to ``~/.rnaglib``.

To obtain different versions or larger sets of RNAs have a look at the command line options ``rnaglib_download --help``.

Load single RNA
__________________

Annotations for each RNA are accessed through networkx graph objects.
You can load one RNA using ``rna_from_pdbid()``

.. code-block:: python

    >>> from rnaglib.utils import available_pdbids
    >>> from rnaglib.data_loading import rna_from_pdbid

    >>> pdbids = available_pdbids()
    >>> rna = rna_from_pdbid(pdbids[0])
    >>> rna['rna'].graph
    {'dbn': {'all_chains': {'num_nts': 143, 'num_chars': 144, 'bseq': 'GCCCGGAUAGCUCAGUCGGUAGAGCAGGGGAUUGAAAAUCCCCGUGUCCUUGGUUCGAUUCCGAGUCUGGGCAC&CGGAUAGCUCAGUCGGUAGAGCAGGGGAUUGAAAAUCCCCGUGUCCUUGGUUCGAUUCCGAGUCCGGGC', 'sstr': '(((((((..((((.....[..)))).(((((.......))))).....(((((..]....))))))))))))..&((((..((((.....[..)))).(((((.......))))).....(.(((..]....))).)))))...', 'form': 'AAAAAA...AA...A.......AAA.AAAA.......A.AAA......AAAAA..A....AAAAAAAAAAAA.-&.AA...AA...A.......AAA.AAAA.......A.AAA......AAAAA..A....A...AAAA.A.-'}...,

See the data :doc:`tutorial <../tutorials/tuto_2.5d>` for more on the data.

Load an RNA Dataset
______________________

For machine learning purposes, we often want a collection of data objects in one place.
For that we have the ``RNADataset`` object.::

   from rnaglib.data_loading import RNADataset

   dataset = RNADataset()


This object holds the same objects as above but also supports ML functionalities such as converting the RNAs to different representations (graphs, point clouds, voxels) and to different frameworks (dgl, torch, pytorch geometric)
See the ML :doc:`tutorial <../tutorials/tuto_tasks>` for more on model training and tasks.

Train a model on an RNA Task
____________________________________

The :mod:`rnaglib.tasks` library contains all utilities necessary for loading predefined tasks with splits and evaluation functions.::


    from torch.nn import BCELoss
    from rnaglib.tasks import BindingSite
    from rnaglib.transforms import GraphRepresentation

    # Load the task data and annotations
    ta = BindingSite("my_root")

    # Select a data representation and framework (see docs for support of other data modalities and deep learning frameworks)

    ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

    train_loader, val_loader, test_loader = ta.get_split_loaders()

    # most tasks ship with a dummy model for debugging, gives random outputs of the right shape
    model = ta.dummy_model

    # Access the predefined splits
    for batch in train_loader:
        pred = ta.dummy_model(batch["graph"]).flatten()
        y = batch["graph"].y
        loss = BCELoss()(y, pred)




