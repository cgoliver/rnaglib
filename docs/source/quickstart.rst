Quickstart
~~~~~~~~~~~


Get the data
______________

Once you have :doc:`installed <install>` RNAglib, you can fetch a pre-built dataset of RNA structures using the command line::

    rnaglib_download


By default you will get non-redundant RNA structures saved to ``~/.rnaglib``.

To obtain different versions or larger sets of RNAs have a look at the command line options ``rnaglib_download --help``.

Load single RNA
__________________

Annotations for each RNA are accessed through networkx graph objects.
You can load one RNA using ``graph_from_pdbid()``

.. code-block:: python

    >>> from rnaglib.utils import available_pdbids
    >>> from rnaglib.utils import graph_from_pdbid

    >>> pdbids = available_pdbids()
    >>> rna = graph_from_pdbid(pdbids[0])
    >>> rna
    DiGraph with 69 nodes and 194 edges
    >>> rna.graph
    {'dbn': {'all_chains': {'num_nts': 143, 'num_chars': 144, 'bseq': 'GCCCGGAUAGCUCAGUCGGUAGAGCAGGGGAUUGAAAAUCCCCGUGUCCUUGGUUCGAUUCCGAGUCUGGGCAC&CGGAUAGCUCAGUCGGUAGAGCAGGGGAUUGAAAAUCCCCGUGUCCUUGGUUCGAUUCCGAGUCCGGGC', 'sstr': '(((((((..((((.....[..)))).(((((.......))))).....(((((..]....))))))))))))..&((((..((((.....[..)))).(((((.......))))).....(.(((..]....))).)))))...', 'form': 'AAAAAA...AA...A.......AAA.AAAA.......A.AAA......AAAAA..A....AAAAAAAAAAAA.-&.AA...AA...A.......AAA.AAAA.......A.AAA......AAAAA..A....A...AAAA.A.-'}...,

See the data :doc:`tutorial <tuto_2.5d>` for more on the data.

Load an RNA Dataset
______________________

For machine learning purposes, we often want a collection of data objects in one place.
For that we have the ``RNADataset`` object.::

   from rnaglib.data_loading import RNADataset

   dataset = RNADataset()


This object holds the same objects as above but also supports ML functionalities such as converting the RNAs to different representations (graphs, point clouds, voxels) and to different frameworks (dgl, torch, pytorch geometric)
See the ML :doc:`tutorial <tuto_ml>` for more on model training and tas.


