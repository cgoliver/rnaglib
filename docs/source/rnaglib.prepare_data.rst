rnaglib.prepare\_data
=============================

If you have an mmCIF containing RNA stored locally and you wish to build a 2.5D graph that can be used in RNAglib you
can use the ``prepare_data`` module. To do so, you need to have ``x3dna-dssr`` executable in your ``$PATH`` which
requires a `license <http://x3dna.org/>`.

.. code-block:: python

    >>> from rnaglib.prepare_data.main import cif_to_graph

    >>> pdb_path = '../data/1aju.cif'
    >>> graph_nx = cif_to_graph(pdb_path)


Another useful functionality is to build the data systematically and in parallel. To do so, use :

.. code-block:: bash

    $ rnaglib_prepare_data -h

Annotations
----------------------------------------

.. automodule:: rnaglib.prepare_data.annotations
   :members:
   :undoc-members:
   :show-inheritance:

Chopper
------------------------------------

.. automodule:: rnaglib.prepare_data.chopper
   :members:
   :undoc-members:
   :show-inheritance:

Describe Datasets
-----------------------------------------------

.. automodule:: rnaglib.prepare_data.describe_datasets
   :members:
   :undoc-members:
   :show-inheritance:

DSSR 2
--------------------------------------------

.. automodule:: rnaglib.prepare_data.dssr_2_graphs
   :members:
   :undoc-members:
   :show-inheritance:

Filters
------------------------------------

.. automodule:: rnaglib.prepare_data.filters
   :members:
   :undoc-members:
   :show-inheritance:

Hash Check
----------------------------------------

.. automodule:: rnaglib.prepare_data.hash_check
   :members:
   :undoc-members:
   :show-inheritance:

Get Interfaces
---------------------------------------

.. automodule:: rnaglib.prepare_data.interfaces
   :members:
   :undoc-members:
   :show-inheritance:

Khop Annotations
-------------------------------------------

.. automodule:: rnaglib.prepare_data.khop_annotate
   :members:
   :undoc-members:
   :show-inheritance:

Retrieve PDB Structures
-------------------------------------------------

.. automodule:: rnaglib.prepare_data.retrieve_structures
   :members:
   :undoc-members:
   :show-inheritance:

Build Data
---------------------------------

.. automodule:: rnaglib.prepare_data.main
   :members:
   :undoc-members:
   :show-inheritance:

