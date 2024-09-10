``rnaglib.utils``
=====================

General utilities for handling RNA structures and graphs.

.. automodule:: rnaglib.utils


Input/Output
------------------

Functions and writing and loading to/from disk.

.. autosummary::
   :toctree: generated/

    load_graph 
    download_graphs
    dump_json
    load_json
    graph_from_pdbid
    

Wrappers
------------

Wrappers for third-party executables.

.. warning::
   Make sure the necessary executables are in your os PATH.


.. autosummary::
    :toctree: generated/

    cdhit_wrapper
    rna_align_wrapper
    locarna_wrapper


PDB/mmCIF Utilities
------------------------


.. autosummary::
    :toctree: generated/

    cif_remove_residues 
