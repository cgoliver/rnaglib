``rnaglib.data_loading``
=========================

Tools for loading and creating collections of RNAs.

.. automodule:: rnaglib.data_loading


RNA Objects
-------------

A single RNA is stored as a dictionary.

.. autosummary::
   :toctree: generated/

    rna_from_pdbid

RNA Dataset
--------------


This is the main object used for holding collections of RNAs. The ``RNAdataset`` object draws from a database.

.. autosummary::
   :toctree: generated/

    RNADataset



Loading
---------

Tools for loading RNAs stored in an ``RNADataset`` batch-wise for deep learning models.



.. autosummary::
   :toctree: generated/

    Collater
    get_loader
    get_inference_loader
    EdgeLoaderGenerator 
    DefaultBasePairLoader
