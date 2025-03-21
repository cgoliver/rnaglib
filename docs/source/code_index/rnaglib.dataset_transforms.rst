``rnaglib.dataset_transforms``
==============================

Splitter objects compute a train, validation, and test split for a given dataset.



.. automodule:: rnaglib.dataset_transforms


Abstract classes
----------------

Ways to split your data.

.. autosummary::
   :toctree: generated/

    Splitter 
    ClusterSplitter
    RandomSplitter

Loading
-------

Tools for loading RNAs stored in an ``RNADataset`` batch-wise for deep learning models.

.. autosummary::
   :toctree: generated/

    Collater
    get_loader
