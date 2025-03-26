``rnaglib.dataset_transforms``
==============================

Dataset transforms are trraansforms which process a whole dataset. They take as input a dataset and return the same dataset with some features being added or modified or some elements removed or added.



.. automodule:: rnaglib.dataset_transforms

Abstract classes
--------------------

Subclass these to create your own dataset transforms.

.. autosummary::
    :toctree: generated/

    DSTransform
    Splitter
    DistanceComputer
    RedundancyRemover

Splitters
---------

Ways to split your data (all of these are subclasses of `Splitter` abstract class).

.. autosummary::
   :toctree: generated/

    ClusterSplitter
    RandomSplitter
    NameSplitter

Distance computers
-------------------

Ways to add to the dataset a distance matrix indicating distance between the samples of the dataset (all of these are subclasses of `DistanceComputer` abstract class)

.. autosummary::
   :toctree: generated/

    CDHitComputer
    StructureDistanceComputer


Loading
-------

Tools for loading RNAs stored in an ``RNADataset`` batch-wise for deep learning models.

.. autosummary::
   :toctree: generated/

    Collater
    get_loader
    EdgeLoaderGenerator
    DefaultBasePairLoader
