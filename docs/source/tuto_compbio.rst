Working with rnaglib datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `rnaglib.data_loading.RNADataset` object builds and provides access to collections of RNAs.
You can choose to provide your own annotaed RNAs or use the data builds distributed with RNAglib.
The base dataset object stores the raw annotations.
We can use these annotations directly, or we can conver them to
different data representations (e.g. graphs, voxels, point clouds) as well as cast to different data science and ML frameworks (DGL, pytorch-geometric, networkx).

Creating a dataset
===================

To create a dataset using our hosted data simply instantiate the `RNADataset` object.

.. code-block:: python
   
   from rnaglib.data_loading import RNADataset

   dataset = RNADataset()


Different datasets can be specified using the following options:

* `version='x.x.x'`: which data build to load
* `nr=False`:  by default, we only load non-redundant structures, if you want all structures in the PDB, set this flag to `False`


Accessing items
=================

Datasets can be indexed like a list or you can inspect an individual RNA by its PDBID.
Thre result of indexing 

.. code-block:: python

    rna_1 = dataset[3]
    pdbid = dataset.available_pdbids[3]
    rna_2 = dataset.get_pdbid(pdbid)

The returned object is a networkx graph which you can inspect as shown in :doc:`this tutorial<tuto_2.5d>`.

Data representations
=======================

The dataset object can be converted to different representations which form the basis of different machine learning architectures:

* Graph: adjacency information with edges as base pairs and backbones
* Voxel: a set of fixed-size volumes defined by the 5' carbon coordinates
* Point-cloud: raw coordinates of 5' carbons


The `rnaglib.reprsentations.Representation` class holds the logic for converting a dataset to one of the above representations and users can easily sub-class this to create their own representations.





