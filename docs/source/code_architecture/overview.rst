Overview
~~~~~~~~

In this section we aim to give a bird-eye view of our API and a general understanding of the design of our library.
We refer the reader to the API reference for a more exhaustive and granular explanation of individual objects and functions.

.. raw:: html
    :file: ../images/fig_overview.svg

Our library is organized around RNA data, that can be processed and grouped into datasets, which can in turn be part of a task.
Transforms can be applied to RNAs or to RNA datasets.
You can find a more detailed explanation of these core objects in the following pages:


The RNA object
==============

Our RNA are stored as a simple networkx graph, with properties stored as node and edge features.
They are prepared by following the :doc:`steps described here<../data_reference/build_data>`.
You can find a reference of the features presented in our data :doc:`here<../data_reference/what_is>`, as well as a tutorial
on how to use this object :doc:`here<../tutorials/tuto_2.5d>`.

RNADataset
==========

A set of RNA objects can be grouped into a RNADataset object, which inherits the Pytorch Dataset object.
These RNAs can either live in the memory, or be found as files.
Datasets can be loaded and saved, subset and looped. They can also hold distance matrices among the different points it holds.
A more detailed description is provided :doc:`here<../code_architecture/dataset>`.

RNA Transforms
==============

We have introduced many transforms that can affect an RNA, split in four categories (annotate, filter, partition and represent).
A detailed description of these transforms is provided :doc:`here<../code_architecture/rna_transforms>`


RNADataset Transforms
=====================

Similar to RNATransforms, we also provide function that can alter a whole dataset.
The two main transformations are computing a distance between the points of the datasets, and filtering a dataset for redundancy.
A more detailed description is provided :doc:`here<../code_architecture/dataset_transforms>`.


Task
====

Finally, the Task object encapsulates a dataset for actionable learning.
It serves as the base object used to establish the benchmark tasks we propose.
It includes the definition of precise steps to build a dataset (with RNATransforms), postprocess it (with DatasetTransforms) and split it.
It also includes routines to get dataloaders looping over this dataset and computing metrics in a principled way.
A more detailed description is provided :doc:`here<../code_architecture/task>`.

