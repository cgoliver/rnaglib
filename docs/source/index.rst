.. naglib documentation master file, created by
   sphinx-quickstart on Thu Aug 26 15:14:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 1
   :caption: Get Started 
   :hidden:

   Homepage <self>
   Installation <pages/install>
   Quickstart <pages/quickstart>
   Citing <pages/citation>

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   How to train a model <tutorials/tuto_tasks>
   A tour of RNA 2.5D graphs <tutorials/tuto_2.5d>

.. toctree::
   :maxdepth: 2
   :caption: Data Reference 
   :hidden:

   What is an RNA 2.5D graph? <data_reference/what_is>
   RNA Annotation Reference <data_reference/rna_ref>
   How is the data built? <data_reference/build_data>
   Available Benchmark Tasks <data_reference/available_tasks>

.. toctree::
   :maxdepth: 2
   :caption: A peek under the hood
   :hidden:

   Overview <code_architecture/overview>
   RNADataset <code_architecture/dataset>
   RNA Transforms <code_architecture/rna_transforms>
   RNADataset Transforms <code_architecture/dataset_transforms>
   Task <code_architecture/task>

.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :hidden:

   Algorithms <code_index/rnaglib.algorithms>
   Databases <code_index/rnaglib.prepare_data>
   Datasets <code_index/rnaglib.data_loading>
   Transforms <code_index/rnaglib.transforms>
   ML Tasks <code_index/rnaglib.tasks>
   Visualization <code_index/rnaglib.drawing>
   Model Training <code_index/rnaglib.learning>
   Utils <code_index/rnaglib.utils>
   Configurations <code_index/rnaglib.config>

RNAGlib Official Documentation
================================

..
 This is a comment : contents:: Table of Contents


``RNAGlib`` (RNA Geometric Library) is a Python package for studying models of RNA 3D structures.

.. figure:: https://raw.githubusercontent.com/cgoliver/rnaglib/c092768f19d32d40329ca822e59db5507ec245ca/images/tasksfig.png
   :alt: Tasks Figure
   :width: 800px
   :align: center

Core Features
-----------------

* Quick and detailed access to all available RNA 3D structures with annotations
* Train and benchmark deep learning models for RNA 3D structure-function tasks
* RNA graph visualization 
* Create fully reproducible custom datasets and tasks


Get started with RNAGlib
---------------------------

* :doc:`Install<pages/install>`
* :doc:`Quickstart<pages/quickstart>`


Tutorials
-----------

Those tutorials are meant to give you on operational overview of the library

* :doc:`Working with 2.5D graphs datasets <tutorials/tuto_2.5d>`
* :doc:`Using Benchmark Tasks <tutorials/tuto_tasks>`

Data reference
-----------------

Pages to read to better understand what is our data and how we build it.

* :doc:`A tour of RNA 2.5D graphs <data_reference/what_is>`
* :doc:`RNA Annotation Reference <data_reference/rna_ref>`
* :doc:`How is the data built? <data_reference/build_data>`
* :doc:`Available Benchmark Tasks <data_reference/available_tasks>`

A peek under the hood
------------------------

Pages to give you an understanding of the main objects shipping with RNAglib

* :doc:`Overview <code_architecture/overview>`
* :doc:`RNADataset <code_architecture/dataset>`
* :doc:`RNA Transforms <code_architecture/rna_transforms>`
* :doc:`RNADataset Transforms <code_architecture/dataset_transforms>`
* :doc:`Task <code_architecture/task>`


Package Structure
-----------------

-  :doc:`code_index/rnaglib.data_loading`: custom PyTorch dataloader and dataset implementations
-  :doc:`code_index/rnaglib.tasks`: prediction tasks for ML benchmarking.
-  :doc:`code_index/rnaglib.transforms`: process and modify RNA data
-  :doc:`code_index/rnaglib.learning`: learning routines and pre-built GCN models for the easiest use of the
-  :doc:`code_index/rnaglib.prepare_data`: processes raw PDB structures and
   builds a database of 2.5D graphs with full structural annotation
   package.
-  :doc:`code_index/rnaglib.drawing`: utilities for visualizing 2.5D graphs

Source Code and Contact
--------------------------

* `Source Code <https://github.com/cgoliver/rnaglib>`_.
* Contact rnaglib@cs.mcgill.ca 

Associated Repositories
-----------------------------------------------

`RNAmigos <https://github.com/cgoliver/RNAmigos>`_ : a research paper published in Nucleic Acid Research that
demonstrates the usefulness of 2.5D graphs for machine learning tasks, exemplified onto the drug discovery application.

`VeRNAl <https://github.com/cgoliver/vernal>`_ : a research paper published in Bioinformatics that uses learnt
vector representations of RNA subgraphs to mine structural motifs in RNA.


References
----------

#. Leontis, N. B., & Zirbel, C. L. (2012). Nonredundant 3D Structure Datasets for RNA Knowledge Extraction and Benchmarking. In RNA 3D Structure Analysis and Prediction N. Leontis & E. Westhof (Eds.), (Vol. 27, pp. 281–298). Springer Berlin Heidelberg. `doi:10.1007/978-3-642-25740-7\\\_13 <doi:10.1007/978-3-642-25740-7\_13>`__

.. |Example graph| image:: https://jwgitlab.cs.mcgill.ca/cgoliver/rnaglib/-/raw/main/images/Fig1.png


Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
