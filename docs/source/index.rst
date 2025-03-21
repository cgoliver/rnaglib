.. naglib documentation master file, created by
   sphinx-quickstart on Thu Aug 26 15:14:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 1
   :caption: Get Started 
   :hidden:

   Homepage <self>
   Installation <install>
   Quickstart <quickstart>
   Citing <citation>

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   A tour of RNA 2.5D graphs <tuto_2.5d>
   Using Benchmark Tasks <tuto_tasks>


.. toctree::
   :maxdepth: 2
   :caption: Data Reference 
   :hidden:

   What is an RNA 2.5D graph? <what_is>
   Available Benchmark Tasks <available_tasks>
   RNA Annotation Reference <rna_ref>
   RNA Transforms <tuto_transforms>
   How is the data built? <tuto_build>

.. toctree::
   :maxdepth: 2
   :caption: Package Reference
   :hidden:

   Algorithms <rnaglib.algorithms>
   Databases <rnaglib.prepare_data>
   Datasets <rnaglib.data_loading>
   Transforms <rnaglib.transforms>
   ML Tasks <rnaglib.tasks>
   Dataset Splitters <rnaglib.splitters>
   Visualization <rnaglib.drawing>
   Model Training <rnaglib.learning>
   Utils <rnaglib.utils>
   Configurations <rnaglib.config>

RNAGlib Official Documentation
================================

..
 This is a comment : contents:: Table of Contents


``RNAGlib`` (RNA Geometric Library) is a Python package for studying models of RNA 3D structures.

Core Features
-----------------

* Quick access to all available RNA 3D structures with annotations
* Rich functionality for 2.5D RNA graphs, point clouds, and voxels
* RNA graph visualization 
* Machine Learning benchmarking tasks 


Get started with RNAGlib
---------------------------

* :doc:`Install<install>`
* :doc:`Quickstart<quickstart>`

Tutorials
-----------

* :doc:`Learn about RNA 2.5D Graphs<what_is>`
* :doc:`Working with 2.5D graphs datasets <tuto_2.5d>`
* :doc:`Training machine learning models <tuto_ml>`
* :doc:`Using and building machine learning tasks <tuto_tasks>`

Package Structure
-----------------

-  :doc:`rnaglib.data_loading`: custom PyTorch dataloader and dataset implementations
-  :doc:`rnaglib.tasks`: prediction tasks for ML benchmarking.
-  :doc:`rnaglib.transforms`: process and modify RNA data
-  :doc:`rnaglib.learning`: learning routines and pre-built GCN models for the easiest use of the
-  :doc:`rnaglib.prepare_data`: processes raw PDB structures and
   builds a database of 2.5D graphs with full structural annotation
   package.
-  :doc:`rnaglib.drawing`: utilities for visualizing 2.5D graphs

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
