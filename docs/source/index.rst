.. rnaglib documentation master file, created by
   sphinx-quickstart on Thu Aug 26 15:14:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Homepage
   :hidden:

   Homepage <self>
   Quickstart <quickstart>
   Data <rnaglib.data>

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   Benchmark <rnaglib.benchmark>
   Config <rnaglib.config>
   Data Loading <rnaglib.data_loading>
   Drawing <rnaglib.drawing>
   Examples <rnaglib.examples>
   Ged <rnaglib.ged>
   Kernels <rnaglib.kernels>
   Learning <rnaglib.learning>
   Prepare Data <rnaglib.prepare_data>
   Utils <rnaglib.utils>





Welcome to RNAGlib's Documentation!
----------------------------------------
..
 This is a comment : contents:: Table of Contents


``RNAglib`` is a Python package for studying RNA 2.5D structures.
Functionality includes automated data loading, analysis, visualization, ML model building
and benchmarking.

-  What are RNA 2.5D structures?

RNA 2.5D structures are discrete graph-based representations of atomic coordinates derived
from techniques such as X-ray crystallography and NMR. This type of representation encodes
all possible base pairing interactions which are known to be crucial for understanding RNA function.

-  Why use RNA 2.5D data?

The benefit is twofold. When dealing with RNA 3D data, a representation centered on
base pairing is a very natural prior which has been shown to carry important signals for
complex interactions, and can be directly interpreted.
Second, adopting graph representations lets us take advantage of many powerful algorithmic tools
such as graph neural networks and graph kernels.

-  What type of functional data is included?

The graphs are annotated with graph, node, and edge-level attributes.
These include, but are not limited to:

-  Secondary structure (graph-level)
-  Protein binding (node-level)
-  Small molecule binding (node-level)
-  Chemical modifications (node-level)
-  3-D coordinates(node-level)
-  Leontis-westhof base pair geometry classification (edge-level)

We provide a visualization of what the graphs in this database contain.
A more detailed description of the data is presented in :doc:`rnaglib.data`.
|Example graph|

Package Structure
-----------------

-  :doc:`rnaglib.prepare_data`: processes raw PDB structures and
   builds a database of 2.5D graphs with full structural annotation
-  :doc:`rnaglib.data_loading`: custom PyTorch dataloader implementations
-  :doc:`rnaglib.learning`: learning routines and pre-built GCN models for the easiest use of the
   package.
-  :doc:`rnaglib.drawing`: utilities for visualizing 2.5D graphs
-  :doc:`rnaglib.ged`: custom graph similarity functions
-  :doc:`rnaglib.kernels`: custom local neighbourhood similarity functions


Installation
------------

The package can be cloned and the source code used directly.
We also deploy it as a pip package and recommend using this install in conda environments.

If one wants to use GPU support, one should install `Pytorch <https://pytorch.org/get-started/locally/>`__
and `DGL <https://www.dgl.ai/pages/start.html>`__ with the appropriate options.
Otherwise you can just skip this step and the pip installs of Pytorch and DGL will be used.

Then, one just needs to run :

::

    pip install rnaglib

and can start using the packages functionalities by importing them in one's python script.

To have an idea on how to use the main functions of RNAGlib, please visit :doc:`quickstart`

Associated Repositories:
------------------------
`RNAMigos <https://github.com/cgoliver/RNAmigos>`__ : a research paper published in Nucleic Acid Research that
demonstrates the usefulness of 2.5D graphs for machine learning tasks, exemplified onto the drug discovery application.

`VeRNAl <https://github.com/cgoliver/vernal>`__ : a research paper published in Bioinformatics that uses learnt
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
