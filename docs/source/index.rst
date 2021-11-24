.. rnaglib documentation master file, created by
   sphinx-quickstart on Thu Aug 26 15:14:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RNAGlib's Documentation!
===================================
.. contents:: Table of Contents

RNAglib
=======

| ``RNAglib`` is a Python package for studying RNA 2.5D structures.
Functionality includes automated data loading, analysis,
| visualization, ML model building and benchmarking.

-  What are RNA 2.5D structures?

| RNA 2.5D structures are discrete graph-based representations of atomic
coordinates derived from techniues such as X-ray
| crystallography and NMR. This type of representation encodes all
possible base pairing interactions which are known to
| be crucial for understanding RNA function.

-  Why use RNA 2.5D data?

| The benefit is twofold. When dealing with RNA 3D data, a
representation centered on base pairing is a very natural prior
| which has been shown to carry important signals for complex
interactions, and can be directly interpreted. Second,
| adopting graph representations lets us take advantage of many powerful
algorithmic tools such as graph neural networks
| and graph kernels.

-  What type of functional data is included?

The graphs are annotated with graph, node, and edge-level attributes.
These include, but are not limited to:

-  Secondary structure (graph-level)
-  Protein binding (node-level)
-  Small molecule binding (node-level)
-  Chemical modifications (node-level)
-  3-d coordinates(node-level)
-  Leontis-westhof base pair geometry classification (edge-level)

Package Structure
-----------------

-  :doc:`rnaglib.prepare_data/`: processes raw PDB structures and
   builds a database of 2.5D graphs with full structural annotation
-  :doc:`rnaglib.data_loading`: custom PyTorch dataloader implementations
-  :doc:`rnaglib.models`: pre-built GCN models.
-  :doc:`rnaglib.learning`: learning routines for the easiest use of the
   package.
-  :doc:`rnaglib.drawing`: utilities for visualizing 2.5D graphs
-  :doc:`rnaglib.ged`: custom graph similarity functions
-  :doc:`rnaglib.kernels`: custom local neighbourhood similarity functions

Data scheme
-----------

| A more detailed description of the data is presented in :doc:`rnaglib.prepare_data`. It comes along with instructions on how to produce the data from public databases. However since this construction is computationally expensive, we offer a pre-built database.

We provide a visualization of what the graphs in this database contain:

|Example graph|

Installation
------------

Code
~~~~

| The package can be cloned and the source code used directly. We also deploy it as a pip package and recommend using this install in conda environments.

| If one wants to use GPU support, one should install `Pytorch <https://pytorch.org/get-started/locally/>`__ and `DGL <https://www.dgl.ai/pages/start.html>`__ with the appropriate options. Otherwise you can just skip this step and the pip installs of Pytorch and DGL will be used.

Then, one just needs to run :

::

    pip install rnaglib

Then one can start using the packages functionalities by importing them
in one's python script.

Data
~~~~

| To perform machine learning one needs RNA data. The instructions to
produce this data are presented in prepare\_data. We
| however offer two possibilities to directly access pre-built databases
:

-  Direct download at the address :
   http://rnaglib.cs.mcgill.ca/static/datasets/iguana
-  In code download : if one instantiates a dataloader and the data
   cannot be found, the corresponding data will be automatically downloaded and cached

| Because of this second option, after installing our tool with pip, one can start learning on RNA data without extra steps.

Example usage
-------------

To provide the user with a hands on tutorial, we offer two example
learning pipelines in :doc:`rnaglib.examples`.

If one has run the pip installation, just run :

::

    $ rnaglib_first
    $ rnaglib_second

Otherwise, after cloning the repository, run :

::

    $ cd examples
    $ python first_example.py
    $ python second_example.py

| You should see data getting downloaded and networks being trained. The
first example is a basic supervised model training to predict protein binding nucleotides. The second one starts by an unsupervised phase that pretrains the network and then performs this supervised training in a principled way, suitable for benchmarking its performance. This simple code was used to produce the benchmark values presented in the
paper.

Building a 2.5D Graph from a local PDB
-----

If you have a PDB containing RNA stored locally and you wish to build a 2.5D graph that can be used in RNAglib you can use the `prepare_data` module.
To do so, you need to have `x3dna-dssr` executable in your `$PATH` which requires a `license <http://x3dna.org/> `.

.. code-block:: python
    from rnaglib.prepare_data import do_one

    pdb_path = '../data/1aju.cif'
    graph = do_one(pdb_path)
    
   


Visualization
-------------

| To visualize the 2.5D graphs in the format described above, we have implemented a drawing toolbox with several functions. The easiest way to use it in your application is to call ``rnaglib.drawing.draw(graph, show=True)``. A functioning installation of Latex is needed to plot the graphs. If one encounters troubles, this `tutorial <https://matplotlib.org/stable/tutorials/text/usetex.html>`__ explains how to use matplotlib with Latex.

Ged
---

| GED is the gold standard of graph comparisons. We have put our ged implementation as a part of networkx, and offer in ``rnaglib/ged`` the weighting scheme we propose to compare 2.5D graphs. One can call ``rnaglib.ged.ged()`` on two graphs to compare them. However, due to the exponential complexity of the comparison, the maximum size of the graphs should be around ten nodes, making it more suited for comparing graphlets or
subgraphs.

Associated Repositories:
------------------------

`VeRNAl <https://github.com/cgoliver/vernal>`__

`RNAMigos <https://github.com/cgoliver/RNAmigos>`__

References
----------

#. Leontis, N. B., & Zirbel, C. L. (2012). Nonredundant 3D Structure Datasets for RNA Knowledge Extraction and Benchmarking. In RNA 3D Structure Analysis and Prediction N. Leontis & E. Westhof (Eds.), (Vol. 27, pp. 281–298). Springer Berlin Heidelberg. `doi:10.1007/978-3-642-25740-7\\\_13 <doi:10.1007/978-3-642-25740-7\_13>`__

.. |Example graph| image:: https://jwgitlab.cs.mcgill.ca/cgoliver/rnaglib/-/raw/main/images/Fig1.png


Modules
=======
.. toctree::
    Benchmark <rnaglib.benchmark>
    Config <rnaglib.config>
    Data Loading <rnaglib.data_loading>
    Data <rnaglib.data>
    Drawing <rnaglib.drawing>
    Examples <rnaglib.examples>
    Ged <rnaglib.ged>
    Kernels <rnaglib.kernels>
    Learning <rnaglib.learning>
    Prepare Data <rnaglib.prepare_data>
    Utils <rnaglib.utils>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
