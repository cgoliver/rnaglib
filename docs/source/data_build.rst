Building a 2.5D graph database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial will take you through the steps needed to build a database
of fully annotated 2.5D graphs. We follow this exact process when building the 
data releases used by default in ``rnaglib``.

What you will need
-------------------

* An internet connection
* A working installation of x3dna-dssr in your system PATH version >= 2.0.0.
* An installation of ``rnaglib``

Command line
--------------

After you install ``rnaglib`` you will be able to execute the ``rnaglib_prepare_data`` script
from the command line.

This script assumes that you have a folder that stores PDB structures. If you do not
have any, just create an empty folder and ``rnaglib`` will populate it with RNA structures.

**Note: structures should be in mmCIF format**

::

    $ rnaglib_prepare_data  -S /path/to/structures -O /path/to/output -u

The ``-u`` flag will automatically download any structures missing from the structure
directory given by ``-S``. If you provided an empty folder, all RNA structures will
be downloaded. This will take a while. If you already have some structures and do not pass
the ``-u`` flag, then only existing structures will be annotated.

To do a quick debug run on a handful of structures, additionally pass the ``-d`` flag.

Once the process is complete, you will have a fully annotated database of RNA 2.5D graphs.

Processing steps
-----------------

Here we have a closer look at what is happening when you run ``rnaglib_preprare_data``.

The steps we take, starting from a PDBID file are the following:

1. Fetch the mmCIF from the local database or from RCSB-PDB.
2. Extract relevant metadata
3. Pass the mmCIF to ``x3dna-dssr`` to get base pairing annotations.
4. Pass the mmCIF to ``x3dna-dssr snap`` to get RNA-protein interfaces.
5. Populate a Networkx graph object with output from 3, 4 and additional annotations we extract such as RNA-small molecule binding sites.
6. Save the resulting graph
7. Extract subgraphs for pre-computed kernel functions. This requires the RNAs to be partitioned (chopped), followed by a subgraph extraction step.

Given optional filtering criteria (non-redundant) we modify the list of PDBs to use. Each of the filtering criteria
results in a different sub-folder. By default we create two sub-folders: ``my/root/graphs/all_graphs`` and ``my/root/graphs/NR``. 
The former contains all graphs obtained from structures in the given structure repository, and the latter only contains those that are also
found in the list of published `non-redundant structures <https://www.bgsu.edu/research/rna/databases/non-redundant-list.html>`_. 


