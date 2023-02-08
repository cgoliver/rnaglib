=====
Data tutorial
=====

Overview of the 2.5D Graphs
------
First, let us have a look at the 2.5d graph object from a code perspective. We use networkx to represent this graph.
Once the graphs are downloaded, they can be fetched directly using their PDBID.
Since nodes represent nucleotides, the node data dictionary will include features such as nucleotide type,
position, 3D coordinates, etc...
Nodes are assigned an ID in the form ``<pdbid.chain.position>``.
Using node IDs we can access node and edge attributes as dictionary keys.

.. code-block:: python

   >>> from rnaglib.utils.graph_io import graph_from_pdbid
   >>> G = graph_from_pdbid("4nlf")
   >>> G.nodes['4nlf.A.2647']
    {'index': 1, 'index_chain': 1, 'chain_name': 'A', 'nt_resnum': 2647, 'nt_name': 'U', 'nt_code': 'U',
     'nt_id': 'A.U2647', 'nt_type': 'RNA', 'dbn': '.',
     'summary': "anti,~C2'-endo,non-stack,non-pair-contact,ss-non-loop,splayed-apart",
     'alpha': None, 'beta': None, 'gamma': 48.553, 'delta': 145.549, 'epsilon': -136.82, 'zeta': 106.418,
     'epsilon_zeta': 116.762, 'bb_type': '--', 'chi': -137.612, 'glyco_bond': 'anti',
     'C5prime_xyz': [-1.821, 8.755, -0.245], 'P_xyz': [None, None, None], 'form': '.', 'ssZp': 1.669,
     'Dp': 1.751, 'splay_angle': 88.977, 'splay_distance': 13.033, 'splay_ratio': 0.702, 'eta': None,
     'theta': None, 'eta_prime': None, 'theta_prime': None, 'eta_base': None, 'theta_base': None,
     'v0': -21.744, 'v1': 36.502, 'v2': -36.488, 'v3': 24.597, 'v4': -2.014, 'amplitude': 37.908,
     'phase_angle': 164.267, 'puckering': "C2'-endo", 'sugar_class': "~C2'-endo", 'bin': 'inc',
     'cluster': '__', 'suiteness': 0.0, 'filter_rmsd': 0.1,
     'frame': {'rmsd': 0.007, 'origin': [24.09, 9.076, -5.96], 'x_axis': [0.09, 0.563, -0.822],
       'y_axis': [-0.848, -0.389, -0.359], 'z_axis': [-0.522, 0.729, 0.443], 'quaternion': [0.535, -0.509, 0.14, 0.66]},
     'sse': {'sse': None}, 'binding_protein': None, 'binding_ion': None, 'binding_small-molecule': None}

To get more information on what each of these fields refer to, please visit :doc:`rnaglib.data`.

Downloading Prebuilt Data Sets
------
To perform machine learning one needs RNA data. We provide a way of obtaining a 2.5D graph from local PDB files (see below).
However since this construction is computationally expensive at database scale, we offer pre-built databases.
We however offer three possibilities to directly access pre-built databases :

-  A download script ships with the install, run : ``$ rnaglib_download -h``
-  Direct download at the address :
   http://rnaglib.cs.mcgill.ca/static/datasets/iguana.tar.gz
-  Dynamic download : if one instantiates a dataloader and the data
   cannot be found, the corresponding data will be automatically downloaded and cached

Because of this last option, after installing our tool with pip, one can start learning on RNA data without extra steps.

Building the Data
------

------------------
What you will need
------------------

* An internet connection
* A working installation of x3dna-dssr in your system PATH version >= 2.0.0.
* An installation of ``rnaglib``
* One or more mmcif files of RNA structures to turn into your 2.5D graph.

------
Building a single 2.5D Graph
------

If you have an mmCIF containing RNA stored locally and you wish to build a 2.5D graph that can be used in RNAglib you
can use the ``prepare_data`` module.
To do so, you need to have ``x3dna-dssr`` executable in your ``$PATH`` which requires a `license <http://x3dna.org/>`.
The first option is to use the library from a python script, following the example :

.. code-block:: python

    >>> from rnaglib.prepare_data.main import cif_to_graph

    >>> pdb_path = '../data/1aju.cif'
    >>> graph_nx = cif_to_graph(pdb_path)

Another possibility is to use the shell function that ships with rnaglib.

::

    $ rnaglib_prepare_data  --one_mmcif $PATH_TO_YOUR_MMCIF -O /path/to/output

------
Building a dataset of 2.5D Graphs
------

Another useful functionality is to build the data systematically and in parallel. To do so, use :

::

    $ rnaglib_prepare_data -h

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

Here we have a closer look at what is happening when you run ``rnaglib_prepare_data``.

The steps we take, starting from a PDBID file are the following:

1. Fetch the mmCIF from the local database or from RCSB-PDB.
3. Pass the mmCIF to ``x3dna-dssr`` to get base pairing annotations. (See :doc:`rnaglib.prepare_data.dssr_2_graphs<code>`)
4. Pass the mmCIF to ``x3dna-dssr snap`` to get RNA-protein interfaces. (See :doc:`rnaglib.prepare_data.dssr_2_graphs<code>`)
5. Populate a Networkx graph object with output from 3, 4
6. Add additional annotations, such as RNA-small molecule binding sites and RNA-ions interactions. (See :doc:`rnaglib.prepare_data.annotations<code>`)
7. Save the resulting graph in the json format
8. Extract subgraphs for pre-computed kernel functions. This requires the RNAs to be partitioned (chopped, see :doc:`rnaglib.prepare_data.chopper<code>`), followed by a subgraph extraction step (see :doc:`rnaglib.prepare_data.khop_annotate<code>`).

Given optional filtering criteria (non-redundant) we modify the list of PDBs to use.
Each of the filtering criteria results in a different sub-folder.
By default we create two sub-folders: ``my/root/graphs/all_graphs`` and ``my/root/graphs/NR``.
The former contains all graphs obtained from structures in the given structure repository, and the latter only contains those that are also
found in the list of published `non-redundant structures <https://www.bgsu.edu/research/rna/databases/non-redundant-list.html>`_.



