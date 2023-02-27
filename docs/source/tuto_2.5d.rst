Working with 2.5D graphs
~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we know :doc:`what is an RNA 2.5D graph<what_is>` we can inspect the graph using `rnaglib`.

Fetching a hosted graph
--------------------------

The libray ships with some pre-built datasets which you can download with the following command line:

..
        $ rnaglib_download


This will download the default data distribution to `~/.rnaglib`

Using a local RNA structure.
-----------------------------


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



Overview of the 2.5D Graphs
-----------------------------

First, let us have a look at the 2.5d graph object from a code perspective.
We use networkx to store the RNA information in a `nx.DiGraph` directed graph object.
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
     'binding_protein': None, 'binding_ion': None, 'binding_small-molecule': None}

To get more information on what each of these fields refer to, please visit :doc:`rnaglib.data`.



Visualization
-------------

To visualize the 2.5D graphs in the format described above, we have implemented a drawing toolbox with several
functions. The easiest way to use it in your application is to call ``rnaglib.drawing.draw(graph, show=True)``.
A functioning installation of Latex is needed for correct plotting of the graphs. If no installation is detected,
the graphs will be plotted using the LaTex reduced features that ships with matplotlib.

.. code-block:: python

    >>> from rnaglib.drawing.rna_draw import rna_draw
    >>> rna_draw(G, show=True)

|Example graph|


Graph Edit Distance (GED)
-------------------------

GED is the gold standard of graph comparisons. We have put our ged implementation as a part of networkx, and offer
in :doc:`rnaglib.ged` the weighting scheme we propose to compare 2.5D graphs. One can call ``rnaglib.ged.ged()`` on two
graphs to compare them. However, due to the exponential complexity of the comparison, the maximum size of the graphs
should be around ten nodes, making it more suited for comparing graphlets or subgraphs.

.. code-block:: python

    >>> from rnaglib.ged.ged_nx import graph_edit_distance
    >>> from rnaglib.utils.graph_io import graph_from_pdbid
    >>> G = graph_from_pdbid("4nlf")
    >>> graph_edit_distance(G, G)
    ... 0.0

.. |Example graph| image:: https://jwgitlab.cs.mcgill.ca/cgoliver/rnaglib/-/raw/main/images/Fig1.png
