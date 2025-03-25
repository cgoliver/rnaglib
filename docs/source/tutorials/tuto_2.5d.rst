Working with RNA 3D graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we know :doc:`what is an RNA 2.5D graph<../data_reference/what_is>` we can inspect the graph using `rnaglib`.

You can obtain the data in several ways.

Loading from a rnaglib database
---------------------------------

The library ships with some pre-built datasets which you can download with the following command line::

    >>> from rnaglib.dataset import RNADataset
    >>> dataset = RNADataset()

This will download the default data distribution to `~/.rnaglib`

.. code-block:: python

   >>> rna = dataset[0]


We will get a dictionary with a three keys (``'rna'``, ``'graph_path'``,
``'cif_path'``) holding a networkx
graph, path to the graph data, and path to the original mmCIF structure. This graph stores the raw information we extracted from the full 3D
structure. As you will learn in the Tasks section, additional keys will hold other useful information about the RNA
such as tensor representations.

Annotating a local structure 
------------------------------

If you have PDB or mmCIF on your device and you would like to annotate it and
work with it using rnaglib you need this one-liner::

    >>> from rnaglib.prepare_data import fr3d_to_graph
    >>> rna = fr3d_to_graph("path/to/file.cif")


Fetching from a pdbid
-------------------------

Finally, we can fetch an RNA from its pdbid by querying RCSB DataBank, running
the annotation and generating the RNA object.::

    >>> from rnaglib.dataset import rna_from_pdbid
    >>> rna = rna_from_pdbid("1fmn")



Overview of the RNA Graphs
-----------------------------

We use networkx to store the RNA information in a `nx.DiGraph` directed graph object.
Once the graphs are downloaded, they can be fetched directly using their PDBID.

Since nodes represent nucleotides, the node data dictionary will include features such as nucleotide type,
position, 3D coordinates, etc...
Nodes are assigned an ID in the form ``<pdbid.chain.position>``.
Using node IDs we can access node and edge attributes as dictionary keys.

.. code-block:: python

   >>> from rnaglib.dataset import rna_from_pdbid
   >>> rna = rna_from_pdbid("1fmn")
   >>> rna['rna'].nodes['1fmn.A.2']
   {'nt': 'G',
    'nt_full': 'G',
    'chain_id': 'A',
    'is_modified': False,
    'xyz_p': [-11.413, 5.64, 10.08],
    ...
    }

The RNA 2.5D graph contains a rich set of annotations.
For a complete list of these annotations see :doc:`this page<../data_reference/rna_ref>`.

Edges are formed between residues when they form base pairs or backbone
connections.

Each edge stores the type of interaction in the ``LW`` key.::

    >>> rna['rna']['1a9n.R.22']['1a9n.R.0']
    {'LW': 'cWW'}


Visualization
-------------

To visualize the 2.5D graphs in the format described above, we have implemented a drawing toolbox with several
functions. The easiest way to use it in your application is to call ``rnaglib.drawing.draw(graph, show=True)``.
A functioning installation of Latex is needed for correct plotting of the graphs. If no installation is detected,
the graphs will be plotted using the LaTex reduced features that ships with matplotlib.

.. code-block:: python

    >>> from rnaglib.drawing import rna_draw
    >>> rna_draw(G, show=True, layout="spring")


.. image:: ../images/g.png

