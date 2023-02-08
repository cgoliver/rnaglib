Handling 2.5D graphs
~~~~~~~~~~

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
