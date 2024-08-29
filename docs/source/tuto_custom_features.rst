How to add custom attributes to RNAs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often you will have information on hand about a particular set of RNAs which you would like to integrate into the dataset for analysis or for use by ML models. This can be the result of some experimental assay, or embeddings from a pretrained model.

This tutorial will cover how to add your own annotations to an RNA dataset and how to encode it for an ML task in 2 easy steps.

1. Enter your annotation
-----------------------------

The first step is to add a annotation to the RNAs such that the necessary information is present in the raw networkx graph representing each RNA.

To do this, we write a small helper function which accepts an RNA networkx graph as input and updates its node/edge/graph attributes following any logic you like.


In this case, we will have a trivial example of an annotator which adds a node-level attribute called ``'MY_FEAT'`` whose value will be a random category with values `'A'` or `'B'`. During dataset construction, the annotator function is applied individually to each RNA.


.. code-block:: python

    def my_annotator(g: nx.Graph):
        feat = {n: random.choice(['A', 'B', 'C']) for n in g.nodes()}
        nx.set_node_attributes(g, feat, 'MY_FEAT')


At this point, you could pass the annotator to the dataset constructor and your dataet will contain the information you have provided. 


.. code-block:: python

   from rnaglib.data_loading import RNADataset
   # returns a list of PDBIDs
    pdbids = ['2pwt', '5v3f', '379d',
              '5bjo', '4pqv', '430d',
              '1fmn', '2mis', '4f8u'
              ]


    dataset = RNADataset.from_database(all_rnas_db=self.all_rnas,
                                       annotator=annotator,
                                       )


Now you can access the node attribute you defined over all the RNAs.


.. code-block:: python

    nx.get_node_attributes(dataset[0]['rna'], 'MY_FEAT')


Next we will see how you can use this attribute for ML.


2. Encode the annotation
-------------------------------

Once the raw information is stored in the graphs, we need to tell rnaglib how to convert it to a numerical representation (e.g. as one-hot encoding for categorical data). For this, we provide the :class:`~rnaglib.data_loading.FeaturesComputer` class which processes desired annotations and turns them into a numerical tensors describing all node, edge and graph features.

The :class:`~rnaglib.data_loading.FeaturesComputer` object natively deals with annotations already provided by rnaglib. For it to work on custom annotations, such as in this case, we need to tell it how to deal with our annotation by passing it the appropriate :class:`~rnaglib.utils.feature_maps.Encoder`.
We provide several ``Encoder`` objects in the ``rnaglib.utils`` subpackage which take care of converting data to numerical representations.
Since our example is a categorical variable with 3 possible values, we will want to use the ``OneHotEncoder``.
We pass the encoder to the ``FeaturesComputer`` in a dictionary keyed by the name of the attribute it will be applied to. If you have more than one attribute to encode, you can add it to the same dictionary.

The above snippet remains almost the same, except this time we pass the ``FeaturesComputer`` object to the dataset constructor. 
Along with an encoder, for ML we need a representation of the data so here we pass the `GraphRepresentation()` object and select the PyG framework.

.. code-block:: python

    from rnaglib.utis import OneHotEncoder
    from rnaglib.data_loading import FeaturesComputer
    from rnaglib.representations import GraphRepresentation


    custom_encoder = {'MY_FEAT': OneHotEncoder({'A': 0, 'B': 1, 'C': 2})}
    ft = FeaturesComputer(custom_encoders_features=custom_encoder)
    rep = GraphRepresentation(framework='pyg')

    dataset = RNADataset.from_database(all_rnas_db=all_rnas,
                                       annotator=annotator,
                                       redundancy='all',
                                       features_computer=ft,
                                       representations=[rep])


Now each data item will contain a `'graph'` key that holds a PyG graph with the 3-dimension feature as a node attribute.


.. code-block:: python

    >>> dataset[0]['graph'].x
    tensor([[0., 0., 1.],
            [0., 0., 1.],
            [0., 0., 1.],
            ...
            ])
