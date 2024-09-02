How to add custom attributes to RNAs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often you will have information on hand about a particular set of RNAs which you would like to integrate into the dataset for analysis or for use by ML models. This can be the result of some experimental assay, external database lookup, embeddings from a pretrained model, etc.

This tutorial will cover two ways you can use to add data to the RNAs in rnaglib: during dataset construction (pre-transform) and during runtime (transform). In both cases, we will make use of the :class:`~rnaglib.transforms.Transform` class. 


Adding new annotations
--------------------------------------------

The first step is to add a annotation to the RNAs such that the necessary information is present in the raw networkx graph representing each RNA by creating a new `Transform` class. 

.. code-block:: python

    from rnaglib.transforms import Transform
    from rnaglib.utils import OneHotEncoder

    class MyTransform(Transform):
        name = 'MY_FEAT'
        encoder = OneHotEncoder({'A': 0, 'B': 1, 'C': 2})
        def forward(data: dict):
            g = data['rna']
            feat = {n: random.choice(['A', 'B', 'C']) for n in g.nodes()}
            nx.set_node_attributes(g, feat, self.name)


.. hint::

    We optionally define the ``name`` and ``encoder`` fields which can be used by :class:`~rnaglib.data_loading.features.FeatureEncoder` objects to cast raw annotations into tensor representations for deep learning.


Now we pass ``MyTransform`` to the dataset construction in the ``pre_transform`` field. During dataset construction (i.e. this happens only once), the transform is applied to each RNA item. In this case, that ensures that each RNA graph's nodes will have tdata in the ``'MY_FEAT'`` field. 

.. code-block:: python

   from rnaglib.data_loading import RNADataset
   t = MyTransform()
   dataset = RNADataset.from_database(debug=True
                                      pre_transform=t,
                                       )


Now you can access the node attribute you defined over all the RNAs.


.. code-block:: python

    nx.get_node_attributes(dataset[0]['rna'], 'MY_FEAT')


Using existing annotations as features for ML tasks
---------------------------------------------------------

The :class:`~rnaglib.data_loading.features.FeaturesComputer` class allows us to cast annotations present in the RNAs, such as any of the available ones in :doc:`the annotation reference <rna_ref>`, or those added by the user as above into tensors for machine learning.

.. code-block:: python

    from rnaglib.data_loading import FeaturesComputer
    from rnaglib.representations import GraphRepresentation


    ft = FeaturesComputer(transforms=t)
    rep = GraphRepresentation(framework='pyg')

    dataset = RNADataset.from_database(debug=True,
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
