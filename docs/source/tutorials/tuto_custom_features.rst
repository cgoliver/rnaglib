How to add custom attributes to RNAs 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often you will have information on hand about a particular set of RNAs which you would like to integrate into the dataset for analysis or for use by ML models. This can be the result of some experimental assay, external database lookup, embeddings from a pretrained model, etc.


Adding new annotations
--------------------------------------------

The first step is to add an annotation to the RNAs such that the necessary information is present in the raw networkx graph representing each RNA by creating a new `Transform` class. 

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

   from rnaglib.dataset import RNADataset
   t = MyTransform()
   dataset = RNADataset(debug=True
                        pre_transforms=t,
                        )


Now you can access the node attribute you defined over all the RNAs. If you pass the transform instead to the ``transforms`` argument of the dataset constructor, the transform will be applied every time the dataset item is fetched by ``__getitem__()``.


.. code-block:: python

    nx.get_node_attributes(dataset[0]['rna'], 'MY_FEAT')


Using existing and custom annotations as features for ML tasks
-----------------------------------------------------------------

The :class:`~rnaglib.data_loading.features.FeaturesComputer` class allows us to cast annotations present in the RNAs, such as any of the available ones in :doc:`the annotation reference <rna_ref>`, or those added by the user as above into tensors for machine learning.


To use existing annotations you can simply pass their name to the ``FeaturesComputer`` object and then to the ``RNADataset``. In this example we ask for the ``'nt_code'`` annotation at the nucleotide (``nt``) level which tells us the nucleotide identity for each residue in the RNA.

.. code-block:: python

    from rnaglib.data_loading import FeaturesComputer
    from rnaglib.representations import GraphRepresentation


    ft = FeaturesComputer(nt_features=['nt_code'])
    rep = GraphRepresentation(framework='pyg')

    dataset = RNADataset.from_database(debug=True,
                                       features_computer=ft,
                                       representations=[rep])


Now each data item will contain a `'graph'` key that holds a PyG graph with the 4-dimension feature representing nucleotide identity.

To additionally include a custom feature you simply add the transform you used to create the annotation to the ``FeaturesComputer``. This is where a transform which defines an ``encoder`` and ``name`` attribute comes in handy since the ``FeaturesComputer`` uses these to know how to cast the annotation into a feature vector. Finally you also pass the transform as a dataset argument so that the transform is actually applied to the data.


.. code-block:: python

    from rnaglib.data_loading import FeaturesComputer
    from rnaglib.representations import GraphRepresentation


    t = MyTransform()
    ft = FeaturesComputer(nt_features=['nt_code', 'MY_FEAT'], transforms=t)
    rep = GraphRepresentation(framework='pyg')

    dataset = RNADataset(debug=True,
                         features_computer=ft,
                         pre_transforms=t,
                         representations=[rep])


