Machine Learning Tutorial
============================


RNAGlib data structures
--------------------------

We have introduced the 2.5D graph format in another tutorial.
RNAGlib provides access to collections of RNAs for machine learning with PyTorch.
It revolves around the usual Dataset and Dataloader objects, along with a Representation object :

* `RNADatasets` objects are an iterable of RNA data, that returns representations of this data
* The `Features_Computer` object can be thought of as Transforms class, selecting relevant node or graph features
and transforming them into tensors ready to be used in deep learning.
* The `Representation` object will return our data in a certain representation (e.g. graphs, voxels, point clouds) as
  well as cast to different data science and ML frameworks (DGL, pytorch-geometric, networkx).
* The `get_loader` function encapsulates automatic data splitting and collating and returns appropriate PyTorch data loaders.


Datasets
~~~~~~~~~~

The `rnaglib.data_loading.RNADataset` object builds and provides access to collections of RNAs.
When using the Dataset class, our standard data distribution should be downloaded automatically.
Alternatively, you can choose to provide your own annotated RNAs by providing a `dataset_path`.

To create a dataset using our hosted data simply instantiate the `RNADataset` object.

.. code-block:: python

   from rnaglib.data_loading import RNADataset

   dataset = RNADataset()


Different datasets can be specified using the following options in the `RNADataset.from_args()` parameters:

* `version='x.x.x'`: which data build to load
* `redundancy`: by default, we only load non-redundant structures `redundancy='nr'` you can also pass `redundancy='all'` to get all RNAs.
* `all_rnas`: a specific list of pdb ids to iterate through

Datasets can be indexed like a list or you can inspect an individual RNA by its PDBID.

.. code-block:: python

    rna_1 = dataset[3]
    pdbid = dataset.available_pdbids[3]
    rna_2 = dataset.get_pdbid(pdbid)

The returned object is a dictionnary with three entries :

* rna : The raw 2.5D graph in the form of a networkx object which you can inspect as shown in :doc:`this tutorial<tuto_2.5d>`.
* rna_name : the name of the PDB being returned
* path : the path to the pdb files

Representations
~~~~~~~~~~~~~~~~~

The next important object for RNAGlib is the `FeaturesComputer`. This object can be thought as a Transforms function
that acts on the raw data to select relevant features and turn them into torch Tensors.
The user can ask for input nucleotide features and nucleotide targets.
As an example, we use nucleotide identity ('nt_code') as input and the binding of an ion ('binding_ion') as output.
We can also ask for additional features after creation of the object, as well as provide it with custom transformations.
These two additions are exemplified below :

.. code-block:: python

    from rnaglib.data_loader import FeaturesComputer
    features_computer = FeaturesComputer(nt_features='nt_code', nt_targets='binding_protein')
    features_computer.add_feature('alpha') # Add alpha angle for illustation purposes
    features_computer.input_dim
    >>> 5

This object will be used internally and exemplified below.

Representations
~~~~~~~~~~~~~~~~~

The next important object for RNAGlib is the `Representation`. Previously, our return only included the raw data.
One can add a `Representation` object with arguments to post-process this raw data into a more usable data format.
The most trivial one is to ask for a `GraphRepresentation`. One can choose either networkx, DGL or PyTorch Geometric as
a return type.

By default, this 2.5D graph only includes the connectivity of the graphs.
The user can ask for input nucleotide features and nucleotide targets.
As an example, we use nucleotide identity ('nt_code') as input and the binding of an ion ('binding_ion') as output.
These two additions are exemplified below :

.. code-block:: python

    from rnaglib.representations import GraphRepresentation

    graph_rep = GraphRepresentation(framework='dgl')
    dataset = RNADataset(features_computer=features_computer, representation=graph_rep)

    print(dataset[0]['graph'])

    >>> {Graph(num_nodes=24, num_edges=58,
            ndata_schemes={'nt_features': Scheme(shape=(5,), dtype=torch.float32),
                           'nt_targets': Scheme(shape=(1,), dtype=torch.float32)}
            edata_schemes={'edge_type': Scheme(shape=(), dtype=torch.int64)})}

We currently support two other data representations : `PointCloudRepresentation` and `VoxelRepresentation`
More generally, `rnaglib.representations.Representation` class holds the logic for converting a dataset to one of the above
representations and users can easily sub-class this to create their own representations.

These classes come with their own set of attributes. Users can use several representations at the same time.

.. code-block:: python

    from rnaglib.representations import PointCloudRepresentation, VoxelRepresentation

    pc_rep = PointCloudRepresentation()
    voxel_rep = VoxelRepresentation(spacing=2)

    dataset.add_representation(voxel_rep)
    dataset.add_representation(pc_rep)
    print(dataset[0].keys())

    >>> dict_keys(['rna_name', 'rna', 'path', 'graph', 'voxel', 'point_cloud'])

As can be seen, we now have additional keys in the returned dictionnary corresponding to the data represented as voxels
or point clouds.
In our case, the RNA has 24 nucleotides and is approximately 12 Angrstroms wide.
Hence, dataset[0]['point_cloud'] is a dictionnary that contains two grids in the PyTorch order :

* ``voxel_feats : torch.Size([5, 6, 5, 6])``
* ``voxel_target : torch.Size([1, 6, 5, 6])``

While dataset[0]['point_cloud'] is a dictionnary that contains one list and three tensors :

* ``point_cloud_coords torch.Size([24, 3])``
* ``point_cloud_feats torch.Size([24, 5])``
* ``point_cloud_targets torch.Size([24, 1])``
* ``point_cloud_nodes ['1a9n.Q.0', '1a9n.Q.1',... '1a9n.Q.9']``

Dataloader
~~~~~~~~~~~~

The missing piece is utilities to efficiently load our dataset for machine learning. The first task is to split our data
in a principled way.
To enhance reproducibility, we offer automatic random splitting procedure that avoid loading useless graphs (for instance
graphs with no positive nodes for node classification) and balance the train/test proportions in the multi-task setting.

The other problematic step is to batch our data automatically, as the batching procedure depends on the representations
that are used. These two functionalities are implemented in a straightforward manner :

.. code-block:: python

    from torch.utils.data import DataLoader
    from rnaglib.data_loading import split_dataset, Collater

    train_set, valid_set, test_set = split_dataset(dataset, split_train=0.7, split_valid=0.85)
    collater = Collater(dataset=dataset)
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=2, num_workers=0, collate_fn=collater.collate)

    for batch in train_loader:
        ...

will yield a dictionnary with the same keys and structure as above, for batches of two graphs.


More advanced functionalities
-------------------------------

Additional inputs and outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding more input features to the graphs is straightforward, as you simply have to specify more items in the features list.
A full description of the input features that can be used is available in :doc:`rnaglib.data`.
Similarly, you can seamlessly switch to a multi-task setting by adding more targets. However, doing this affects the splitting procedure.
A side effect could be a slight deviation in the train/validation/test fractions.
The tasks currently implemented are in the set : {'node_binding_small-molecule', 'node_binding_protein', 'node_binding_ion', "node_is_modified"}.
An example of a variation is provided below, the rest of the code is unaffected.

.. code-block:: python

    nt_features = ['nt_code', "alpha", "C5prime_xyz", "is_modified"]
    nt_targets = ['binding_ion', 'binding_protein']
    features_computer = FeaturesComputer(nt_features=nt_features, nt_targets=nt_targets)


Unsupervised pre-training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to a relatively scarse data, we have found useful to pretrain our networks.
The semi-supervised setting was found to work well, where node embeddings are asked to approximate a similarity function over subgraphs.
More precisely, given two subgraphs g1 and g2, a similarity function K, and a neural embedding function f, we want to approximate K(sg1,sg2) ~ <f(sg1), f(sg2)> .
This was described more precisely in `VeRNAl <https://github.com/cgoliver/vernal>`__ .

The datasets and dataloaders natively support the computation of many comparison functions, factored in the SimFunctionNode object.
We also offer the possibility to compute this comparison on a fixed number of sampled nodes from the batch, using the max_size_kernel argument.
To use this functionality, we packaged into an additional Representation.
The loader will then return an additional field in the batch, with a 'ring' key that represents the values of the similarity function over subgraphs.

.. code-block:: python
   
    from rnaglib.kernels import node_sim
    from rnaglib.representations import RingRepresentation


    node_simfunc = node_sim.SimFunctionNode(method='R_1', depth=2)
    ring_rep = RingRepresentation(node_simfunc=node_simfunc, max_size_kernel=100)
    da.add_representation(ring_rep)
    train_loader, _, _ = graphloader.get_loader(dataset=unsupervised_dataset)

The coordinated use of these functionalities is illustrated in the :doc:`rnaglib.examples`: section.
