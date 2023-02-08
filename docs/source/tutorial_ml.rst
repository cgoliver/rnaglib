Machine Learning Tutorial
~~~~~~~~~~

First usage of the data
--------------------------------

Following datasets like OGB, we provide ready-to-use data sets.
The most common task is node classification.
The user is asked for input node features and node targets.
As an example, we use nucleotide identity ('nt_code') as input and the binding of an ion ('binding_ion') as output.



Then you have one-liner access to a Pytorch dataset and to three loaders that
return DGL batched graphs.
Running the following code

::

    from rnaglib.data_loading.graphloader import GraphDataset, get_loader

    node_features = ['nt_code']
    node_target = ['binding_ion']
    dataset = GraphDataset(node_features=node_features, node_target=node_target)
    train_loader, validation_loader, test_loader = get_loader(dataset=dataset, batch_size=3, num_workers=2)
    for batch in enumerate(train_loader):
        print(batch)

will yield a dictionnary with two keys, "graph" associated to a batched DGL graph and "num_nodes",
a list of the respective number of nodes of each individual graph.

::

    {'graphs': Graph(num_nodes=131, num_edges=356,
        ndata_schemes={'features': Scheme(shape=(4,), dtype=torch.float32), 'target': Scheme(shape=(1,), dtype=torch.float32)}
        edata_schemes={'edge_type': Scheme(shape=(), dtype=torch.int64)}),
     'num_nodes': [44, 44, 43]}
    ...

More modalities of the data
---------------------------

More input representations
==========================

We believe that 2.5D graphs are a good representation for learning on the structure of RNA (based on our paper `RNAMigos <https://github.com/cgoliver/RNAmigos>`__ ).
However, point clouds and voxel based representations are also very common for 3D objects.
RNAGlib handles these representations with no extra burden, you just have to specify it using the 'return_type' argument.
You can ask for one or several representations as is illustrated here :

::

    node_features = ['nt_code']
    node_target = ['binding_ion']
    dataset = GraphDataset(node_features=node_features, node_target=node_target, return_type=['voxel', 'point_cloud'])
    train_loader, validation_loader, test_loader = get_loader(dataset=dataset, batch_size=3, num_workers=0, hstack=False)
    for batch in train_loader:
        for k, v in sorted(batch.items()):
            print((k, [value.shape if isinstance(value, torch.Tensor) else value for value in v]))

Similarly as above, you get a dictionnary with your return type as keys and list of representations for each RNA in the batch.
These lists cannot easily be batched for voxels as the 3D grids might have different dimensions.
However, their point clouds representations can be stacked into one large tensor with 'hstack' argument.
The output of the previous snippet looks like this :

::

    ('node_coords', [torch.Size([92, 3]), torch.Size([77, 3]), torch.Size([76, 3])])
    ('node_feats', [torch.Size([92, 4]), torch.Size([77, 4]), torch.Size([76, 4])])
    ('node_targets', [torch.Size([92, 1]), torch.Size([77, 1]), torch.Size([76, 1])])
    ('num_nodes', [92, 77, 76])
    ('voxel_feats', [torch.Size([4, 4, 4, 5]), torch.Size([4, 6, 6, 7]), torch.Size([4, 6, 6, 6])])
    ('voxel_target', [torch.Size([1, 4, 4, 5]), torch.Size([1, 6, 6, 7]), torch.Size([1, 6, 6, 6])])

Additional inputs and outputs
=============================
Adding more input features to the graphs is straightforward, as you simply have to specify more items in the features list.
A full description of the input features that can be used is available in :doc:`rnaglib.data`.
Similarly, you can seamlessly switch to a multi-task setting by adding more targets. However, doing this affects the splitting procedure.
A side effect could be a slight deviation in the train/validation/test fractions.
The tasks currently implemented are in the set : {'node_binding_small-molecule', 'node_binding_protein', 'node_binding_ion', "node_is_modified"}.
An example of a variation is provided below, the rest of the code is unaffected.

::

    node_features = ['nt_code', "alpha", "C5prime_xyz", "is_modified"]
    node_target = ['binding_ion', 'binding_protein']


Unsupervised pre-training
--------------------------------
Due to a relatively scarse data, we have found useful to pretrain our networks.
The semi-supervised setting was found to work well, where node embeddings are asked to approximate a similarity function over subgraphs.
More precisely, given two subgraphs g1 and g2, a similarity function K, and a neural embedding function f, we want to approximate K(sg1,sg2) ~ <f(sg1), f(sg2)> .
This was described more precisely in `VeRNAl <https://github.com/cgoliver/vernal>`__ .

The datasets and dataloaders natively support the computation of many comparison functions, factored in the SimFunctionNode object.
We also offer the possibility to compute this comparison on a fixed number of sampled nodes from the batch, using the max_size_kernel argument.
The loader will then return an additional field in the batch, with a 'K' key that represents the values of the similarity function over subgraphs.

::

    from rnaglib.kernels import node_sim
    node_sim_func = node_sim.SimFunctionNode(method='R_graphlets', depth=2)
    unsupervised_dataset = graphloader.GraphDataset(node_simfunc=node_sim_func, node_features=node_features)
    train_loader, _, _ = graphloader.get_loader(dataset=unsupervised_dataset, num_workers=4, max_size_kernel=100)

The coordinated use of these functionalities is illustrated in the :doc:`rnaglib.examples`: section.