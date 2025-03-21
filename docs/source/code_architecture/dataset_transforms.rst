RNADataset Transforms
#####################

``DSTransform`` are transformed that take a whole RNADataset (see :doc:`dataset<code_architecture/dataset>`) as input and return a whole dataset.
They mostly revolve around computing distances between RNAs in the dataset.
Once distances are computed, they can be used to remove redundancy in the dataset, as well to split it in meaningful splits that avoid data leakage.

.. raw:: html
    :file: ../images/fig_dataset_transform.svg

Let's use an example dataset with three points: ::

    >>> from rnaglib.dataset import RNADataset
    >>> rna_names = ['1a9n', '1av6', '1b23']
    >>> dataset = RNADataset(rna_id_subset=rna_names)


Computing distances:
********************


We provide support for computing sequence alignments with CD-hit or structural alignments with USalign or RNAlign.

.. note::
   Those metrics are all cast as distances for simplicity (so 0.9 means low similarity)

The first step for dataset post-processing is to compute distances between points.
The generic class to do so is the ``DistanceComputer``, used as follow: ::

    >>> from rnaglib.dataset_transforms import CDHitComputer
    >>> dataset = CDHitComputer()(dataset)
    >>> dataset.distances
    {'cd_hit': array([[0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.]])}

.. note::
   CD-Hit is used in cluster mode : input sequences are cut into chunks, and a tanimoto score is computed through cluster attributions

One can compute additional distances, resulting in extra keys in the dataset.distance dictionnary: ::

    >>> from rnaglib.dataset_transforms import StructureDistanceComputer
    >>> dataset = StructureDistanceComputername(name="USalign")(dataset)
    >>> dataset.distances
    {'cd_hit': array([[0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.]]),
     'USalign': array([[0.     , 0.86451, 0.75911],
            [0.86451, 0.     , 0.78482],
            [0.75911, 0.78482, 0.     ]])}

.. note::
   Distance computation does not happen in-place.

Removing redundancy and computing splits:
*****************************************

Equiped with those distances, we can easily remove the redundancy of our dataset. ::


    >>> from rnaglib.dataset_transforms import RedundancyRemover
    >>> usalign_rr = RedundancyRemover(distance_name="USalign", threshold=0.8)
    >>> dataset_non_redundant = usalign_rr(dataset)
    >>> len(dataset_non_redundant)
    3

Here, the number remains the same since our data points are quite dissimilar.
We can also split this dataset by using one of our ``Splitter`` class. ::

    >>> from rnaglib.dataset_transforms import ClusterSplitter
    >>> splitter = ClusterSplitter(split_train=0.34, split_valid=0.33, split_test=0.33, distance_name="USalign", balanced=False)
    >>> splitter(dataset_non_redundant)
    ([2], [0], [1])

This returns a list of train, val and test ids. Of course, here we only have one of each.

Moreover, in the ``ClusterSplitter``, we can set balanced to True to provide balanced splits with regards to some labels.
They need to have been pre-computed by a FeaturesComputer beforehands.

Creating a loader:
******************

The final step of manipulating an RNADataset is to make it a DataLoader. To do so, we rely on native Pytorch Dataloader.
We provide a utility ``Collater`` class, that automatically handles collation based on the representations present in the RNAdataset.
We provide an example here, relying on the GraphRepresentation ::

    >>> from torch.utils.data import DataLoader
    >>> from rnaglib.dataset_transforms import Collater
    >>> from rnaglib.transforms import GraphRepresentation
    >>> dataset_non_redundant.add_representation(GraphRepresentation(framework='pyg'))
    >>> collater = Collater(dataset_non_redundant)
    >>> loader = DataLoader(dataset_non_redundant, batch_size=2, collate_fn=collater)
    >>> for batch in loader:
    >>>     print(batch['graph'])
    DataBatch(edge_index=[2, 126], edge_attr=[126], batch=[54], ptr=[3])
    DataBatch(edge_index=[2, 194], edge_attr=[194], batch=[74], ptr=[2])

