Manipulating RNAs: Transforms API
#################################

The ``Transforms`` API handles any operations that modify RNA dictionaries.

Reminder, an RNA dictionary is the item provided by an ``RNADataset()[i]`` and looks like::

    >>> from rnaglib.dataset import RNADataset
    >>> dataset = RNADataset(debug=True)
    >>> rna = dataset[10]
    {'rna': <nx.DiGraph...>, 'graph_path': .., 'cif_path': ... }


Transforms are ``Callable`` objects. Let's see by importing a transform that does nothing.::

    >>> from rnaglib.transforms import IdentityTransform
    >>> t = IdentityTransform()
    >>> new_rna = t(rna)

To customize the behaviour of the transform you can usually pass arguments to the object constructor. Looking inside a transform all you have is::

    class Transform:
        def __init__(self):
            # any setup for the transform

        def forward(self, data: dict):
            # apply operation to the RNA dictionary
            pass

Transforms come in several flavors depending on the kind of manipulation they apply to the provided data:

* **Filter**: accept or reject an RNA based on some criteria (e.g. remove RNAs that are too large)

* **Partition**: generate a collection of substructure from a whole RNA (e.g. break up an RNA into individual chains)

* **Annotation**: adds or removes annotations from the RNA (e.g. query a database and store results in the RNA)

  * **Featurize**: A special kind of annotation that convert some RNA features into tensors for learning.

* **Represent**: compute tensor-based representations of RNAs (e.g. convert to voxel grid)

.. raw:: html
 :file: ../images/fig_rna_transform.svg


Applying transforms to RNAs:
****************************

Filter Transforms: narrow down datasets
=======================================

Filters return the boolean result of a given test criterion. For example, if you want to only keep RNAs that have a certain maximum size.::

    >>> from rnaglib.transforms import SizeFilter
    >>> t = SizeFilter(max_size=200)
    >>> t(rna)
    True

To implement a filtering transform, the ``forward()`` method accepts an RNA dictionary and returns True or False.


Partition Transforms: focus on substructures
============================================

If you want to only keep certain substructures of an RNA, such as extracting only binding sites, use the partition transforms family.
For example, to split an RNA into individual chains, you can run: ::

    >>> from rnaglib.transforms import ChainSplitTransform
    >>> t = ChainSplitTransform()
    >>> list(t(rna))
    [ {'rna': <nx.DiGraph...>, ..., },  {'rna': <nx.DiGraph...>, ..., }]


To implement a partition transform, the ``forward()`` method defines a **generator** which accepts a single RNA dictionary and yields substructures from the given RNA.


Annotation Transforms: add/remove data from RNAs
================================================

Annotation transforms update the attributes of an RNA, usually by adding a new key/value pair to node/edge/graph-level annotations. This is useful when the annotations provided by default are not enough.

For example, if you want to store the Rfam class of an RNA you can use the ``RfamTransform``::

    >>> from rnaglib.transforms import RfamTransform
    >>> t = RfamTransform()
    >>> t(rna)
    >>> rna['rna'].graph['rfam']
    'RF00005'

For annotation transforms, the ``forward()`` method modifies the given RNA dictionary, optionally returns it if you don't want to work in-place.

Featurize: encode attributes for ML models
------------------------------------------

A special transform is used to convert raw RNA attibutes which have no constraints on their format (e.g. they can be strings representing the Rfam family or nucleotide value) to tensors.
The feature encoder transforms can do this both for input features provided to the model at learning time, or as target features which are the variable the model is trying to predict.::

    >>> from rnaglib.transforms import FeaturesComputer
    >>> from rnaglib.dataset import RNADataset
    >>> ft = FeaturesComputer(nt_features=['nt_code'], nt_targets=['is_modified'])
    >>> features_dict = ft(rna)
    {'nt_features': {'1vfg.C.2': tensor([0., 0., 0., 1.]), ..., '1vfg.D.75': tensor([0., 0., 1., 0.])},
     'nt_targets': {'1vfg.C.2': tensor([0.]),... }}

The above features computer, when called on an RNA graph returns a dictionary of tensors representing the nucleotide ID and chemical modification status.

Represent Transform: geometric representations of RNAs for learning
===================================================================

For deep learning, the raw RNA data has to be encoded into a mathematical structure known as a **representation** which hold the geometric information (e.g. base pairing graph, voxel grid, point cloud).
This is a special representation as it takes as input the `features_dict` computed above ::

    >>> from rnaglib.transforms import GraphRepresentation 
    >>> graph_rep = GraphRepresentation(framework='pyg')
    >>> graph_rep(rna['rna'], features_dict)
    Data(x=[65, 4], edge_index=[2, 146], edge_attr=[146], y=[65])

Check the documentation for arguments to representations. You will typically pass an ID of the deep learning framework you need for the representation (e.g. ``GraphRepresentation(framework='pyg')`` to use pytorch geometric).


Combining Transforms
====================

Transforms of the same kind can be stitched together to avoid repeated iterations on the same list of RNAs using the ``Compose`` transform.::

    >>> from rnaglib.transforms import RNAFMTransform
    >>> from rnaglib.transforms import Compose
    >>> t = Compose([RfamTransform(), RNAFMTransform()])
    >>> transformed = t(rna)
    >>> node, node_data = list(transformed['rna'].nodes(data=True))[0]
    >>> transformed['rna'].graph['rfam'], node_data['rnafm']
    ('RF00005', [0.25730735,  0.20865716, ...,  -0.32694867])

Filter Transforms have their own compose object to deal with their slightly different behaviours (``ComposeFilters``).
Partitions cannot be composed.

Applying transforms to RNA datasets:
************************************

One shot application
====================

Most transforms can be applied to whole RNADatasets or Iterable of RNAs on one go.

.. note::
   We only support such applications on datasets held in memory.

.. note::
   Transforms can usually be applied in parallel for faster computing by passing `parallel=True` to the constructor.

We now provide examples of applying aforementioned transformed to datasets

Filters : ::

    >>> from rnaglib.transforms import SizeFilter
    >>> t = SizeFilter(max_size=60)
    >>> rnas = list(t(dset))
    >>> len(rnas), len(dset)
    29, 50

The new ``rnas`` list will contain only the RNAs that have fewer than 50 residues.

Partitions : ::

    >>> t = ChainSplit()
    >>> t(dset)

Now instead of the dataset containing a list of RNAs that can each have multiple chains, the new list will contain possibly more entries but each entry only consists of a single chain.

Annotations : ::

    >>> from rnaglib.dataset import RNADataset
    >>> dset = RNADataset(debug=True)
    >>> t = RfamTransform()
    >>> t(dset)
    >>> dset[0]['rna'].graph['rfam']
    'RF0005'


Dataset attributes
==================

The other, better supported way to apply transforms to individual elements of a dataset is to add it in the RNADataset constructor.
As expected, this does not work for FilterTransform and PartitionTransform (that would dynamically affect the dataset length...).
To do so, one can do : ::

    >>> from rnaglib.dataset import RNADataset
    >>> from rnaglib.transforms import PDBIDNameTransform
    >>> dset.transforms.append(PDBIDNameTransform())
    >>> dset[0]['rna'].name
    '1d0t'

Featurizations and Representations have a special role in machine learning: even on a built Task, practitioners might be interested to use them.
For this reason, we have added those Transforms in the RNADataset constructor, as exemplified below: ::

    >>> from rnaglib.transforms import FeaturesComputer, GraphRepresentation
    >>> fc = FeaturesComputer(nt_features=['nt_code'], nt_targets=['is_modified'])
    >>> rep = GraphRepresentation(framework='pyg')
    >>> dset = RNADataset(debug=True, features_computer=fc, representations=rep)
    >>> dset[0]
    {'rna': <networkx.classes.digraph.DiGraph at 0x773be50037d0>,
     'graph_path': ..., 'cif_path': ...,
     'graph': Data(x=[21, 4], edge_index=[2, 58], edge_attr=[58], y=[21])}

The features computer and representation stay exposed and can therefore be modified on the fly. ::

    >>> dset.features_computer.add_feature('alpha')
    >>> dset[0]
    {'rna':..., 'graph_path': ..., 'cif_path': ...,
     'graph': Data(x=[21, 5], edge_index=[2, 58], edge_attr=[58], y=[21])}

Notice how the input variable x in the PyG graph is now 5-dimensional instead of 4.