Manipulating RNAs: Transforms API
------------------------------------------

.. raw:: html

    <object data="../images/fig_task.svg" type="image/svg+xml"></object>


The ``Transforms`` API handles any operations that modify RNA dictionaries.

Reminder, an RNA dictionary is the item provided by an ``RNADataset()[i]`` and looks like::

    >>> from rnaglib.data_loading import RNADataset
    >>> dataset = RNADataset(debug=True)
    >>> rna = dataset[3]
    {'rna': <nx.DiGraph...>, ..., }


Transforms are ``Callable`` objects which operate on individual RNAs or collections of RNAs. Let's see by importing a transform that does nothing.::

    >>> from rnaglib.transforms import Transform
    >>> t = Transform()
    >>> new_rna = t(rna)
    >>> new_rnas = t(dataset)

To customize the behaviour of the transform you can usually pass arguments to the object constructor. Looking inside a transform all you have is::

    class Transform:
        def __init__(self):
            # any setup for the transform

        def forward(self, data: dict):
            # apply operation to the RNA dictionary
            pass



.. note::
   Transforms can usually be applied in parallel for faster computing by passing `parallel=True` to the constructor.


Transforms come in several flavors depending on the kind of manipulation they apply to the provided data:

* **Annotation**: adds or removes annotations from the RNA (e.g. query a database and store results in the RNA)
* **Filter**: accept or reject certain RNAs from a collection based on some criteria (e.g. remove RNAs that are too large)
* **Partition**: generate a collection of substructure from a whole RNA (e.g. break up an RNA into individual chains)
* **Featurize**: convert RNA annotations to tensors for learning.
* **Represent**: compute tensor-based representations of RNAs (e.g. convert to voxel grid)


Annotation Transforms: add/remove data from RNAs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annotation transforms update the attributes of an RNA, usually by adding a new key/value pair to node/edge/graph-level annotations. This is useful when the annotations provided by default are not enough.

For example, if you want to store the Rfam class of an RNA you can use the ``RfamTransform``::

    >>> from rnaglib.transforms import RfamTransform
    >>> from rnaglib.data_loading import RNADataset
    >>> dset = RNADataset(debug=True)
    >>> t = RfamTransform()
    >>> t(dset)
    >>> dset[0]['rna'].graph['rfam']
    'RF0005'

For annotation transforms, the ``forward()`` method modifies the given RNA dictionary, optionally returns it if you don't want to work in-place.

Filter Transforms: narrow down datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Filters reduce a collection of RNAs based on a given test criterion. For example, if you want to only keep RNAs that have a certain maximum size.::

    >>> from rnaglib.transforms import SizeFilter
    >>> t = SizeFilter(max_size=50)
    >>> rnas = t(dset)

The new ``rnas`` list will contain only the RNAs that have fewer than 50 residues.

To implement a filtering transform, the ``forward()`` method accepts an RNA dictionary and returns True or False.


Partition Transforms: focus on substructures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to only keep certain substructures of an RNA. For example by extracting only binding sites, or splitting into individual chians, use the partition transforms family.::

    >>> from rnaglib.transforms import ChainSplit
    >>> from rnaglib.data_loading import RNADataset
    >>> t = ChainSplit()
    >>> dset = RNADataset(debug=True)
    >>> t(dset)

Now instead of the dataset containing a list of RNAs that can each have multiple chains, the nuew list will contain possibly more entries but each entry only consists of a single chain.

To implement a partition transform, the ``forward()`` method defines a **generator** which accepts a single RNA dictionary and yields substructures from the given RNA.

Represent Transform: geometric representations of RNAs for learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For deep learning, the raw RNA data has to be encoded into a mathematical structure known as a **representation** which hold the geometric information (e.g. base pairing graph, voxel grid, point cloud). ::

    >>> from rnaglib.transforms import GraphRepresentation 
    >>> from rnaglib.transforms import PointCloudRepresentation 
    >>> t1 = GraphRepresentation()
    >>> t2 = PointCloudRepresentation()
    >>> dset = RNADataset(debug=True, representations=[t1, t2])
    >>> dset[0]
    {'rna': ..., 'graph': ..., 'point_cloud'...}


You can apply the representation directly to an RNA as with the other transforms. However most of the time you will be passing it to a dataset so that when you load the RNAs they are converted to the necessary representation.

Check the documentation for arguments to representations. You will typically pass an ID of the deep learning framework you need for the representation (e.g. ``GraphRepresentation(framework='pyg')`` to use pytorch geometric).

Featurize: encode attributes for ML models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, a special transform is used to convert raw RNA attibutes which have on constraints on their format (e.g. they can be strings representing the Rfam family or nucleotide value) to tensors. The feature encoder transforms can do this both for input features provided to the model at learning time, or as target features which are the variable the model is trying to predict.::

    >>> from rnaglib.transforms import FeaturesComputer
    >>> from rnaglib.data_loading import RNADataset
    >>> ft = FeaturesComputer(nt_features=['nt_code'], nt_targets=['is_modified'])
    >>> dataset = RNADataset(debug=True)
    >>> features_dict = ft(dataset[0])
    {'nt_features': Tensor(...), 'nt_targets': Tensor(...)}

The above features computer, when called on an RNA graph returns a dictionary of tensors representing the nucleotide ID and chemical modification status.

Most likely you won't use this directly and instead pass the featuers computer to the ``RNADatsaet`` object so that the featuers are served by the loader.::

    >>> RNADataset(features_computer=features_computer)


Additionally, you can load a task and choose which variables you want to feed your model::

    >>> from rnaglib.tasks import ChemicalModification
    >>> ta = ChemicalModification()
    >>> ta.dataset.features_computer.add_feature('alpha')

The features computer has a method to add and remove features so you can go beyond the default features provided by the task.

Combining Transforms
~~~~~~~~~~~~~~~~~~~~~~~

Transforms of the same kind can be stitched together to avoid repeated iterations on the same list of RNAs using the ``Compose`` transform.::

    >>> from rnaglib.transforms import FilterTransform
    >>> from rnaglib.trasforms import RfamTransform
    >>> from rnaglib.transforms import RNAFMTransform
    >>> from rnaglib.data_loading import RNADataet
    >>> dataset = RNADataset(debug=True)
    >>> t = [RfamTransform(), RNAFMTransform()]
    >>> t(dataset)


Each type of transform has its own compose object to deal with the slightly different behaviour. If you are composing filters use the ``ComposeFilters`` or composing partitions use the ``ComposePartitions``.
