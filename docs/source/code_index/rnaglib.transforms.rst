``rnaglib.transforms``
=========================

Transforms are objects which modify RNA dictionaries in various ways. You can apply a transform
to an individual RNA or to a collection (filters can only be applied to collections).

In this example, we add a field ``'rfam'`` with the Rfam ID of an RNA.::

    >>> from rnaglib.transforms import RfamTransform
    >>> from rnaglib.dataset import RNADataset
    >>> dataset = RNADataset(debug=True)
    >>> t = RfamTransform()
    >>> dataset = t(dataset)
    >>> dataset[2]['rna'].graph['rfam']
    'RF00005'


.. note::

    You can often speed up a transform by passing ``parallel=True`` to the transform constructor to apply the transform in parallel.


.. automodule:: rnaglib.transforms

Generic transforms
--------------------

This is the general formulation of the transform, from which specific Transforms described below are derived.

.. autosummary::
    :toctree: generated/

    Transform


Annotation Transforms
----------------------

These transforms update the information stored in an RNA dictionary.

.. autosummary::
    :toctree: generated/

    AnnotationTransform
    RfamTransform
    RNAFMTransform
    PDBIDNameTransform
    ChainNameTransform
    SecondaryStructureTransform
    SmallMoleculeBindingTransform
    RBPTransform
    AnnotatorFromDict
    DummyAnnotator
    CifMetadata
    BindingSiteAnnotator


Filters
----------

These transforms filter out RNAs from a collection of RNAs based on various 
criteria.

.. autosummary::
    :toctree: generated/

    FilterTransform
    SizeFilter
    RNAAttributeFilter
    ResidueAttributeFilter
    ResidueNameFilter
    RibosomalFilter
    NameFilter
    ChainFilter
    ResolutionFilter


Partitions
-------------

These transforms take an RNA and return an iterator of RNAs. Useful
for splitting the RNA into substructures (e.g. by chain ID, binding pockets, etc.)

.. autosummary::
    :toctree: generated/

    PartitionTransform
    ChainSplitTransform
    ConnectedComponentPartition
    PartitionFromDict


Representations
---------------------

These transforms convert a raw RNA into a geometric representation such as graph, voxel and point cloud.

.. autosummary::
    :toctree: generated/

    Representation 
    SequenceRepresentation
    GraphRepresentation
    PointCloudRepresentation
    VoxelRepresentation 
    RingRepresentation

Featurizers
-------------

These transforms take an annotation in the RNA and cast it into a feature vector.


.. autosummary::
    :toctree: generated/

    FeaturesComputer
