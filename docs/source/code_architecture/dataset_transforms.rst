

TODO
A set of RNA objects can be grouped into a RNADataset object, which inherits the Pytorch Dataset object.
These RNAs can either live in the memory, or be found as files.
Datasets can be loaded and saved, subset and looped. They can also hold distance matrices among the different points it holds.
A more detailed description is provided :doc:`here<../code_architecture/dataset>`.


.. raw:: html
    :file: ../images/fig_dataset_transform.svg


The ``Transforms`` API handles any operations that modify RNA dictionaries.

Reminder, an RNA dictionary is the item provided by an ``RNADataset()[i]`` and looks like::

    >>> from rnaglib.data_loading import RNADataset
    >>> dataset = RNADataset(debug=True)
    >>> rna = dataset[3]
    {'rna': <nx.DiGraph...>, ..., }


.. note::
   Transforms can usually be applied in parallel for faster computing by passing `parallel=True` to the constructor.


Transforms come in several flavors depending on the kind of manipulation they apply to the provided data:

* **Represent**: compute tensor-based representations of RNAs (e.g. convert to voxel grid)


Annotation Transforms: add/remove data from RNAs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annotation transforms update the attributes of an RNA, usually by adding a new key/value pair to node/edge/graph-level annotations. This is useful when the annotations provided by default are not enough.
