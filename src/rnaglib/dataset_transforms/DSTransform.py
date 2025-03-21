from rnaglib.dataset.rna_dataset import RNADataset


class DSTransform:
    """
    Transforms is just a base class that performs a processing of a whole RNADataset

    Example
    --------

    Transforms are callable objects that modify a dataset passed to it::

        >>> from rnaglib.dataset_transforms import DSTransform
        >>> t = DSTransform()
        >>> dataset = RNADataset(debug=True)
        >>> t(dataset)

    """

    def __init__(self):
        pass

    def __call__(self, dataset: RNADataset) -> RNADataset:
        return dataset

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

