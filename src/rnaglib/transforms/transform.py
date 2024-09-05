from typing import List, Union, Any

import networkx as nx
from rnaglib.data_loading import RNADataset

class Transform:
    """ Transforms modify and add information to an RNA graph via
    the ``networkx.Graph`` data structure.
    Receives an RNA graph and returns an RNA graph.
    This can be applied at dataset construction time, or a retrieval.
    Implementation inspired by torch-geometric Transforms library.


    Example
    --------

    Transforms are callable objects that modify an RNA graph or dataset passed to it::

        >>> from rnaglib.transforms import Transform
        >>> t = Transform()
        >>> dataset = RNADataset(dummy=True)
        >>> t(rna[0])

    """
    def __call__(self, data: Any) -> Any:
        if isinstance(data, (list, RNADataset)):
            for d in data:
                yield self.forward(d)
        else:
            return self.forward(data)

    def forward(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class FilterTransform(Transform):
    """ Reject items from a dataset based on some conditions """
    def __call__(self, data: Any) -> Iter[Any]:
        if isinstance(data, (list, RNADataset)):
            return filter(data, self.forward(data))
        else:
            return self.forward(data)
        pass
    pass

class PartitionTransform(Transform):
    """ Break up a whole RNAs into substructures.
    Returns a new flat iterator over RNA data items.
    For example, splitting a list of multi-chain RNAs into a
    flat list of single-chain RNAs.
    """
    def __call__(self, data: Any) -> Iter[Any]
        if isinstance(data, (list, RNADataset)):
            for rna in data:
                yield from self.forward(rna)
            pass
        else:
            yield from self.forward(data)
        pass
    pass

class Compose(Transform):
    """ Combine multiple transforms into one, applying
    each individual transform on each item consecutively. """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
        pass

    def forward(self, data: Any):
        for tr in self.transforms:
            data = tr(data)
        return data

    def __repr__(self) -> str:
        """ From PyG"""
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))


class ComposeFilters:
    """Composes several filters together.

    :param filters: List of filter transforms to compose.
    """
    def __init__(self, filters: List[FilterTransform]):
        self.filters = filters

    def __call__(self, data: dict) -> bool:
        for filter_fn in self.filters:
            if isinstance(data, (list, RNADataset)):
                if not all filter_fn(data):
                    return False
            elif not filter_fn(data):
                return False
        return True

    def __repr__(self) -> str:
        args = [f'  {filter_fn}' for filter_fn in self.filters]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))

class ComposePartitions:
    def __init__(self, partitions: List[PartitionTransform]):
        pass
    pass
