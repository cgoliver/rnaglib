from joblib import Parallel, delayed
from typing import List, Union, Any, Iterable, Generator

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
    def __init__(self, parallel: bool = False, num_workers: int = -1):
        self.parallel = parallel
        self.num_workers = num_workers
        pass

    def __call__(self, data: Any) -> Any:
        if isinstance(data, (list, Generator, RNADataset)):
            if self.parallel:
                return Parallel(n_jobs=self.num_workers)(delayed(self.forward)(d) for d in data)
            else:
                return (self.forward(d) for d in data)
        else:
            return self.forward(data)

    def forward(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class FilterTransform(Transform):
    """ Reject items from a dataset based on some conditions.
    The ``forward()`` method returns True/False for the given RNA and
    the ``__call__()`` method returns the RNAs which pass the ``forward()`` filter.
    """
    def __call__(self, data: Any) -> Union[bool, Iterable[Any]]:
        if not isinstance(data, (list, Generator, RNADataset)):
            raise ValueError("Filter transforms only apply to collections of RNAs.")
        if self.parallel:
            keeps = Parallel(n_jobs=self.num_workers)(delayed(self.forward)(d) for d in data)
            return (d for d, keep in zip(data, keeps) if keep)
        else:
            return (d for d in data if self.forward(d))

    def forward(self, data: dict) -> bool:
        """ Returns true/ or false on the given RNA"""
        raise NotImplementedError
    pass

class PartitionTransform(Transform):
    """ Break up a whole RNAs into substructures.
    Returns a new flat iterator over RNA data items.
    For example, splitting a list of multi-chain RNAs into a
    flat list of single-chain RNAs.
    """
    def __call__(self, data: Any) -> Iterable[Any]:
        if isinstance(data, (list, Generator, RNADataset)):
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

    def __init__(self, transforms: List[Transform], **kwargs):
        self.transforms = transforms
        super().__init__(**kwargs)
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
        if not isinstance(data, (list, Generator, RNADataset)):
            raise ValueError("Filter compose only works on collections of RNAs")
        for filter_fn in self.filters:
                data = (d for d in data if filter_fn.forward(d))
        return data

    def __repr__(self) -> str:
        args = [f'  {filter_fn}' for filter_fn in self.filters]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))

class ComposePartitions:
    def __init__(self, partitions: List[PartitionTransform]):
        pass
    pass
