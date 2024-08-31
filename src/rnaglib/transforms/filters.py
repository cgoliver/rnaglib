from typing import Iterator

import networkx as nx
from rnaglib.transforms import Transform

class RNAFilter(Transform):
    def __call__(self, data: nx.Graph) -> bool:
        raise NotImplementedError

class SubstructureFilter(Transform):
    """ Used for chunking RNAs. The ``__call__()`` accepts a whole RNA and
    produces an iterator of nx.Graph
    """
    def __call__(self, data: nx.Graph) -> Iterator[nx.Graph]:
        raise NotImplementedError
    pass


class ChainSplitter(SubstructureFilter):
    def __call__(self, data: nx.Graph) -> Iterator[nx.Graph]:
        pass
    pass
