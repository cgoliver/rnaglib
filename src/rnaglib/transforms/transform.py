from typing import List, Union, Any

import networkx as nx

class Transform:
    """ Transforms modify and add information to an RNA graph via
    the ``networkx.Graph`` data structure.
    Receives an RNA graph and returns an RNA graph.
    This can be applied at dataset construction time, or a retrieval.


    Example
    --------

    Transforms are callable objects that modify an RNA graph or dataset passed to it::

        >>> from rnaglib.transforms import Transform
        >>> t = Transform()
        >>> dataset = RNADataset(dummy=True)
        >>> t(rna[0])

    """
    def __call__(self, data: Any) -> Any:
        return self.forward(data)

    def forward(self, data: Any) -> Any:
        raise NotImplementedError
