from typing import Iterator

import networkx as nx
from rnaglib.transforms import FilterTransform

""" Filters return a boolean after receiving an RNA.
This can be used to exclude RNAs from a datataset based on some
desired conditione.
"""

class SizeFilter(FilterTransform):
    """ Reject RNAs that are not in the given size bounds.

    :param min_size: smallest allowed number of residues
    :param max_size: largest allowed number of residues. Default -1 which means no upper bound.
    """
    def __init__(self, min_size:int = 0, max_size: int = -1, **kwargs):
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(**kwargs)

    def forward(self, rna_dict: dict) -> bool:
        n = len(rna_dict['rna'].nodes())
        if self.max_size == -1:
            return n > self.min_size
        else:
            return n > self.min_size and n < self.max_size

class RNAAttributeFilter(FilterTransform):
    """ Reject RNAs that lack a certain annotation at the whole RNA level."""
    def __init__(self, attribute: str, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)
        pass

    def forward(self, data: dict):
        try:
            annot = data['rna'].graph[self.attribute]
        except KeyError:
            return False
        else:
            if annot is None:
                return False
        return True
    pass
