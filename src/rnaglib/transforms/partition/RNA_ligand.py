import os
from typing import Iterator
import pandas as pd

from rnaglib.transforms import PartitionTransform
class NTSubgraphTransform(PartitionTransform):
    """Reject nucleotides from a dataset based on some conditions.
    The ``forward()`` method returns True/False for the given RNA and
    the ``__call__()`` method returns the RNAs which pass the ``forward()`` filter.
    """

    def forward(self, rna_dict: dict) -> Iterator[dict]:
        g = rna_dict["rna"]
        
        nodes_to_keep = []
        for node, ndata in g.nodes(data=True):
            if self.node_filter(node, ndata): 
                nodes_to_keep.append(node)

        yield {"rna": g.subgraph(nodes_to_keep).copy()}
    
    def node_filter(self, node, ndata) -> bool:
        raise NotImplementedError
    pass