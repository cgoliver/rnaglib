from typing import Iterator
import networkx as nx

from rnaglib.transforms import Transform, PartitionTransform


class ConnectedComponentSplitTransform(PartitionTransform):
    """Split up an RNA by connected components. Yields all nodes belonging to the same connected 
    component at a time.
    """

    def forward(self, rna_dict: dict) -> Iterator[dict]:
        g = rna_dict["rna"]
        connected_components = nx.weakly_connected_components(g)
        for c in connected_components:
            yield {"rna": g.subgraph(c).copy()}