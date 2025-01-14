from typing import Iterator
import networkx as nx

from rnaglib.transforms import Transform, PartitionTransform


class ConnectedComponentPartition(PartitionTransform):
    """Split up an RNA by connected components. Yields all nodes belonging to the same connected 
    component at a time.
    """

    def forward(self, rna_dict: dict) -> Iterator[dict]:
        g = rna_dict["rna"]
        connected_components = nx.weakly_connected_components(g)
        for i, c in enumerate(connected_components):
            connected_component_subgraph = g.subgraph(c).copy()
            try:
                connected_component_subgraph.name = g.name + "_" + str(i)
            except:
                connected_component_subgraph.name = g.graph["pdbid"].lower() + "_" + str(i)
            yield {"rna": connected_component_subgraph}