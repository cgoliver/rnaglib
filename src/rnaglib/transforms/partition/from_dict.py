from typing import Iterator

from rnaglib.transforms import PartitionTransform

class PartitionFromDict(PartitionTransform):

    def __init__(
            self, 
            partition_dict,
            **kwargs
            ):
        
        self.partition_dict = partition_dict
        super().__init__(**kwargs)

    def forward(self, rna_dict: dict) -> Iterator[dict]:
        g = rna_dict["rna"]
        subgraph_idx = 0
        for current_subgraph_nodes in self.partition_dict[g.graph["name"]]:
            subgraph = g.subgraph(current_subgraph_nodes).copy()
            if len(subgraph.nodes())>0:
                subgraph.name += "_" + str(subgraph_idx)
                yield {"rna": subgraph}
                subgraph_idx += 1

