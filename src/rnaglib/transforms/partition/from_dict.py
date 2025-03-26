from typing import Iterator

from rnaglib.transforms import PartitionTransform

class PartitionFromDict(PartitionTransform):
    """Partitions an RNA according to a partition defined in a dictionary.

    :param dict partition_dict: dictionary of the form {RNA_name:[[residue_name_1,...,residue_name_6],...,[residue_name_i,...,residue_name_N]]} where we want the RNA to be broken down into several sub-RNAs among which one sub-RNA containing residue_name_1,...,residue_name_6, another one with residue_name_i,...,residue_name_N etc.
    """

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

