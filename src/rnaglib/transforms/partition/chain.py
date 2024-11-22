from typing import Iterator

from rnaglib.transforms import Transform, PartitionTransform


class ChainSplitTransform(PartitionTransform):
    """Split up an RNA by chain. Yields all nodes belonging to the same chain
    one chain at a time.
    """

    def forward(self, rna_dict: dict) -> Iterator[dict]:
        g = rna_dict["rna"]
        chain_sort_nodes = sorted(
            list(g.nodes(data=True)), key=lambda ndata: ndata[1]["chain_name"]
        )
        current_chain_name = chain_sort_nodes[0][1]["chain_name"]  # .upper()
        current_chain_nodes = []
        for node, ndata in chain_sort_nodes:
            if ndata["chain_name"] == current_chain_name:  # .upper()
                current_chain_nodes.append(node)
            else:
                yield {"rna": g.subgraph(current_chain_nodes).copy()}
                current_chain_nodes = [node]
                current_chain_name = ndata["chain_name"]  # .upper()

        yield {"rna": g.subgraph(current_chain_nodes).copy()}
