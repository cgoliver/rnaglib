from typing import Iterator

from rnaglib.transforms import PartitionTransform


class ChainSplitTransform(PartitionTransform):
    """Split up an RNA by chain. Yields all nodes belonging to the same chain
    one chain at a time.
    """

    def forward(self, rna_dict: dict) -> Iterator[dict]:
        rna_dict_copy = rna_dict.copy()
        g = rna_dict["rna"]
        chain_sort_nodes = sorted(list(g.nodes(data=True)), key=lambda ndata: ndata[0].split(".")[1])
        current_chain_name = chain_sort_nodes[0][0].split(".")[1]  # .upper()
        current_chain_nodes = []
        for node, ndata in chain_sort_nodes:
            if node.split(".")[1] == current_chain_name:  # .upper()
                current_chain_nodes.append(node)
            else:
                return_dict = rna_dict_copy.copy()
                return_dict["rna"] = g.subgraph(current_chain_nodes).copy()
                yield return_dict
                current_chain_nodes = [node]
                current_chain_name = node.split(".")[1]  # .upper()

        return_dict = rna_dict_copy.copy()
        return_dict["rna"] = g.subgraph(current_chain_nodes).copy()
        yield return_dict
