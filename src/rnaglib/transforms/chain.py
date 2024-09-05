from typing import Iterator

from rnaglib.transforms import Transform

class ChainSplitTransform(Transform):
    """ Split up an RNA by chain. Yields all nodes belonging to the same chain
    one chain at a time.
    """
    def __init__(self):
        pass
    
    def forward(self, rna_dict: dict) -> Iterator[dict]:
        g = rna_dict['rna']
        chain_sort_nodes = sorted(list(g.nodes(data=True)), key=lambda ndata:ndata[1]['chain_name'])
        current_chain_name = chain_sort_nodes[0][1]['chain_name']
        current_chain_nodes = []
        for node, ndata in chain_sort_nodes:
            if ndata['chain_name'] == current_chain_name:
                current_chain_nodes.append(node)
            else:
                yield {'rna': g.subgraph(current_chain_nodes).copy()}
                current_chain_nodes = [node]
                current_chain_name = ndata['chain_name']
        yield {'rna': g.subgraph(current_chain_nodes).copy()}

class ChainNameTransform(Transform):
    """ Set the rna.name field using the pdbid and chain ID.
    Use when the given RNA consists of a single chain. """
    def forward(self, rna_dict: dict) -> dict:
        g = rna_dict['rna']
        nid, ndata = next(iter(g.nodes(data=True)))
        g.name = g.graph['pdbid'][0].lower() + "-" + ndata['chain_name']
        return rna_dict
    pass
