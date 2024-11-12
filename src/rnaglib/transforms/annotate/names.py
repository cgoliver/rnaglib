from rnaglib.transforms import Transform


class PDBIDNameTransform(Transform):
    """Assign the RNA name using its PDBID"""

    def forward(self, data: dict) -> dict:
        g = data["rna"]
        g.name = g.graph["pdbid"][0].lower()
        return data

    pass


'''
class ChainNameTransform(Transform):
    """Set the rna.name field using the pdbid and chain ID.
    Use when the given RNA consists of a single chain."""

    def forward(self, rna_dict: dict) -> dict:
        g = rna_dict["rna"]
        nid, ndata = next(iter(g.nodes(data=True)))
        g.name = g.graph["pdbid"][0].lower() + "_" + ndata["chain_name"] # .upper()
        return rna_dict

    pass
'''


class ChainNameTransform(Transform):
    """Set the rna.name field using the pdbid and chain ID.
    Use when the given RNA consists of a single chain."""

    def forward(self, rna_dict: dict) -> dict:
        g = rna_dict["rna"]
        # Access graph attributes directly
        pdbid = g.graph["pdbid"][0].lower()

        # Get the first node's chain name directly from the node data
        first_node_id = list(g.nodes)[0]  # Get first node ID
        chain_name = g.nodes[first_node_id]["chain_name"]  # .upper()

        # Construct name directly
        g.name = f"{pdbid}_{chain_name}"

        return rna_dict
