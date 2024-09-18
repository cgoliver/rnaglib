from rnaglib.transforms import Transform


class PDBIDNameTransform(Transform):
    """ Assign the RNA name using its PDBID"""

    def forward(self, data: dict) -> dict:
        g = data['rna']
        g.name = g.graph['pdbid'][0].lower()
        return data

    pass


class ChainNameTransform(Transform):
    """ Set the rna.name field using the pdbid and chain ID.
    Use when the given RNA consists of a single chain. """

    def forward(self, rna_dict: dict) -> dict:
        g = rna_dict['rna']
        nid, ndata = next(iter(g.nodes(data=True)))
        g.name = g.graph['pdbid'][0].lower() + "-" + ndata['chain_name']
        return rna_dict

    pass
