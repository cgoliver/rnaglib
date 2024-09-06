from rnaglib.transforms import Transform

class PDBIDNameTransform(Transform):
    """ Assign the RNA name using its PDBID"""

    def forward(self, data: dict) -> dict:
        g = data['rna']
        g.name = g.graph['pdbid'][0].lower()
        return data
    pass
