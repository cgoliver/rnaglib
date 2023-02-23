FRAMEWORKS = ['dgl', 'torch', 'pyg', 'nx']

class Representation:
    """ Callable object that accepts a raw RNA networkx object
    and returns a representation of it (e.g. graph, voxel, point cloud)
    along with necessary nucleotide / base pair features """
    def __init__(self,
                 rna_features=None,
                 nt_features=None,
                 bp_features=None,
                 framework='dgl'
                 ):

        self.check_framework(self.framework)
        self.framework = framework

        self.rna_features = rna_features
        self.nt_features = nt_features
        self.bp_features = bp_features
        pass

    def __call__(self, rna_dict):
        """ This function is applied to each RNA in the dataset and updates
        `rna_dict`"""
        raise NotImplementedError

    def check_framework(self, framework):
        assert framework in self.frameworks, f"Framework {framework} not supported for this representation. Choose one of {self.frameworks}." 
