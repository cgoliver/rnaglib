import torch

from rnaglib.utils import build_node_feature_parser

FRAMEWORKS = ['dgl', 'torch', 'pyg', 'nx']

class Representation:
    """ Callable object that accepts a raw RNA networkx object
    and returns a representation of it (e.g. graph, voxel, point cloud)
    along with necessary nucleotide / base pair features """
    def __init__(self,
                 framework='nx',
                 frameworks=['nx'],
                ):

        self.framework = framework
        self.frameworks = frameworks
        self.check_framework(self.framework)
        
    def __call__(self, rna_dict, features_dict):
        """ This function is applied to each RNA in the dataset and updates
        `rna_dict`"""
        return self.call(rna_dict, features_dict)

    @property
    def name(self):
        raise NotImplementedError

    def call(self, rna_dict):
        raise NotImplementedError

    def check_framework(self, framework):
        assert framework in self.frameworks, f"Framework {framework} not supported for this representation. Choose one of {self.frameworks}."
