from rnaglib.datasets import DGLGraphDataset
from rnaglib.utils import fix_buggy_edges

class GraphDataset:
    def __init__(self, rnas):
        """
        This class converts an RNAGlibDataset to a dataset of graphs.
        Nothing to do there since the base data type is already a graph.

        :param rnas: list of rna data objects from RNAGlibDataset.
        """
        self.graphs = [fix_buggy_edges(graph=graph) for graph in rnas]

    def dgl(self):
        return DGLGraphDataset(self.graphs)

    def pyg(self):
        pass
