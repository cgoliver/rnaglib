import dgl
from dgl.data import DGLDataset

class DGLGraphDataset(DGLDataset):
    def __init__(self, rna_graphs):
        self.rna_graphs = rna_graphs
        self.graphs = [dgl.from_networkx(g) for g in self.rna_graphs]
        super().__init__()

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.rna_graphs)
