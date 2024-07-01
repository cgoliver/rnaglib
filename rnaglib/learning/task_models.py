import torch
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn.functional as F

class RGCN_graph(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_unique_edge_attrs):
        super().__init__()
        self.conv1 = RGCNConv(num_node_features, 16, num_unique_edge_attrs)
        self.conv2 = RGCNConv(16, 32, num_unique_edge_attrs)
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Graph-level pooling
        x = self.fc(x)
        return x
    


class RGCN_node(torch.nn.Module):
    def __init__():
        pass

    def forward(self, data):
        return None