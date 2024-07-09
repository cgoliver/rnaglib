import torch
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout

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
    def __init__(self, num_node_features, num_classes, num_unique_edge_attrs):
        super().__init__()
        self.conv1 = RGCNConv(num_node_features, 16, num_unique_edge_attrs)
        self.bn1 = BatchNorm1d(16)  
        self.dropout1 = Dropout(0.1) 
        self.conv2 = RGCNConv(16, num_classes, num_unique_edge_attrs)
        self.bn2 = BatchNorm1d(num_classes)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        x = F.relu(x) 
    
        return F.log_softmax(x, dim=1)