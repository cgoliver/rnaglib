import torch
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout

class RGCN_graph(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_unique_edge_attrs, num_layers=2, hidden_channels=16):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        if num_layers > 0:
            self.convs.append(RGCNConv(num_node_features, hidden_channels, num_unique_edge_attrs))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_unique_edge_attrs))
        
        # Output layer
        self.fc = torch.nn.Linear(hidden_channels if num_layers > 0 else num_node_features, num_classes)

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
        
        x = global_mean_pool(x, batch)  # Graph-level pooling
        x = self.fc(x)
        return x

class RGCN_node(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_unique_edge_attrs, num_layers=2, hidden_channels=16, dropout_rate=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        in_channels = num_node_features
        for i in range(num_layers):
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_unique_edge_attrs))
            self.bns.append(BatchNorm1d(hidden_channels))
            self.dropouts.append(Dropout(dropout_rate))
            in_channels = hidden_channels

        # Final linear layer
        self.final_linear = torch.nn.Linear(in_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)

        # Apply final linear layer
        x = self.final_linear(x)

        return F.log_softmax(x, dim=1)