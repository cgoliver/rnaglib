import torch
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout


class RGCN_graph(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 num_classes,
                 num_unique_edge_attrs,
                 num_layers=2,
                 hidden_channels=128,
                 dropout_rate=0.5,
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        # Input layer
        in_channels = num_node_features
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(in_channels, hidden_channels, num_unique_edge_attrs)
            )
            self.bns.append(BatchNorm1d(hidden_channels))
            self.dropouts.append(Dropout(dropout_rate))
            in_channels = hidden_channels

        # Output layer
        self.final_linear = torch.nn.Linear(hidden_channels, num_classes)

        # Initialize training components
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.device = None

    def forward(self, data):
        x, edge_index, edge_type, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)

        x = global_mean_pool(x, batch)  # Graph-level pooling
        x = self.final_linear(x)
        return x

    def configure_training(
            self, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Configure training settings."""
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def train_model(self, task, epochs=500):
        """Training loop with evaluation."""
        if self.optimizer is None:
            self.configure_training()

        for epoch in range(epochs):
            # Training phase
            self.train()
            for batch in task.train_dataloader:
                graph = batch["graph"].to(self.device)
                self.optimizer.zero_grad()
                out = self(graph)
                print(out.flatten())
                print(graph.y.argmax())
                loss = self.criterion(out.flatten(), graph.y.argmax())
                loss.backward()
                self.optimizer.step()

            # Evaluation phase
            train_metrics = self.evaluate(task.train_dataloader)
            val_metrics = self.evaluate(task.val_dataloader)

            print(
                f"Epoch {epoch + 1}, "
                f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )


class RGCN_node(torch.nn.Module):
    def __init__(
            self,
            num_node_features,
            num_classes,
            num_unique_edge_attrs,
            num_layers=2,
            hidden_channels=128,
            dropout_rate=0.5,
            final_activation=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.final_activation = final_activation

        in_channels = num_node_features
        for i in range(num_layers):
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_unique_edge_attrs))
            self.bns.append(BatchNorm1d(hidden_channels))
            self.dropouts.append(Dropout(dropout_rate))
            in_channels = hidden_channels

        # Final linear layer, for the two-class case, pytorch has a slightly different formulation, with BCE
        if num_classes == 2:
            self.final_linear = torch.nn.Linear(in_channels, 1)
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.final_linear = torch.nn.Linear(in_channels, num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()

        # Initialize training components
        self.optimizer = None
        self.device = None
        if not self.final_activation is None:
            if self.final_activation == "sigmoid":
                self.final_activation = torch.nn.Sigmoid()
            if self.final_activation == "softmax":
                self.final_activation = torch.nn.Softmax()

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)

        # Apply final linear layer
        x = self.final_linear(x)
        if not self.final_activation is None:
            self.final_activation(x)
        return x

    def configure_training(self, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Configure training settings."""
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def train_model(self, task, epochs=500):
        """Training loop with evaluation."""
        if self.optimizer is None:
            self.configure_training()

        for epoch in range(epochs):
            # Training phase
            self.train()
            for batch in task.train_dataloader:
                graph = batch["graph"].to(self.device)
                self.optimizer.zero_grad()
                out = self(graph)
                target = graph.y

                # If just two classes, flatten outputs since BCE behavior expects equal dimensions and CE (N,k):(N)
                # Otherwise CE expects long as outputs
                if self.num_classes == 2:
                    out = out.flatten()
                else:
                    target = target.long()
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()

            # Evaluation phase
            train_metrics = task.evaluate(model=self, loader=task.train_dataloader)
            val_metrics = task.evaluate(model=self, loader=task.val_dataloader)

            print(
                f"Epoch {epoch + 1}, "
                f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
