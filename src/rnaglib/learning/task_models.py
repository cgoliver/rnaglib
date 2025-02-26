import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import RGCNConv, global_mean_pool
from rnaglib.utils.misc import tonumpy


class PygModel(torch.nn.Module):
    def __init__(
        self,
        num_node_features,
        num_classes,
        num_unique_edge_attrs=20,
        graph_level=False,
        num_layers=2,
        hidden_channels=128,
        dropout_rate=0.5,
        multi_label=False,
        final_activation="sigmoid"
    ):
        super().__init__()
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.num_unique_edge_attrs = num_unique_edge_attrs
        self.graph_level = graph_level
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate
        self.multi_label = multi_label

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()

        if final_activation == "sigmoid":
            self.final_activation = torch.nn.Sigmoid()
        elif final_activation == "softmax":
            self.final_activation = torch.nn.Softmax(dim=1)
        else:
            self.final_activation = torch.nn.Identity()

        # Input layer

        self.input_non_linear_layer = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, self.hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
        )

        for i in range(self.num_layers):
            self.convs.append(RGCNConv(self.hidden_channels, self.hidden_channels, self.num_unique_edge_attrs))
            self.bns.append(BatchNorm1d(self.hidden_channels))
            self.dropouts.append(Dropout(self.dropout_rate))

        # Initialize training components
        # Output layer
        if self.multi_label:
            self.final_linear = torch.nn.Linear(self.hidden_channels, self.num_classes)
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            if num_classes == 2:
                self.final_linear = torch.nn.Linear(self.hidden_channels, 1)
                self.criterion = torch.nn.BCEWithLogitsLoss()
            else:
                self.final_linear = torch.nn.Linear(self.hidden_channels, self.num_classes)
                self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = None
        self.device = None

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_non_linear_layer(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)
        if self.graph_level:
            x = global_mean_pool(x, batch)  # Graph-level pooling
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x

    def configure_training(self, learning_rate=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Configure training settings."""
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def compute_loss(self, out, target):
        # If just two classes, flatten outputs since BCE behavior expects equal dimensions and CE (N,k):(N)
        # Otherwise CE expects long as outputs
        if not self.multi_label:
            if self.num_classes == 2:
                out = out.flatten()
            else:
                target = target.long()
        loss = self.criterion(out, target)
        return loss

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
                loss = self.compute_loss(out, graph.y)
                loss.backward()
                self.optimizer.step()

            # Evaluation phase
            train_metrics = self.evaluate(task, split="train")
            val_metrics = self.evaluate(task, split="val")

            print(
                f"Epoch {epoch + 1}, "
                f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}"
                f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )

            if self.multi_label:
                print(
                    f"Train Jaccard: {train_metrics['jaccard']:.4f}, Val Jaccard: {val_metrics['jaccard']:.4f}"
                )

    def inference(self, loader) -> tuple:
        """
        Evaluate model performance on a dataset
            loader: Data loader to use
        Returns:
            3 list containing predictions, probs, targets if residue level,
        else 3 np arrays
        """
        self.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"]
                graph = graph.to(self.device)
                out = self(graph)
                labels = graph.y
                loss = self.compute_loss(out, labels)
                total_loss += loss.item()

                # get preds and probas + cast to numpy
                if self.multi_label:
                    probs = torch.sigmoid(out)
                    preds = (probs > 0.5).float()
                else:
                    if self.num_classes == 2:
                        probs = torch.sigmoid(out.flatten())
                        preds = (probs > 0.5).float()
                    else:
                        probs = torch.softmax(out, dim=1)
                        preds = probs.argmax(dim=1)
                probs = tonumpy(probs)
                preds = tonumpy(preds)
                labels = tonumpy(labels)

                # split predictions per RNA if residue level
                if not self.graph_level:
                    cumulative_sizes = tuple(tonumpy(graph.ptr))
                    probs = [probs[start:end] for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:])]
                    preds = [preds[start:end] for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:])]
                    labels = [labels[start:end] for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:])]
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        if self.graph_level:
            all_probs = np.stack(all_probs)
            all_preds = np.stack(all_preds)
            all_labels = np.stack(all_labels)
        mean_loss = total_loss / len(loader)
        return mean_loss, all_preds, all_probs, all_labels

    def get_dataloader(self, task, split="test"):
        if split == "test":
            dataloader = task.test_dataloader
        elif split == "val":
            dataloader = task.val_dataloader
        else:
            dataloader = task.train_dataloader
        return dataloader

    def evaluate(self, task, split="test"):
        dataloader = self.get_dataloader(task=task, split=split)
        mean_loss, all_preds, all_probs, all_labels = self.inference(loader=dataloader)
        metrics = task.compute_metrics(all_preds, all_probs, all_labels)
        metrics["loss"] = mean_loss
        return metrics
