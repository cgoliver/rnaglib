import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
try:
    from torch_geometric.nn import RGCNConv, global_mean_pool
except ImportError:
    # Fallback for when torch_geometric is not available (e.g., during docs build)
    RGCNConv = None
    global_mean_pool = None

from rnaglib.utils.misc import tonumpy


class PygModel(torch.nn.Module):
    @classmethod
    def from_task(cls, 
                  task, 
                  num_node_features=None, 
                  num_classes=None,
                  graph_level=None, 
                  multi_label=None,
                  **model_args):
        """Create a model based on task metadata.

        Will fail if number of node features is not the default.

        :param task: Task object containing metadata
        :param num_node_features: Number of node features (defaults to task metadata)
        :param num_classes: Number of classes (defaults to task metadata)
        :param graph_level: Whether task is graph-level (defaults to task metadata)
        :param multi_label: Whether task is multi-label (defaults to task metadata)
        :param model_args: Additional keyword arguments to pass to model constructor
        :return: PygModel instance configured for the task
        """
        if num_node_features is None:
            num_node_features = task.metadata["num_node_features"]
        if num_classes is None:
            num_classes = task.metadata["num_classes"]
        if graph_level is None:
            graph_level = task.metadata["graph_level"]
        if multi_label is None:
            multi_label = task.metadata["multi_label"]

        activation = 'softmax' if num_classes > 2 else 'sigmoid'

        return cls(
                  num_node_features=num_node_features,
                  num_classes=num_classes,
                  graph_level=graph_level,
                  multi_label=multi_label,
                  final_activation=activation,
                  **model_args
                  )
        pass

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
        final_activation="sigmoid",
        device=None
    ):
        """Initialize PygModel.

        :param num_node_features: Number of input features per node
        :param num_classes: Number of output classes
        :param num_unique_edge_attrs: Number of unique edge attribute types
        :param graph_level: If True, perform graph-level classification; if False, node-level
        :param num_layers: Number of RGCN layers
        :param hidden_channels: Number of hidden channels in each layer
        :param dropout_rate: Dropout rate for regularization
        :param multi_label: If True, use multi-label classification
        :param final_activation: Final activation function ("sigmoid", "softmax", or None)
        :param device: Device to run on ("cuda", "mps", or "cpu"). If None, auto-detects
        """
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
            BatchNorm1d(self.hidden_channels),
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
            self.final_activation = torch.nn.Identity()  # Use Identity for multi-label
        elif self.num_classes == 2:
            self.final_linear = torch.nn.Linear(self.hidden_channels, 1)
            # Weight will be set in train_model based on actual class distribution
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.final_activation = torch.nn.Identity()  # Use Identity for binary
        else:
            self.final_linear = torch.nn.Linear(self.hidden_channels, self.num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()
            if final_activation == "sigmoid":
                self.final_activation = torch.nn.Sigmoid()
            elif final_activation == "softmax":
                self.final_activation = torch.nn.Softmax(dim=1)
            else:
                self.final_activation = torch.nn.Identity()

        self.optimizer = None
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.configure_training()

    def forward(self, data):
        """Forward pass through the model.

        :param data: PyTorch Geometric data object containing graph structure and features
        :return: Model output predictions
        """
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.input_non_linear_layer(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_type)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x)

        if self.graph_level:
            x = global_mean_pool(x, batch)
        x = self.final_linear(x)
        x = self.final_activation(x)
        return x

    def configure_training(self, learning_rate=0.001):
        """Configure training settings.

        :param learning_rate: Learning rate for the Adam optimizer
        """
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)  # Move criterion to device for all cases
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def compute_loss(self, out, target):
        """Compute loss between predictions and targets.

        :param out: Model output predictions
        :param target: Ground truth labels
        :return: Loss value
        """
        # If just two classes, flatten outputs since BCE behavior expects equal dimensions and CE (N,k):(N)
        # Otherwise CE expects long as outputs
        if self.multi_label:
            target = target.float()
            pass
        else:
            if self.num_classes == 2:
                out = out.flatten()
        loss = self.criterion(out, target)
        return loss

    def train_model(self, task, epochs=500):
        """Train the model on a task.

        :param task: Task object containing train and validation dataloaders
        :param epochs: Number of training epochs
        """
        if self.optimizer is None:
            self.configure_training()

        # Set class weights for binary classification based on actual distribution
        if self.num_classes == 2:
            neg_count = float(task.metadata["class_distribution"]["0"])
            pos_count = float(task.metadata["class_distribution"]["1"])
            pos_weight = torch.tensor(np.sqrt(neg_count / pos_count)).to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for epoch in range(epochs):
            # Training phase
            self.train()
            epoch_loss = 0
            num_batches = 0
            for batch in task.train_dataloader:
                graph = batch["graph"].to(self.device)
                self.optimizer.zero_grad()
                out = self(graph)
                loss = self.compute_loss(out, graph.y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            # Validation phase
            if epoch % 10 == 0:
                val_metrics = self.evaluate(task, split="val")
                print(
                    f"Epoch {epoch}: train_loss = {epoch_loss / num_batches:.4f}, val_loss = {val_metrics['loss']:.4f}",
                )

    def inference(self, loader) -> tuple:
        """Evaluate model performance on a dataset.

        :param loader: DataLoader containing the data to evaluate on
        :return: Tuple of (mean_loss, predictions, probabilities, labels)
            If residue-level: predictions, probs, labels are lists
            If graph-level: predictions, probs, labels are numpy arrays
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

                # For binary/multilabel, threshold the logits at 0 (equivalent to prob > 0.5 after sigmoid)
                preds = (out > 0).float() if (self.multi_label or self.num_classes == 2) else out.argmax(dim=1)
                probs = out

                probs = tonumpy(probs)
                preds = tonumpy(preds)
                labels = tonumpy(labels)

                # split predictions per RNA if residue level
                if not self.graph_level:
                    cumulative_sizes = tuple(tonumpy(graph.ptr))
                    probs = [
                        probs[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                    preds = [
                        preds[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
                    labels = [
                        labels[start:end]
                        for start, end in zip(cumulative_sizes[:-1], cumulative_sizes[1:], strict=False)
                    ]
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
        """Get a dataloader for a specific split.

        :param task: Task object containing dataloaders
        :param split: Split to get ("test", "val", or "train")
        :return: DataLoader for the specified split
        """
        if split == "test":
            dataloader = task.test_dataloader
        elif split == "val":
            dataloader = task.val_dataloader
        else:
            dataloader = task.train_dataloader
        return dataloader

    def evaluate(self, task, split="test"):
        """Evaluate model on a task split and compute metrics.

        :param task: Task object containing dataloaders and compute_metrics method
        :param split: Split to evaluate on ("test", "val", or "train")
        :return: Dictionary of metric names and values, including loss
        """
        dataloader = self.get_dataloader(task=task, split=split)
        mean_loss, all_preds, all_probs, all_labels = self.inference(loader=dataloader)
        metrics = task.compute_metrics(all_preds, all_probs, all_labels)
        metrics["loss"] = mean_loss
        return metrics
