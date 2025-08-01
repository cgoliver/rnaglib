import torch, functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean

from rnaglib.utils.misc import tonumpy
from .gvp_utils import GVP, LayerNorm, GVPConvLayer, MultiGVPConvLayer
    
class GVPModel(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        graph_level=False,
        num_layers=2,
        dropout_rate=0.5,
        multi_label=False,
        final_activation="sigmoid",
        device=None,
        node_in_dim=(4,2),
        node_h_dim=(4,2),
        edge_in_dim=(32,1),
        edge_h_dim=(32,1),
        multi_state=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.graph_level = graph_level
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.multi_label = multi_label
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.multi_state = multi_state
        
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, activations=(None, None))
        )
        if not self.multi_state:
            self.layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, self.edge_h_dim, drop_rate=self.dropout_rate) for _ in range(self.num_layers)
            )
        else:
            self.layers = nn.ModuleList(
                MultiGVPConvLayer(self.node_h_dim, self.edge_h_dim, drop_rate=self.dropout_rate) for _ in range(self.num_layers)
            )

        ns, _ = self.node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(self.node_h_dim),
            GVP(self.node_h_dim, (ns, 0))
        )

        # Initialize training components
        # Output layer
        if self.multi_label:
            self.final_linear = torch.nn.Linear(ns, self.num_classes)
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.final_activation = torch.nn.Identity()  # Use Identity for multi-label
        elif self.num_classes == 2:
            self.final_linear = torch.nn.Linear(ns, 1)
            # Weight will be set in train_model based on actual class distribution
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.final_activation = torch.nn.Identity()  # Use Identity for binary
        else:
            self.final_linear = torch.nn.Linear(ns, self.num_classes)
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
                self.device = "gpu"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
    
        self.configure_training()
    
    def forward(self, data): 
        h_V, edge_index, h_E, batch = data.h_V, data.edge_index, data.h_E, data.batch
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        if self.multi_state:
            h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))
        out = self.W_out(h_V)
        if self.graph_level:
            if batch is None:
                out = out.mean(dim=0, keepdims=True)
            else:
                out = scatter_mean(out, batch, dim=0)
        out = self.final_linear(out)
        out = self.final_activation(out)
        return out.squeeze(-1)
    
    def configure_training(self, learning_rate=0.001):
        """Configure training settings."""
        self.to(self.device)
        self.criterion = self.criterion.to(self.device)  # Move criterion to device for all cases
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def compute_loss(self, out, target):
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

    def train_model(self, task, epochs=500, print_all_epochs=False):
        if self.optimizer is None:
            self.configure_training()

        # Set class weights for binary classification based on actual distribution
        if self.num_classes == 2:
            if "0" in task.metadata["class_distribution"]:
                neg_count = float(task.metadata["class_distribution"]["0"])
                pos_count = float(task.metadata["class_distribution"]["1"])
            else:
                neg_count = float(task.metadata["class_distribution"][0])
                pos_count = float(task.metadata["class_distribution"][1])
            pos_weight = torch.tensor(np.sqrt(neg_count / pos_count)).to(self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        for epoch in range(epochs):
            # Training phase
            self.train()
            epoch_loss = 0
            num_batches = 0
            for batch in task.train_dataloader:
                if batch['gvp_graph'].num_nodes>0:
                    graph = batch["gvp_graph"].to(self.device)
                    self.optimizer.zero_grad()
                    out = self(graph)
                    loss = self.compute_loss(out, graph.y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

            # Validation phase
            if print_all_epochs or epoch % 10 == 0:
                val_metrics = self.evaluate(task, split="val")
                print(
                    f"Epoch {epoch}: train_loss = {epoch_loss / num_batches:.4f}, val_loss = {val_metrics['loss']:.4f}",
                )

    def inference(self, loader) -> tuple:
        """Evaluate model performance on a dataset.

        Args:
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
                graph = batch["gvp_graph"]
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
