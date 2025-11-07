import numpy as np
import torch


def print_statistics(loader):
    """Print statistics about a dataset from a PyTorch dataloader.

    :param loader: PyTorch DataLoader containing the dataset
    :return: Dictionary with max_length, min_length, average_length, median_length
    """
    lengths = [data["graph"].x.shape[0] for data in loader.dataset]

    max_length = np.max(lengths)
    min_length = np.min(lengths)
    avg_length = np.mean(lengths)
    median_length = np.median(lengths)

    for batch in loader:
        graph = batch["graph"]
        print(f"Batch node features shape: \t{graph.x.shape}")
        print(f"Batch edge index shape: \t{graph.edge_index.shape}")
        print(f"Batch labels shape: \t\t{graph.y.shape}")
        break

    print("Max Length:", max_length)
    print("Min Length:", min_length)
    print("Average Length:", avg_length)
    print("Median Length:", median_length)

    return {
        "max_length": max_length,
        "min_length": min_length,
        "average_length": avg_length,
        "median_length": median_length,
    }


class DummyResidueModel(torch.nn.Module):
    """Dummy model for residue-level tasks that returns random predictions."""
    
    def __init__(self, num_classes=2):
        """Initialize dummy residue model.

        :param num_classes: Number of output classes
        """
        super(DummyResidueModel, self).__init__()
        self.device = torch.device("cpu")  # Default device is CPU
        self.num_classes = num_classes

    def forward(self, g):
        """Forward pass returning random predictions.

        :param g: PyTorch Geometric graph data object
        :return: Random predictions tensor of shape (num_nodes, num_classes)
        """
        # PyTorch syntax changes a bit for n=2
        predicted_classes = self.num_classes if self.num_classes > 2 else 1
        return torch.rand(g.x.shape[0], predicted_classes)


class DummyGraphModel(torch.nn.Module):
    """Dummy model for graph-level tasks that returns random predictions."""
    
    def __init__(self, num_classes=2):
        """Initialize dummy graph model.

        :param num_classes: Number of output classes
        """
        super(DummyGraphModel, self).__init__()
        self.device = torch.device("cpu")  # Default device is CPU
        self.num_classes = num_classes

    def forward(self, g):
        """Forward pass returning random predictions.

        :param g: PyTorch Geometric graph data object
        :return: Random predictions tensor of shape (1, num_classes)
        """
        # PyTorch syntax changes a bit for n=2
        predicted_classes = self.num_classes if self.num_classes > 2 else 1
        return torch.rand(1, predicted_classes)
