from rnaglib.tasks import BindingSiteDetection
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import RGCN_node

import torch.optim as optim
import torch
import argparse
from pathlib import Path
import dill as pickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--frompickle", action="store_true", help="To load task from pickle"
)
args = parser.parse_args()


# Creating task

if args.frompickle is True:
    print("loading task from pickle")
    file_path = Path(__file__).parent / "data" / "binding_site.pkl"

    with open(file_path, "rb") as file:
        ta = pickle.load(file)
else:
    print("generating task")
    ta = BindingSiteDetection("RNA-Site")
    ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
train_loader, val_loader, test_loader = ta.get_split_loaders()

# Printing statistics
info = ta.describe
num_node_features = info["num_node_features"]
num_classes = info["num_classes"]
num_unique_edge_attrs = info["num_edge_attributes"]


# Define model
learning_rate = 0.0001
epochs = 5  # 100

device = "cpu"

model = RGCN_node(num_node_features, num_classes, num_unique_edge_attrs)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Evaluate function

for epoch in range(epochs):
    # Training step
    model.train()
    for batch in train_loader:
        graph = batch["graph"]
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out, torch.flatten(graph.y).long())
        loss.backward()
        optimizer.step()

    # Evaluation
    train_metrics = ta.evaluate(model, train_loader, device, criterion)
    val_metrics = ta.evaluate(model, val_loader, device, criterion)
    print(
        f"Epoch {epoch + 1}, "
        f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
        f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
    )


# Final evaluation
test_metrics = ta.evaluate(model, test_loader, device, criterion)
print(
    f"Test Loss: {test_metrics['loss']:.4f}, "
    f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
    f"Test F1 Score: {test_metrics['f1']:.4f}, "
    f"Test AUC: {test_metrics['auc']:.4f}, "
    f"Test MCC: {test_metrics['mcc']:.4f}"
)
