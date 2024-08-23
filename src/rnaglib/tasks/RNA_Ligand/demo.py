import argparse
import dill as pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader as PygLoader

from rnaglib.tasks import GMSM
from rnaglib.representations import GraphRepresentation
from rnaglib.learning.task_models import RGCN_graph

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--frompickle', action='store_true', help="To load task from pickle")
args = parser.parse_args()

# Creating task
if args.frompickle is True:
    print('loading task from pickle')
    file_path = Path(__file__).parent / 'data' / 'gmsm.pkl'

    with open(file_path, 'rb') as file:
        ta = pickle.load(file)

else:
    print('generating task')
    ta = GMSM('gmsm')
    ta.dataset.add_representation(GraphRepresentation(framework='pyg'))

# Splitting data

train_ind, val_ind, test_ind = ta.split()
train_set = ta.dataset.subset(train_ind)
val_set = ta.dataset.subset(val_ind)
test_set = ta.dataset.subset(test_ind)

# Extracting graph representations

train_graphs = list((d['graph'] for d in train_set))
val_graphs = list((d['graph'] for d in val_set))
test_graphs = list((d['graph'] for d in test_set))


# Creating node level labels
def node_to_graph_label(dataset):
    # convert node levels to graph levels
    for data in dataset:
        data.y = data.y[0].argmax().unsqueeze(0)  # Convert to tensor of shape [1], otherwise batching will cause issues
    return dataset


train_graphs = node_to_graph_label(train_graphs)
val_graphs = node_to_graph_label(val_graphs)
test_graphs = node_to_graph_label(test_graphs)

# Creating loaders
pyg_train_loader = PygLoader(train_graphs, batch_size=8, shuffle=True)
pyg_val_loader = PygLoader(val_graphs, batch_size=8, shuffle=False)
pyg_test_loader = PygLoader(test_graphs, batch_size=8, shuffle=False)


def count_unique_edge_attrs(train_loader):
    all_edge_attrs = []
    for data in train_loader.dataset:
        if data.edge_attr is not None:
            all_edge_attrs.append(data.edge_attr)
    if all_edge_attrs:
        all_edge_attrs = torch.cat(all_edge_attrs, dim=0)
        return all_edge_attrs.unique().numel()
    else:
        return 0


# Extract dimension information
num_node_features = train_set[0]['graph'].x.shape[1]  # Number of node-level classes
num_classes = train_set[0]['graph'].y.shape[1]  # Number of graph-level classes
num_unique_edge_attrs = count_unique_edge_attrs(pyg_train_loader)  # Number of unique edge attributes
print(f"# node features {num_node_features}, # classes {num_classes}, # edge attributes {num_unique_edge_attrs}")

# Define model and parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = RGCN_graph(num_node_features, num_classes, num_unique_edge_attrs).to(device)

learning_rate = 0.01
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()


# Evaluate function
def validate(model, loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            val_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return val_loss / len(loader), correct / len(loader.dataset)


# Training loop
model.train()
print('Training model')
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for data in pyg_train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(pyg_train_loader)
    # Evaluate on validation set
    val_loss, val_acc = validate(model, pyg_val_loader, criterion)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
print("Training complete")

test_loss, test_accuracy, test_mcc, test_f1 = ta.evaluate(model, pyg_test_loader, criterion, device)

# Further improvements:
# - make RGCN more parametrisable
# - make the same thing with pytorch lightning
