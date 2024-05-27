#!/usr/bin/env python
# coding: utf-8
from rnaglib.tasks import BenchmarkLigandBindingSiteDetection, BindingSiteDetection
from rnaglib.representations import GraphRepresentation
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, RGCNConv
import torch.optim as optim
import wandb
from collections import Counter
from torch.nn import Linear
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from pathlib import Path
from networkx import get_node_attributes


if Path('test_fri').exists():
    shutil.rmtree('test_fri')

ta = BenchmarkLigandBindingSiteDetection('test_fri')
train_ind, val_ind, test_ind = ta.splitter(ta.dataset)
ta.dataset.add_representation(GraphRepresentation(framework = 'pyg'))


train_data_list, val_data_list, test_data_list = [], [], []
for ind in train_ind:
    train_data_list.append(ta.dataset[ind]['graph'])
for ind in val_ind:
    val_data_list.append(ta.dataset[ind]['graph'])
for ind in test_ind:
    test_data_list.append(ta.dataset[ind]['graph'])

train_loader = DataLoader(train_data_list, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=2, shuffle=False)
test_loader = DataLoader(test_data_list, batch_size=2, shuffle=False)

# rnaglib converter provides right format for pyg objects but its contents are lists or lists of lists, not tensors.
def pyg_converter(loader):
    for data in loader.dataset:
        data.edge_index =  torch.tensor(data.edge_index).t()
        data.edge_attr =  torch.tensor(data.edge_attr)
        data.y = data.y.squeeze().long()

pyg_converter(train_loader)
pyg_converter(val_loader)
pyg_converter(test_loader)

for batch in train_loader:
    print(batch)
    print(f'Batch node features shape: {batch.x.shape}')
    print(f'Batch edge index shape: {batch.edge_index.shape}')
    print(f'Batch labels shape: {batch.y.shape}')
    break

num_node_features = train_data_list[0].num_features
num_classes = 2 

# Printing some statistics

def calculate_length_statistics(loader):
    lengths = [data.x.shape[0] for data in loader.dataset]
    
    max_length = np.max(lengths)
    min_length = np.min(lengths)
    avg_length = np.mean(lengths)
    median_length = np.median(lengths)
    
    return {
        "max_length": max_length,
        "min_length": min_length,
        "average_length": avg_length,
        "median_length": median_length
    }

stats = calculate_length_statistics(train_loader)
print("Max Length:", stats["max_length"])
print("Min Length:", stats["min_length"])
print("Average Length:", stats["average_length"])
print("Median Length:", stats["median_length"])


def calculate_fraction_of_ones(loader):
    total_ones = 0
    total_elements = 0
    
    for data in loader.dataset:
        y = data.y
        total_ones += (y == 1).sum().item()
        total_elements += y.numel()
    
    fraction_of_ones = total_ones / total_elements if total_elements > 0 else 0
    return fraction_of_ones

fraction = calculate_fraction_of_ones(train_loader)
print("Fraction of positives:", fraction)

def count_unique_edge_attrs(train_loader):
    unique_edge_attrs = set()
    
    for data in train_loader.dataset:
        unique_edge_attrs.update(data.edge_attr.tolist())
    
    return len(unique_edge_attrs), unique_edge_attrs

num_unique_edge_attrs, unique_edge_attrs = count_unique_edge_attrs(train_loader)
print("Number of unique edge attributes:", num_unique_edge_attrs)
print("Unique edge attributes:", unique_edge_attrs)

# Model

wandb.init(project="gcn-node-classification", config={
    "learning_rate": 0.001,
    "epochs": 5000,
    "batch_size": 1
})

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_unique_edge_attrs):
        super(GCN, self).__init__()
        self.conv1 = RGCNConv(num_node_features, 16, num_unique_edge_attrs)
        #self.conv2 = GCNConv(16, 32) 
        #self.conv3 = GCNConv(32, 16) 
        self.conv4 = RGCNConv(16, num_classes, num_unique_edge_attrs)

    def forward(self, data):
        x, edge_index, edge_type= data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        x = self.conv4(x, edge_index, edge_type)
    
        return F.log_softmax(x, dim=1)


model = GCN(num_node_features, num_classes, num_unique_edge_attrs)


# Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

all_labels = []
for data in train_data_list:
    all_labels.extend(data.y.tolist())
class_counts = Counter(all_labels)
total_samples = len(all_labels)
class_weights = {cls: total_samples/count for cls, count in class_counts.items()}
weights = [class_weights[i] for i in range(num_classes)]
class_weights_tensor = torch.tensor(weights).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)


# Evaluation function to get predictions and calculate metrics
def get_predictions_and_loss(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(data.y.tolist())
    
    avg_loss = total_loss / len(loader)
    return all_preds, all_labels, avg_loss

def calculate_metrics(loader):
    preds, labels, avg_loss = get_predictions_and_loss(loader)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    return accuracy, f1, auc, avg_loss, mcc  

# Training function
def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": loss.item()})

# Main training loop
for epoch in range(5000):
    train()
    train_acc, train_f1, train_auc, train_loss, train_mcc = calculate_metrics(train_loader) 
    val_acc, val_f1, val_auc, val_loss, val_mcc = calculate_metrics(val_loader)  
    print(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    wandb.log({
        "train_acc": train_acc,
        "train_f1": train_f1,
        "train_auc": train_auc,
        "train_loss": train_loss,
        "train_mcc": train_mcc, 
        "val_acc": val_acc,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "val_loss": val_loss,
        "val_mcc": val_mcc  
    })

# Final evaluation on test set
test_accuracy, test_f1, test_auc, test_loss, test_mcc = calculate_metrics(test_loader)  
print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')  
wandb.log({"test_accuracy": test_accuracy, "test_f1": test_f1, "test_auc": test_auc, "test_loss": test_loss, "test_mcc": test_mcc})  

wandb.finish()