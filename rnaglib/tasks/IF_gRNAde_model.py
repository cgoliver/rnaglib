#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rnaglib.tasks import gRNAde, BindingSiteDetection, BenchmarkLigandBindingSiteDetection, InverseFolding
from rnaglib.representations import GraphRepresentation
from rnaglib.data_loading import Collater
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, RGCNConv
import torch.optim as optim
import wandb
from collections import Counter
from torch.nn import BatchNorm1d, Dropout
import shutil
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from pathlib import Path


# In[3]:


#change when ready
if Path('test_IF').exists():
    shutil.rmtree('test_IF')
ta = gRNAde(root='test_IF-gRNAde', recompute=False)
#shutil.rmtree('ifchim') #somehow, if I don't remove the folder, the graph rep loses node infor

ta = InverseFolding(root='ifchim')
ta.dataset.add_representation(GraphRepresentation(framework = 'pyg'))


# In[5]:


train_ind, val_ind, test_ind = ta.split()
train_set = ta.dataset.subset(train_ind)
val_set = ta.dataset.subset(val_ind)
test_set = ta.dataset.subset(test_ind)

print(train_set)
print(val_set)

print(len(train_set))
print(len(val_set))


# In[7]:


collater = Collater(train_set)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collater)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collater)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=collater)


# In[46]:


for batch in train_loader:
    print(batch)
    graph = batch['graph']
    print(f'Batch node features shape: \t{graph.x.shape}')
    print(f'Batch edge index shape: \t{graph.edge_index.shape}')
    print(f'Batch labels shape: \t\t{graph.y.shape}')
    break

def calculate_length_statistics(loader):
    lengths = [data['graph'].x.shape[0] for data in loader.dataset]
    
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

def count_unique_edge_attrs(train_loader):
    unique_edge_attrs = set()
    
    for batch in train_loader.dataset:
        unique_edge_attrs.update(batch['graph'].edge_attr.tolist())
    
    return len(unique_edge_attrs), unique_edge_attrs

num_unique_edge_attrs, unique_edge_attrs = count_unique_edge_attrs(train_loader)
print("Number of unique edge attributes:", num_unique_edge_attrs)
print("Unique edge attributes:", unique_edge_attrs)


# In[21]:


train_loader.dataset[0]['graph'].x.size(1)


# # Model

# In[35]:


wandb.init(project="inverse_design_gRNAde", config={
    "learning_rate": 0.0001,
    "epochs": 500,
    "batch_size": 1,
    "dropout_rate": 0.1,  
    "num_layers": 2, 
    "batch_norm": True 
})


# In[42]:


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.bn1 = BatchNorm1d(16)  
        self.dropout1 = Dropout(0.1) 
        self.conv2 = GCNConv(16, num_classes)
        self.bn2 = BatchNorm1d(num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x) 
    
        return F.log_softmax(x, dim=1)


# In[43]:


num_classes = train_loader.dataset[0]['graph'].x.size(1)
model = GCN(train_set.input_dim, num_classes)


# In[44]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()#weight=class_weights_tensor)


# In[ ]:





# In[74]:


# from chatgpt
def train():
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch in train_loader:
        graph = batch['graph']
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)

        # Convert one-hot encoded labels to class indices
        labels = graph.y.argmax(dim=1).long()
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        # Log the loss
        wandb.log({"train_loss": loss.item()})
        running_loss += loss.item()

        # Convert one-hot encoded predictions to class indices
        preds = out.argmax(dim=1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in loader:
            graph = batch['graph']
            graph = graph.to(device)
            out = model(graph)

            # Convert one-hot encoded labels to class indices
            labels = graph.y.argmax(dim=1).long()
            loss = criterion(out, labels)
            total_loss += loss.item()

            # Convert one-hot encoded predictions to class indices
            preds = out.argmax(dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

# Main training loop
for epoch in range(500):
    train_loss, train_accuracy = train()
    val_loss, val_accuracy = evaluate(val_loader)
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })


# In[ ]:




