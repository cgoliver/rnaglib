from rnaglib.tasks import InverseFolding
from rnaglib.representations import GraphRepresentation
from rnaglib.data_loading import Collater
from rnaglib.learning.task_models import RGCN_node

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
import argparse
from pathlib import Path
import dill as pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--frompickle', action='store_true', help="To load task from pickle")
args = parser.parse_args()

# Creating task

if  args.frompickle is True:
    print('loading task from pickle')
    file_path = Path(__file__).parent / 'data' / 'inverse_folding.pkl'

    with open(file_path, 'rb') as file:
        ta = pickle.load(file)
else:
    print('generating task')
    ta = InverseFolding('RNA-IF')
    ta.dataset.add_representation(GraphRepresentation(framework = 'pyg'))

# Splitting dataset

train_ind, val_ind, test_ind = ta.split()
train_set = ta.dataset.subset(train_ind)
val_set = ta.dataset.subset(val_ind)
test_set = ta.dataset.subset(test_ind)


# Creating loaders

print('Creating loaders')
collater = Collater(train_set)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collater)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collater)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=collater)

# Printing statistics

for batch in train_loader:
    graph = batch['graph']
    print(f'Batch node features shape: \t{graph.x.shape}')
    print(f'Batch edge index shape: \t{graph.edge_index.shape}')
    print(f'Batch labels shape: \t\t{graph.y.shape}')
    break

def count_unique_edge_attrs(train_loader):
    unique_edge_attrs = set()
    for batch in train_loader.dataset:
        unique_edge_attrs.update(batch['graph'].edge_attr.tolist())
    return len(unique_edge_attrs)

num_unique_edge_attrs = count_unique_edge_attrs(train_loader)
num_node_features = train_set[0]['graph'].x.shape[1]
num_classes = train_set[0]['graph'].y.shape[1] 

print(f"# node features {num_node_features}, # classes {num_classes}, # edge attributes {num_unique_edge_attrs}")

# Define model
learning_rate = 0.001
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCN_node(num_node_features, num_classes, num_unique_edge_attrs)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Evaluate function

def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    with torch.no_grad(): 
        for batch in loader:
            graph = batch['graph']
            graph = graph.to(device)
            out = model(graph)
            loss = criterion(out, graph.y)# torch.flatten(graph.y).long())
            total_loss += loss.item()
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.tolist())
            labels = graph.y.argmax(dim=1)
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())

        
        avg_loss = total_loss / len(loader)
        
        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)
        all_probs = torch.cat(all_probs, dim=0)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        return accuracy, f1, auc, avg_loss, mcc

# Training

def train():
    model.train()
    for batch in train_loader:
        graph = batch['graph']
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out, graph.y)#torch.flatten(graph.y).long())
        loss.backward()
        optimizer.step()

for epoch in range(epochs):
    train()
    train_acc, train_f1, train_auc, train_loss, train_mcc = evaluate(train_loader) 
    val_acc, val_f1, val_auc, val_loss, val_mcc = evaluate(val_loader)  
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, TrainAcc {train_acc:.4f} Val Acc: {val_acc:.4f}")

test_accuracy, test_f1, test_auc, test_loss, test_mcc = ta.evaluate(model, test_loader, criterion, device)