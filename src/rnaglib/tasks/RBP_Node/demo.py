from rnaglib.transforms import GraphRepresentation
from rnaglib.tasks.RBP_Node.protein_binding_site import ProteinBindingSiteDetection
from rnaglib.data_loading import Collater
from rnaglib.learning.task_models import RGCN_node

from torch.utils.data import DataLoader
from rnaglib.encoders import ListEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

import torch
import torch.optim as optim

ta = ProteinBindingSiteDetection('demoroot', recompute=True, debug=True)

# Here, you could potentially ask for more features
features_computer = ta.dataset.features_computer
features_computer.add_feature(custom_encoders={'embeddings': ListEncoder(list_length=640)})

# Choose your representation !
ta.dataset.add_representation(GraphRepresentation(framework='pyg'))


ta.set_loaders()
# Create loaders
train_loader, val_loader, test_loader = ta.get_split_loaders(batch_size=8)
'''
# Splitting dataset

train_ind, val_ind, test_ind = ta.train_ind, ta.val_ind, ta.test_ind 
train_set = ta.dataset.subset(train_ind)
val_set = ta.dataset.subset(val_ind)
test_set = ta.dataset.subset(test_ind)

print(train_set[0])
print(len(train_set))

# Creating loaders

print("Creating loaders")
collater = Collater(train_set)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collater)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collater)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=collater)
'''

# Create model
num_node_features = features_computer.input_dim
num_unique_edge_attrs = 20
num_classes = 2
model = RGCN_node(num_node_features, num_classes, num_unique_edge_attrs)

# Set up optim and device
learning_rate = 0.001
epochs = 100
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Training
def train():
    model.train()
    for batch in train_loader:
        graph = batch['graph']
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out, torch.flatten(graph.y).long())
        loss.backward()
        optimizer.step()

def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    for batch in loader:
        graph = batch["graph"]
        graph = graph.to(device)
        out = model(graph)
        loss = criterion(out, torch.flatten(graph.y).long())
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(graph.y.tolist())

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return accuracy, f1, auc, avg_loss, mcc


for epoch in range(epochs):
    train()
    train_acc, train_f1, train_auc, train_loss, train_mcc = evaluate(train_loader) 
    val_acc, val_f1, val_auc, val_loss, val_mcc = evaluate(val_loader) 
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
          f" TrainAcc {train_acc:.4f} Val Acc: {val_acc:.4f}")

# Evaluate on test set
test_accuracy, test_f1, test_auc, test_loss, test_mcc = ta.evaluate(model, device) #test_loader, criterion, 
print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, '
      f'Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')
