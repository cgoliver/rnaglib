from rnaglib.representations import GraphRepresentation
from rnaglib.tasks.RBP_Node.protein_binding_site import ProteinBindingSiteDetection, \
    BenchmarkProteinBindingSiteDetection
from rnaglib.learning.task_models import RGCN_node
from rnaglib.utils.feature_maps import ListEncoder

import torch
import torch.optim as optim

ta = BenchmarkProteinBindingSiteDetection('../data/RNA_VS', recompute=True)

# Here, you could potentially ask for more features
features_computer = ta.dataset.features_computer
features_computer.add_feature(custom_encoders={'embeddings': ListEncoder(list_length=640)})

# Choose your representation !
ta.dataset.add_representation(GraphRepresentation(framework='pyg'))

# Create loaders
train_loader, val_loader, test_loader = ta.get_split_loaders(batch_size=8)

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


for epoch in range(epochs):
    train()
    train_acc, train_f1, train_auc, train_loss, train_mcc = ta.evaluate(model, train_loader, criterion, device)
    val_acc, val_f1, val_auc, val_loss, val_mcc = ta.evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
          f" TrainAcc {train_acc:.4f} Val Acc: {val_acc:.4f}")

# Evaluate on test set
test_accuracy, test_f1, test_auc, test_loss, test_mcc = ta.evaluate(model, test_loader, criterion, device)
print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, '
      f'Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')
