import wandb
from rnaglib.tasks import BindingSite
from rnaglib.representations import GraphRepresentation
from rnaglib.data_loading import Collater
from rnaglib.learning.task_models import RGCN_node

from rnaglib.tasks import BenchmarkLigandBindingSiteDetection

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
import argparse
from pathlib import Path
import dill as pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--frompickle', action='store_true', help="To load task from pickle")
parser.add_argument('--layers', type=int, default=2, help="Number of RGCN layers")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
parser.add_argument('--task', type=str, default='RNA-Site', help="Name of the task")
parser.add_argument('--run_name', type=str, default=None, help="Name for this run")
args = parser.parse_args()

# Initialize wandb
wandb.init(
    project="rebuttal-experiments",
    config=args,
    name=args.run_name
)

# Log all arguments explicitly
wandb.config.update({
    "layers": args.layers,
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "task": args.task,
    "from_pickle": args.frompickle
})

# Creating task
if args.frompickle:
    print('loading task from pickle')
    file_path = Path(__file__).parent / 'data' / 'binding_site.pkl'
    with open(file_path, 'rb') as file:
        ta = pickle.load(file)
else:
    print('generating task')
    ta =  BenchmarkLigandBindingSiteDetection(args.task) #BindingSiteDetection(args.task)
    ta.dataset.add_representation(GraphRepresentation(framework='pyg'))

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
num_classes = 2 

print(f"# node features {num_node_features}, # classes {num_classes}, # edge attributes {num_unique_edge_attrs}")

# Define model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCN_node(num_node_features, num_classes, num_unique_edge_attrs, num_layers=args.layers, dropout_rate=0.1, hidden_channels=128)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# Evaluate function

def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    for batch in loader:
        graph = batch['graph']
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


# Training
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        graph = batch['graph']
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out, torch.flatten(graph.y).long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

for epoch in range(args.epochs):
    train_loss = train()
    train_acc, train_f1, train_auc, _, train_mcc = ta.evaluate(model, train_loader, criterion, device) 
    val_acc, val_f1, val_auc, val_loss, val_mcc = ta.evaluate(model, val_loader, criterion, device)  
    print(f"Epoch {epoch + 1}, "
      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
      f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, "
      f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, "
      f"Train MCC: {train_mcc:.4f}, Val MCC: {val_mcc:.4f}")
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_f1": train_f1,
        "val_f1": val_f1,
        "train_auc": train_auc,
        "val_auc": val_auc,
        "train_mcc": train_mcc,
        "val_mcc": val_mcc
    })

# Evaluate on test set
test_accuracy, test_f1, test_auc, test_loss, test_mcc = ta.evaluate(model, test_loader, criterion, device)  

print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')

# Log final test metrics to wandb
wandb.log({
    "test_accuracy": test_accuracy,
    "test_f1": test_f1,
    "test_auc": test_auc,
    "test_loss": test_loss,
    "test_mcc": test_mcc
})

wandb.finish()