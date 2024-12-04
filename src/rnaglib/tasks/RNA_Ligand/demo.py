import argparse
import dill as pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

import torch
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader as PygLoader

from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import RGCN_graph

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--frompickle', action='store_true', help="To load task from pickle")
args = parser.parse_args()
batch_size = 8

# Creating task
if args.frompickle is True:
    print('loading task from pickle')
    file_path = Path(__file__).parent / 'data' / 'gmsm.pkl'

    with open(file_path, 'rb') as file:
        ta = pickle.load(file)

else:
    print('generating task')
    ta = LigandIdentification('RNA-Ligand', recompute=True)

    # Splitting dataset
    print("Splitting Dataset")
    ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
    ta.set_loaders(batch_size=batch_size)

# Printing statistics
info = ta.describe(recompute=True)
num_node_features = info["num_node_features"]
num_classes = info["num_classes"]
num_unique_edge_attrs = info["num_edge_attributes"]
# need to set to 20 (or actual edge type #) if not all edges are present, such as in debugging

# Train model
model = RGCN_graph(num_node_features, num_classes, num_unique_edge_attrs)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=100)

# Final evaluation
test_metrics = ta.evaluate(model, ta.test_dataloader)
print(
    f"Test Loss: {test_metrics['loss']:.4f}, "
    f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
    f"Test F1 Score: {test_metrics['f1']:.4f}, "
    f"Test AUC: {test_metrics['auc']:.4f}, "
    f"Test MCC: {test_metrics['mcc']:.4f}"
)
