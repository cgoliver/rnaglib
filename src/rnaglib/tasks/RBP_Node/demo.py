from rnaglib.transforms import GraphRepresentation
from rnaglib.tasks import ProteinBindingSiteDetection
from rnaglib.learning.task_models import RGCN_node

from rnaglib.encoders import ListEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef

import torch
import torch.optim as optim

import argparse
from pathlib import Path
import dill as pickle

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--frompickle", action="store_true", help="To load task from pickle")
args = parser.parse_args()

# Creating task
if args.frompickle is True:
    print("loading task from pickle")
    file_path = Path(__file__).parent / "data" / "protein_binding_site.pkl"

    with open(file_path, "rb") as file:
        ta = pickle.load(file)
else:
    print("generating task")
    ta = ProteinBindingSiteDetection("demoroot", recompute=True, debug=False)
    ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
    # Splitting dataset
    print("Splitting Dataset")
    ta.get_split_loaders()


# Here, you could potentially ask for more features
# If you do, you need to recreate the loaders and maybe graph

"""
features_computer = ta.dataset.features_computer
features_computer.add_feature(
    custom_encoders={"embeddings": ListEncoder(list_length=640)}
)
"""


# Printing statistics
info = ta.describe
num_node_features = info["num_node_features"]
num_unique_edge_attrs = info["num_edge_attributes"]
# need to set to 20 (or actual edge type cardinality) manually if not all edges are present, such as in debugging

# Train model
model = RGCN_node(num_node_features, 1, num_unique_edge_attrs, final_activation="sigmoid")
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
