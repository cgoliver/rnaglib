"""Demo for training a simple model using an rnaglib task"""

import argparse
from pathlib import Path
import dill as pickle

from rnaglib.tasks import gRNAde
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import RGCN_node


parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--frompickle", action="store_true", help="To load task from pickle"
)
args = parser.parse_args()

# Creating task
if args.frompickle is True:
    print("Loading task from pickle")
    file_path = Path(__file__).parent / "data" / "gRNAde_first_run_08112024.pkl"

    with open(file_path, "rb") as file:
        ta = pickle.load(file)
else:
    print("Generating task")
    ta = gRNAde("gRNAde", recompute=True)
    ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
    # Splitting dataset
    print("Splitting dataset")
    ta.get_split_loaders()


# Printing statistics
info = ta.describe
num_node_features = info["num_node_features"]
num_classes = info["num_classes"]
num_unique_edge_attrs = info["num_edge_attributes"]
# need to set to 20 (or actual edge type #)  if not all edges are present, such as in debugging

# Train model
model = RGCN_node(num_node_features, num_classes, num_unique_edge_attrs)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=100)

# Final evaluation
test_metrics = ta.evaluate(model, ta.test_dataloader)
print(
    f"Test Loss: {test_metrics['loss']:.4f}, "
    f"Sequence Recovery: {test_metrics['accuracy']:.4f}, "
    f"MCC: {test_metrics['mcc']:.4f}, "
    f"Macro F1: {test_metrics['macro_f1']:.4f}, "
    f"Mean AUC: {test_metrics['mean_auc']:.4f}, "
    f"Coverage: {test_metrics['coverage']:.4f}, "
    f"Non-standard ratio: {test_metrics['non_standard_ratio']:.4f}"
)
