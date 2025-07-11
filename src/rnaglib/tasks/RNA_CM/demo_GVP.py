import torch
import numpy as np

from rnaglib.learning.gvp import GVPModel
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GVPGraphRepresentation

ta = ChemicalModification(
    root="RNA_CM",
    coors_annotation="P_only",
    recompute=False,
    debug=False,
)
# need to set coors_annotation="heavy_only" (or "all_atom") and recompute=True to run with features involving coordinates 
# of other atoms than P (for instance, angles, dihedrals and lengths)

# Adding representation
gvp_rep = GVPGraphRepresentation()
ta.add_representation(gvp_rep)

# Splitting dataset
print("Splitting Dataset")
ta.get_split_loaders(recompute=False)

info = ta.describe()

# Training model
# Either by hand:
# for epoch in range(100):
#     for batch in ta.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Get the dimensions of the (scalar and vector, node and edge) features
node_s, node_v = ta.dataset[0]['gvp_graph'].h_V
edge_s, edge_v = ta.dataset[0]['gvp_graph'].h_E

# Or using a wrapper class
model = GVPModel(
    num_classes=ta.metadata["num_classes"],
    graph_level=False,
    node_in_dim=(node_s.shape[1],node_v.shape[1]),
    node_h_dim=(node_s.shape[1],node_v.shape[1]),
    edge_in_dim=(edge_s.shape[1],edge_v.shape[1]),
    edge_h_dim=(edge_s.shape[1],edge_v.shape[1]),
)
model.configure_training(learning_rate=0.001)

# Set class weights for binary classification based on actual distribution
if model.num_classes == 2:
    if "0" in ta.metadata["class_distribution"]:
        neg_count = float(ta.metadata["class_distribution"]["0"])
        pos_count = float(ta.metadata["class_distribution"]["1"])
    else:
        neg_count = float(ta.metadata["class_distribution"][0])
        pos_count = float(ta.metadata["class_distribution"][1])
    pos_weight = torch.tensor(np.sqrt(neg_count / pos_count)).to(model.device)
    model.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Training the model
model.train_model(ta, epochs=10, print_all_epochs=True)

# Evaluating the model
test_metrics = model.evaluate(ta, split="test")
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
