"""Demo for training a simple model using an rnaglib task"""

from rnaglib.tasks import gRNAde
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import RGCN_node

ta = gRNAde("gRNAde", recompute=False, in_memory=False)

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders()

# Printing statistics
info = ta.describe()

# Train model
model = RGCN_node(info["num_node_features"], info["num_classes"], info["num_edge_attributes"])
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
