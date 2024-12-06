from rnaglib.tasks import RNAFamily

from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

ta = RNAFamily(root="RNA-Family", recompute=False, debug=True)

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(recompute=False, batch_size=1)

# Printing statistics
info = ta.describe(recompute=True)

num_node_features = info["num_node_features"]
num_classes = info["num_classes"]
num_unique_edge_attrs = 20  # info["num_edge_attributes"]
# need to set to 20 (or actual edge type #) if not all edges are present, such as in debugging
# describe needs to be moved to the task type class since it's different for graph level tasks

# Train model
model = PygModel(num_node_features, num_classes, num_unique_edge_attrs, graph_level=True)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=1)

# Final evaluation
test_metrics = ta.evaluate(model, ta.test_dataloader)
print(
    f"Test Loss: {test_metrics['loss']:.4f}, "
    f"Test Accuracy: {test_metrics['accuracy']:.4f}, "
    f"Test F1 Score: {test_metrics['f1']:.4f}, "
    f"Test AUC: {test_metrics['auc']:.4f}, "
    f"Test MCC: {test_metrics['mcc']:.4f}"
)
