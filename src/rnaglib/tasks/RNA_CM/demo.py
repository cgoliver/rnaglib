from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import RGCN_node

print("Generating task")
ta = ChemicalModification("RNA-CM")

# Add representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders()

# Printing statistics
info = ta.describe()
num_node_features = info["num_node_features"]
num_classes = info["num_classes"]
num_unique_edge_attrs = info["num_edge_attributes"]

# Train model
model = RGCN_node(num_node_features, num_classes, num_unique_edge_attrs)
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
