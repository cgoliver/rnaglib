"""Demo for training a simple model using an rnaglib task"""

from rnaglib.tasks import BindingSiteDetection
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import RGCN_node

# Creating task
ta = BindingSiteDetection("RNA-Site", in_memory=False)

# Add representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(recompute=False)

# Computing and printing statistics
info = ta.describe()

# Train model
# Either by hand:
# for epoch in range(100):
#     for batch in task.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
model = RGCN_node(info["num_node_features"], info["num_classes"], info["num_edge_attributes"])
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
