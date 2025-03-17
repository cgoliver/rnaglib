"""Demo script for running the RNA protein task."""

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ProteinBindingSite
from rnaglib.transforms import GraphRepresentation

ta = ProteinBindingSite("RNA_RBP_struc", recompute=False, debug=False, precomputed=False)

# Add representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(recompute=False)

# Train model
# Either by hand:
# for epoch in range(100):
#     for batch in task.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
model = PygModel(
    ta.metadata["num_node_features"],
    ta.metadata["num_classes"],
    graph_level=False,
)
model.configure_training(learning_rate=0.01)
model.train_model(ta, epochs=10)

# Final evaluation
test_metrics = model.evaluate(ta, split="test")
# Print metrics
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
