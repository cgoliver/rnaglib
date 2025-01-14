from rnaglib.transforms import GraphRepresentation
from rnaglib.tasks import ProteinBindingSite
from rnaglib.learning.task_models import PygModel

ta = ProteinBindingSite("RNA_Prot", recompute=True, debug=False, size_thresholds=[5,500])

# Add representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(recompute=True)

# Train model
# Either by hand:
# for epoch in range(100):
#     for batch in task.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
model = PygModel(ta.metadata["description"]["num_node_features"], ta.metadata["description"]["num_classes"], graph_level=False)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=1)

# Final evaluation
test_metrics = model.evaluate(ta, split="test")
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")

