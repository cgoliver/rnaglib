from rnaglib.tasks import BenchmarkBindingSite
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

# Creating task
ta = BenchmarkBindingSite(root="RNA_Site_Bench", recompute=True)

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
model = PygModel(ta.metadata["num_node_features"], ta.metadata["num_classes"], graph_level=False)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=1)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
