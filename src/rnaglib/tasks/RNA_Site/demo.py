from rnaglib.tasks import BindingSite
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

# Creating task
ta = BindingSite(root="RNA_Site", debug=False, precomputed=True)

# Add representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
train, val, test = ta.get_split_loaders(recompute=False)

print(len(train), len(val), len(test))

info = ta.describe()

# Train model
# Either by hand:
# for epoch in range(100):
#     for batch in task.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
model = PygModel(ta.metadata["num_node_features"], ta.metadata["num_classes"],
                 graph_level=False, device='cpu')
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=1)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
