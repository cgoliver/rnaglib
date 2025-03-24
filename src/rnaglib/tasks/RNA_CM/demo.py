from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation

ta = ChemicalModification(
    root="RNA_CM",
    recompute=False,
    debug=False,
)

# Adding representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

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

# Or using a wrapper class
model = PygModel(
    ta.metadata["num_node_features"],
    ta.metadata["num_classes"],
    graph_level=False,
)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=10)

# Evaluating the model
test_metrics = model.evaluate(ta, split="test")
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
