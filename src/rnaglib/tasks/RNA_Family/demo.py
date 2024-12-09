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

# Train model
model = PygModel(num_node_features, num_classes, graph_level=True)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=1)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
