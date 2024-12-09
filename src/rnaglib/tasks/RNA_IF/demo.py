from rnaglib.tasks import InverseFolding
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

ta = InverseFolding(root="IF", recompute=False, in_memory=False, debug=True)

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(recompute=False)

# Printing statistics
info = ta.describe(recompute=True)

# Train model
model = PygModel(
    num_node_features=info["num_node_features"],
    num_classes=info["num_classes"],
    graph_level=False
)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=10)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {f'{v:.4f}' if k != 'confusion_matrix' else v}")

