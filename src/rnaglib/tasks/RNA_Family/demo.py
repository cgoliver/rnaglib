from rnaglib.tasks import RNAFamily
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

ta = RNAFamily(root="RNA-Family", recompute=False, debug=True, filter_by_size=True, filter_by_resolution=True)

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(recompute=True, batch_size=1)

# Train model
model = PygModel(ta.metadata["description"]["num_node_features"],
    num_classes=len(ta.metadata["label_mapping"]),
    graph_level=True,
    multi_label=True)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=1)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
