"""Demo script for running Inverse Folding on the gRNAde dataset."""

from rnaglib.learning.task_models import PygModel
from rnaglib.tasks import gRNAde
from rnaglib.transforms import GraphRepresentation

ta = gRNAde(root="gRNAde", recompute=True, in_memory=False, debug=True)

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(recompute=False)

# Train model
model = PygModel(
    num_node_features=ta.metadata["num_node_features"],
    num_classes=ta.metadata["num_classes"] + 1,
    graph_level=False,
)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=1)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {f'{v:.4f}' if k != 'confusion_matrix' else v}")
