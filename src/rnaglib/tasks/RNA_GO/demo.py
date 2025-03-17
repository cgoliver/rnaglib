from rnaglib.dataset_transforms import RandomSplitter
from rnaglib.tasks import RNAGo
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

ta = RNAGo(
    root="RNA_GO_random",
    splitter=RandomSplitter(),
    recompute=False,
    debug=False)

ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.get_split_loaders(batch_size=1)

# Train model
model = PygModel(ta.metadata["num_node_features"],
    num_classes=len(ta.metadata["label_mapping"]),
    graph_level=True,
    multi_label=True)
model.configure_training(learning_rate=0.0001)
model.train_model(ta, epochs=20)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
