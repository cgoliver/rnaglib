import os
import pandas as pd
from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

# Hyperparameters to tune
batch_size = 8

# Creating task 
ta = LigandIdentification('RNA_Ligand', recompute=True, use_balanced_sampler=True)

# Splitting dataset
print("Splitting Dataset")
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
ta.set_loaders(batch_size=batch_size, recompute=False)

# Train model
# Either by hand:
# for epoch in range(100):
#     for batch in task.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
model = PygModel(ta.metadata["num_node_features"], ta.metadata["num_classes"], graph_level=True, hidden_channels=128, num_layers=4)
model.configure_training(learning_rate=1e-5)
model.train_model(ta, epochs=10)

# Final evaluation
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
