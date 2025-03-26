import os
import pandas as pd
from rnaglib.tasks import LigandIdentification
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

# Creating task 
ta = LigandIdentification('RNA_Ligand', recompute=True, use_balanced_sampler=True)

# Adding representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
print("Splitting Dataset")
ta.set_loaders(batch_size=8, recompute=False)

info = ta.describe()

# Training model
# Either by hand:
# for epoch in range(100):
#     for batch in ta.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
model = PygModel(ta.metadata["num_node_features"], ta.metadata["num_classes"], graph_level=True, hidden_channels=128, num_layers=4)
model.configure_training(learning_rate=1e-5)
model.train_model(ta, epochs=10)

# Evaluating model
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
