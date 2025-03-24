from rnaglib.tasks import BindingSite
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

# Creating task
ta = BindingSite(root="RNA_Site", debug=False, precomputed=True)

# Adding representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
print("Splitting Dataset")
train, val, test = ta.get_split_loaders(recompute=False, batch_size=8)

info = ta.describe()

# Training model
# Either by hand:
# for epoch in range(100):
#     for batch in ta.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
model = PygModel.from_task(ta, device='cpu', num_layers=4, hidden_channels=256)
model.configure_training(learning_rate=0.001)
model.train_model(ta, epochs=100)

# Evaluating model
test_metrics = model.evaluate(ta)
for k, v in test_metrics.items():
    print(f"Test {k}: {v:.4f}")
