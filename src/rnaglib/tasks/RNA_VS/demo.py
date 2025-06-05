from rnaglib.tasks.RNA_VS.vs_task import VirtualScreening
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel


# Creating task
ta = VirtualScreening(root="RNA-VS", debug=False, precomputed=True)

# Adding representation
ta.dataset.add_representation(GraphRepresentation(framework="pyg"))

# Splitting dataset
print("Splitting Dataset")
train, val, test = ta.get_split_loaders(recompute=False, batch_size=8)
print(len(train.dataset))
print(len(val.dataset))
print(len(test.dataset))

info = ta.describe()

for da in train.dataset:
    print(da)
    break

# TODO: include ligand in return dict using representations


# Training model
# Either by hand:
# for epoch in range(100):
#     for batch in ta.train_dataloader:
#         graph = batch["graph"].to(self.device)
#         ...

# Or using a wrapper class
# model = PygModel.from_task(ta, device='cpu', num_layers=4, hidden_channels=256)
# model.configure_training(learning_rate=0.001)
# model.train_model(ta, epochs=100)
#
# # Evaluating model
# test_metrics = model.evaluate(ta)
# for k, v in test_metrics.items():
#     print(f"Test {k}: {v:.4f}")