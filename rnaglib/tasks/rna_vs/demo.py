import os
import sys

if __name__ == "__main__":
    sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..")] + sys.path

from rnaglib.tasks.rna_vs.task import VSTask
from rnaglib.representations.graph import GraphRepresentation

# Create a task
root = "../../data/tasks/rna_vs"
framework = 'pyg'
ef_task = VSTask(root)

# Build corresponding datasets and dataloader
representations = [GraphRepresentation(framework=framework)]
rna_dataset_args = {'representations': representations, 'nt_features': 'nt_code'}
rna_loader_args = {'batch_size': 2}
train_dataloader, val_dataloader, test_dataloader = ef_task.get_split_loaders(dataset_kwargs=rna_dataset_args,
                                                                              dataloader_kwargs=rna_loader_args)

# Check both models work well
for i, elt in enumerate(train_dataloader):
    # print(elt)
    a = 1
    # if i > 3:
    #     break
    if not i % 50:
        print(i, len(train_dataloader))

for i, elt in enumerate(test_dataloader):
    # print(elt)
    a = 1
    # if i > 3:
    #     break
    if not i % 10:
        print(i, len(train_dataloader))

# train.loss=bce
# train.num_epochs=1000
# train.early_stop=100

