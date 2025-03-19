# ML tasks FAQ

![](https://github.com/cgoliver/rnaglib/blob/2f00342a70f5b7476492cff0779cfae9376b7e99/images/tasksfig.png)

`rnaglib`'s task module provides you with readymade dataset splits for your model evaluation in just a few lines of code.

## What is the fastest way to get training?

Everything you need to train and evaluate a model is built on 3 basic
ingredients:

1. A ``rnaglib.Task`` object with holds all the relevant data, splits and
   functionality.
2. A ``rnaglib.Representation`` object which converts raw RNAs to tensor
   formats.
3. A model of your choosing, though we provide a basic one to get started
   ``rnaglib.learning.PyGmodel``

```python
from rnaglib.tasks import ChemicalModification
from rnaglib.transforms import GraphRepresentation
from rnaglib.learning.task_models import PygModel

# Load task, representation, and get loaders task = ChemicalModification(root="my_root")
model = PygModel.from_task(task)
pyg_rep = GraphRepresentation(framework="pyg")

task.add_representation(pyg_rep)
train_loader, val_loader, test_loader = task.get_split_loaders(batch_size=8)

for batch in train_loader:
    batch = batch['graph'].to(model.device)
    output = model(batch)

test_metrics = model.evaluate(task, split='test')
```

By default, most tasks use the residue type as the only residue-level feature
and if you choose a graph representation, the graph is computed using the
[Leontis Westhof nomenclature](https://nakb.org/basics/basepairs.html). 


## How do I use custom features for training?

Features are handled through the ``rnaglib.Transforms`` class. Each
``Transform`` is a callable which receives an RNA, applies any operation to it
(e.g. adding a piece of data) and returns it.
We provide some Transforms which you can use to add more features to datasets
by simply passing it to the `Task.add_feature()` method.

This is an example of adding [RNAFM](https://arxiv.org/abs/2204.00300) embeddings as features to a dataset.

```python
from rnaglib.tasks import RNAGo
from rnaglib.transforms import RNAFMTransform

# Take out the ingredients
task = RNAGo(root="go")
tr = RNAFMTransform()
pyg_rep = GraphRepresentation(framework="pyg")
model = PygModel.from_task(task, num_node_features=644)

task.add_feature(tr)
task.add_representation(pyg_rep)

for batch in train_loader:
    batch = batch['graph'].to(model.device)
    output = model(batch)

test_metrics = model.evaluate(task, split='test')

```

Since the RNAFM embeddings have 640 dimensions, and the nucleotide tensor is 4
dimensional, we initialize the model with 644 dimensions. 
