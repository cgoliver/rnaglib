from rnaglib.dataset_transforms import RandomSplitter

# `rnaglib`'s Task module

The new tasks module allows the use and creation of a variety of machine learning tasks on RNA structure. 
Those tasks include:

* One line to fetch annotated dataset and splits loading
* Compute custom features and encode structures as graphs, voxels or point clouds
* Principled way to compute metrics

Each task definition is found in its named directory: 
* [RNA_CM](./RNA_CM)
* [RNA_GO](./RNA_IF)
* [RNA_IF](./RNA_IF)
* [RNA_Ligands](./RNA_Ligands)
* [RNA_Prot](./RNA_Prot)
* [RNA_Site](./RNA_IF)
* [RNA_VS](./RNA_IF)

Each contains a brief description as well as a `demo.py` script which trains a simple model on the task and outputs 
relevant metrics. More tutorials can be found in the online documentation ([**rnaglib.org**](https://rnaglib.org)).

Below, we provide two small tutorials :

1. [Using an existing task for model evaluation](#using-an-existing-task-for-model-evaluation)
2. [Creating a new task](#creating-a-new-task) 

## Using an existing task for model evaluation

### What is the fastest way to get training?

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

# Load task, representation, and get loaders
task = ChemicalModification(root="my_root")
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

### How do I use custom features for training?

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
task.add_feature(tr)
task.add_representation(pyg_rep)

model = PygModel.from_task(task, num_node_features=644)

for batch in train_loader:
    batch = batch['graph'].to(model.device)
    output = model(batch)

test_metrics = model.evaluate(task, split='test')

```

Since the RNAFM embeddings have 640 dimensions, and the nucleotide tensor is 4
dimensional, we initialize the model with 644 dimensions.


## Creating a new task
The task module provides the logic to develop new tasks from scratch with little effort. 

1.) Start with the task type you would like to implement. In this case, we will build a residue classification task and can inherit from that class type. You can inherit directly from the `Task` class if preferred.

```python
class TutorialTask(ResidueClassificationTask):
```
2.) Specify your input and target variables, which in the case of a residue classification task should be node attributes.

```python
input_var = "nt_code" # if sequence information should be used. 
target_var = 'binding_ion'  # for example
```
3.) Next, you can define a splitter you want to use for your task. This can always be overwritten at instantiation. You can chose any available splitter object, write your own splitter object and call it here, or simply have the default_splitter return three lists of indices.

```python
    def default_splitter(self):
        return RandomSplitter()
```


4.) In the simplest case, you just need to include the code to create the dataset and your new task is ready to go.

```python
def process(self):
    dataset = RNADataset()
    return dataset
```

5.) However, you may want your dataset to contain only a selection of RNA structures or you may want to use a node label not available in the base dataset or you may want to include only certain nucleotides with specific properties. In this case `rna_filter` andor `annotator` andor `nt_filter`  can be passed to `RNADataset`.

For example:
- `rna_filter=lambda x: x.graph['pdbid'][0] in rnas_keep` where rnas_keep is a list of pdbids that you want your dataset to contain.
- `annotator=self._annotator`

A simple annotator could add a dummy variable to each node:

```python

from networkx import set_node_attributes

   def _annotator(self, x):
        dummy = {
            node: 1
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, dummy, 'dummy')
        return x
```

6.) Here an example of a complete task definition (including init method). You are done now and ready to go!
```
#import statements left out for brevity
class TutorialTask(ResidueClassificationTask):
    target_var = 'binding_ion' 
    input_var = "nt_code" # 
    name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def default_splitter(self):
        return RandomSplitter()

   def _annotator(self, x):
        dummy = {
            node: 1
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, dummy, 'dummy')
        return x

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            if "node_" + self.target_var in graph_attrs:
                rnas_keep.append(graph.split(".")[0])

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep,
                             annotator=self._annotator
                             )
        return dataset
```

7.) Don't forget to add your task name to the `__init__.py` file. (And if you feel like it, submit a pull request ;) )
