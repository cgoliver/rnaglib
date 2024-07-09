
<p align="center">
<img src="https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/rgl.png#gh-light-mode-only" width="30%">
</p>


# RNA Geometric Library (`rnaglib`)
[![Documentation Status](https://readthedocs.org/projects/rnaglib/badge/?version=latest)](https://rnaglib.readthedocs.io/en/latest/?badge=latest)

`RNAglib` is a Python package for studying RNA 2.5D and 3D structures. Functionality includes automated data loading, analysis,
visualization, ML model building and benchmarking.

We host RNAs annotated with molecule, base pair, and nucleotide level attributes. These include, but are not limited to:

* Secondary structure
* 3D coordinates
* Protein binding 
* Small molecule binding 
* Chemical modifications 
* Leontis-westhof base pair geometry classification


![Example graph](https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/rgl_fig.png)


## ❗**New Feature**: Full Support for ML Tasks!

We now support fully implemented prediction tasks. With the `rnaglib.tasks` subpackage you can load annotated RNA datasets with train/val/test splits for various biologically relevant prediction tasks.

All tasks take as input an RNA 3D structure and have as output either residue or whole RNA-level properties.

We currently support the following 6 prediction tasks and the ability to create [new tasks](https://rnaglib.readthedocs.io/en/latest/tasks.html#tutorial-2-creating-a-new-task):


### [Reidue-Level RNA-Protein Binding](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RBP_Node)

**Input:** A full RNA 3D structure

**Output:** Binary variable at each residue representing protein-binding likelihood

### [Chemical Modification](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RBP_Node)

**Input:** A full RNA 3D structure

**Output:** Binary variable at each residue with likelihood of covalent modification.

### [Inverse Folding](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_IF)


**Input:** A full RNA 3D structure

**Output:** The nulceotide identity at each position for the native structure.

### [Small Molecule Ligand Classification](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_Ligand)


**Input:** Small molecule binding site

**Output:** Multi-class variable corresponding to the chemical family of the native ligand.

### [Small Molecule Binding Site Detection](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_Site)

**Input:** A full RNA 3D structure

**Output:** Binary variable at each residue corresponding to the likelihood of belonging to a binding pocket.

### [RNA Virtual Screening](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_VS)

**Input:** Small molecule binding site + list of chemical compounds.

**Output:** Sort the list of chemical compounds by likelihood of binding to the given site.


See [docs](https://rnaglib.readthedocs.io/en/latest/tasks.html) for more info on usage.


Each link for the tasks above takes you to a `demo.py` file with example usage for each task.

Here is a snippet for a full data loading and model training of the **RNA Binding Site Detection** task. All tasks follow a similar pattern.

```python

from rnaglib.tasks import BindingSiteDetection
from rnaglib.representations import GraphRepresentation

from rnaglib.tasks import BindingSiteDetection
from rnaglib.representations import GraphRepresentation
from rnaglib.data_loading import Collater
from rnaglib.learning.task_models import RGCN_node

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
import argparse
from pathlib import Path

# Load the task data and annotations
ta = BindingSiteDetection('RNA-Site')

# Select a data representation and framework (see docs for support of other data modalities and deep learning frameworks)

ta.dataset.add_representation(GraphRepresentation(framework = 'pyg'))

# Access the predefined splits

train_ind, val_ind, test_ind = ta.split()
train_set = ta.dataset.subset(train_ind)
val_set = ta.dataset.subset(val_ind)
test_set = ta.dataset.subset(test_ind)


# Create loaders

print('Creating loaders')
collater = Collater(train_set)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collater)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collater)
test_loader = DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=collater)

def count_unique_edge_attrs(train_loader):
    unique_edge_attrs = set()
    for batch in train_loader.dataset:
        unique_edge_attrs.update(batch['graph'].edge_attr.tolist())
    return len(unique_edge_attrs)

num_unique_edge_attrs = count_unique_edge_attrs(train_loader)
num_node_features = train_set[0]['graph'].x.shape[1]
num_classes = 2

print(f"# node features {num_node_features}, # classes {num_classes}, # edge attributes {num_unique_edge_attrs}")

# Define model
learning_rate = 0.0001
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCN_node(num_node_features, num_classes, num_unique_edge_attrs)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Evaluate function

def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    for batch in loader:
        graph = batch['graph']
        graph = graph.to(device)
        out = model(graph)
        loss = criterion(out, torch.flatten(graph.y).long())
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(graph.y.tolist()) 
    
    avg_loss = total_loss / len(loader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    return accuracy, f1, auc, avg_loss, mcc


# Training

def train():
    model.train()
    for batch in train_loader:
        graph = batch['graph']
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out, torch.flatten(graph.y).long())
        loss.backward()
        optimizer.step()

for epoch in range(epochs):
    train()
    train_acc, train_f1, train_auc, train_loss, train_mcc = evaluate(train_loader) 
    val_acc, val_f1, val_auc, val_loss, val_mcc = evaluate(val_loader)  
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, TrainAcc {train_acc:.4f} Val Acc: {val_acc:.4f}")

# Evaluate on test set

test_accuracy, test_f1, test_auc, test_loss, test_mcc = ta.evaluate(model, test_loader, criterion, device)  

print(f'Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}')

```



## Cite

```
@article{mallet2022rnaglib,
  title={RNAglib: a python package for RNA 2.5 D graphs},
  author={Mallet, Vincent and Oliver, Carlos and Broadbent, Jonathan and Hamilton, William L and Waldisp{\"u}hl, J{\'e}r{\^o}me},
  journal={Bioinformatics},
  volume={38},
  number={5},
  pages={1458--1459},
  year={2022},
  publisher={Oxford University Press}
}
```

## Data

Data can be downloaded directrly from [Zenodo](https://sandbox.zenodo.org/record/1168342) or through the provided command 
line utility `$ rnaglib_download`.

| Version | Date | Total RNAs | Total Non-Redundant | Non-redundant version | `rnaglib` commit  |
----------|------|------------|---------------------|-----------------------|-------------------|
1.0.0    | 15-02-23 | 5759   | 1176                | 3.269                 |  5446ae2c         |
0.0.0     | 20-07-21 | 3739   | 899                 | 3.186                 |  eb25dabd         |


## Installation

The package can be cloned and the source code used directly. We also deploy it as a pip package and recommend using this
install in conda environments.

If one wants to use GPU support, one should install [Pytorch](https://pytorch.org/get-started/locally/)
and [DGL](https://www.dgl.ai/pages/start.html) with the appropriate options. Otherwise you can just skip this step and
the pip installs of Pytorch and DGL will be used.

Then, one just needs to run :

```
pip install rnaglib
```

Then one can start using the packages functionalities by importing them in one's python script.

### Optional Dependencies

To build 2.5D graphs from scratch locally, for the moment you need to install a fork of `fr3d-python` manually.

```
pip install git+https://github.com/cgoliver/fr3d-python.git
```


## Associated Repositories:

[VeRNAl](https://github.com/cgoliver/vernal)

[RNAMigos](https://github.com/cgoliver/RNAmigos)


## Resources

* [Documentation](https://rnaglib.readthedocs.io/en/latest/?badge=latest)
* [Homepage](https://rnaglib.cs.mcgill.ca/)
* [Twitter](https://twitter.com/rnaglib)
* Contact: `rnaglib@cs.mcgill.ca`

## References

1. Leontis, N. B., & Zirbel, C. L. (2012). Nonredundant 3D Structure Datasets for RNA Knowledge Extraction and
   Benchmarking. In RNA 3D Structure Analysis and Prediction N. Leontis & E. Westhof (Eds.), (Vol. 27, pp. 281–298).
   Springer Berlin Heidelberg. doi:10.1007/978-3-642-25740-7\_13

