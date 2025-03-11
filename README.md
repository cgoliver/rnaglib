<p align="center">
<img src="https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/rgl.png#gh-light-mode-only" width="30%">
</p>

# RNA Geometric Library (`rnaglib`)


<div align="center">

![build](https://img.shields.io/github/actions/workflow/status/cgoliver/rnaglib/build.yml)
[![pypi](https://img.shields.io/pypi/v/rnaglib?)](https://pypi.org/project/rnaglib/)
[![docs](https://img.shields.io/readthedocs/rnaglib)](https://rnaglib.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/cgoliver/rnaglib/graph/badge.svg?token=AOQIF59SFT)](https://codecov.io/gh/cgoliver/rnaglib)
</div>

`RNAglib` is a Python package for studying RNA 2.5D and 3D structures. Functionality includes automated data loading,
analysis,
visualization, ML model building and benchmarking.

We host RNAs annotated with molecule, base pair, and nucleotide level attributes. These include, but are not limited to:

* Secondary structure
* 3D coordinates
* Protein binding
* Small molecule binding
* Chemical modifications
* Leontis-westhof base pair geometry classification

![Example graph](https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/rgl_fig.png)

## Installation

The package can be cloned and the source code used directly. We also deploy it as a pip package and recommend using this
install in conda environments.

If one wants to use GPU support, one should install [Pytorch](https://pytorch.org/get-started/locally/)
and [DGL](https://www.dgl.ai/pages/start.html) with the appropriate options.
Otherwise, you can just skip this step and the pip installs of Pytorch and DGL will be used.

Then, one just needs to run :

```
pip install rnaglib
```

Then one can start using the packages functionalities by importing them in one's python script.

**Optional Dependencies**

We use `fr3d-python` to [build 2.5D annotations from PDBs and mmCIFs](https://rnaglib.readthedocs.io/en/latest/rnaglib.prepare_data.html#rnaglib.prepare_data.fr3d_to_graph). This has to be installed manually with:

```
pip install git+https://github.com/cgoliver/fr3d-python.git
```

To load graphs and train with dgl:

```
pip install dgl
```

To load graphs and train with pytorch geometric:

```
pip install torch_geometric
```

Advanced data splitting of datasets (i.e. by sequence or structure-based similarity) depends on executables [cd-hit](https://sites.google.com/view/cd-hit) and [RNAalign](https://zhanggroup.org/RNA-align/download.html). You may install these yourself or use the convenience script provided in this repo as follows, though we do not guarantee it will work on any system and has only bee tested on linux:

```
chmod u+x install_dependencies.sh
./install_dependencies.sh /my/path/to/executables
```


## Setup: updating RNA structure database

Run the `rnaglib_download` script to fetch the database of annotated structures. By default it will download to `~/.rnaglib`.
Use the `--help` flag for more options.

```
$ rnaglib_download
```

In addition to analysing RNA data, RNAglib also distributes available parsed RNA structures.
Databases of annotated structures can be downloaded directly from [Zenodo](https://zenodo.org/records/14625192).
They can also be obtained through the provided command line utility `$ rnaglib_download -r all|nr`.

| Version | Date     | Total RNAs | Total Non-Redundant | Non-redundant version | `rnaglib` commit |
---------|----------|------------|---------------------|-----------------------|------------------|
 2.0.2   | 25-02-25 | 8441       | 2921                | 3.375                 | ac303c7          |
 2.0.0   | 12-01-25 | 8305       | 2877                | 3.369                 | 33a9e989         |
 1.0.0   | 15-02-23 | 5759       | 1176                | 3.269                 | 5446ae2c         |
 0.0.0   | 20-07-21 | 3739       | 899                 | 3.186                 | eb25dabd         |


To speed up some functions we also build an efficient index of the data. This is needed for the `rnaglib.tasks` functionality.

```
$ rnaglib_index
```

## What can you do with `rnaglib`?

### Benchmark ML models on RNA 3D structures (**new**)

Datasets of RNA 3D structures ready-to-use for machine learning model benchmarking in various biologically relevant
tasks.

* One line to fetch annotated dataset and splits loading
* Encode structures as graphs, voxels and point clouds

All tasks take as input an RNA 3D structure and have as output either residue or whole RNA-level properties.

We currently support the following 6 prediction tasks and the ability to
create [new tasks](https://rnaglib.readthedocs.io/en/latest/tasks.html#tutorial-2-creating-a-new-task):

* [Residue-Level RNA-Protein Binding](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RBP_Node)
* [Chemical Modification](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RBP_Node)
* [Inverse Folding](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_IF)
* [Small Molecule Ligand Classification](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_Ligand)
* [Small Molecule Binding Site Detection](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_Site)
* [RNA Virtual Screening](https://github.com/cgoliver/rnaglib/tree/master/src/rnaglib/tasks/RNA_VS)

See [docs](https://rnaglib.readthedocs.io/en/latest/tasks.html) for more info on usage.

Each link for the tasks above takes you to a `demo.py` file with example usage for each task.


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


## Fetch and browse annotated RNA 3D structures

Current release contains annotations generated by x3dna-dssr as well as some additional ones that we added for all
available PDBs at the time of release.

Each RNA is stored as a networkx graph where nodes are residues and edges are backbone and base pairing edges.
The networkx graph object has graph-level, node-level and edge-level attributes.
[Here](https://rnaglib.readthedocs.io/en/latest/rna_ref.html) is a reference for all the annotations currently
available.

```python

>>> from rnaglib.data_loading import rna_from_pdbid
>>> rna_dict = graph_from_pdbid('1fmn') # fetch from local database or RCSB if not found
>>> rna_dict['rna'].graph
{'name': '1fmn', 'pdbid': '1fmn', 'ligands': [{'id': ('H_FMN', 36, ' '), 'name': 'FMN', 'smiles': 'Cc1cc2c(cc1C)N(C3=NC(=O)NC(=O)C3=N2)CC(C(C(COP(=O)(O)O)O)O)O', 'rna_neighs': ['1fmn.A.10', '1fmn.A.11', '1fmn.A.12', '1fmn.A.13', '1fmn.A.24', '1fmn.A.25', '1fmn.A.26', '1fmn.A.27', '1fmn.A.28', '1fmn.A.7', '1fmn.A.8', '1fmn.A.9']}], 'ions': []}
```
## Annotate your own structures

You can extract Leontis-Westhof interactions and convert 3D structures to 2.5D graphs.
We wrap a fork of [fr3d-python](https://github.com/cgoliver/fr3d-python) to support this functionality.

```python
from rnaglib.prepare_data import fr3d_to_graph

G = fr3d_to_graph("../data/structures/1fmn.cif")
```

Warning: this method currently does not support non-standard residues. Support coming soon. Up to version 1.0.0 of the
RNA database were created using x3dna-dssr which do contain non-standard residues.

## Quick visualization of 2.5D graphs

We customize networkx graph drawing functionalities to give some convenient visualization of 2.5D base pairing networks.
This is not a dedicated visualization tool, it is only intended for quick debugging. We point you
to [VARNA]()https://varna.lisn.upsaclay.fr/ or [RNAscape](https://academic.oup.com/nar/article/52/W1/W354/7648766) for a
full-featured visualizer.

```python
from rnaglib.drawing import rna_draw

rna_draw(G, show=True, layout="spring")
```

![](https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/g.png)

## 2.5D graph comparison and alignment

When dealing with 3D structures as 2.5D graphs we support graph-level comparison through the graph edit distance.

```python
from rnaglib.ged import graph_edit_distance
from rnaglib.utils import graph_from_pdbid

G = graph_from_pdbid("4nlf")
print(graph_edit_distance(G, G))  # 0.0

```

## Testing library functionality

Go to root of the rnaglib directory from a git clone and run pytest.

```
pip install pytest
pytest tests/
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

## Projects using `rnaglib`

If you use rnaglib in one of your projects, please cite and feel free to make a pull request so we can list your project
here.

* [RNAMigos2](https://github.com/cgoliver/RNAmigos2)
* [Structure-and Function-Aware Substitution Matrices](https://github.com/BorgwardtLab/GraphMatchingSubstitutionMatrices)
* [MultiModRLBP: A Deep Learning Approach for RNA-Small Molecule Ligand Binding Site Prediction using Multi-modal features](https://github.com/lennylv/MultiModRLBP)
* [VeRNAl](https://github.com/cgoliver/vernal)
* [RNAMigos](https://github.com/cgoliver/RNAmigos)

## Resources

* [Documentation](https://rnaglib.readthedocs.io/en/latest/?badge=latest)
* [Twitter](https://twitter.com/rnaglib)
* Contact: `rnaglib@cs.mcgill.ca`

## References

1. Leontis, N. B., & Zirbel, C. L. (2012). Nonredundant 3D Structure Datasets for RNA Knowledge Extraction and
   Benchmarking. In RNA 3D Structure Analysis and Prediction N. Leontis & E. Westhof (Eds.), (Vol. 27, pp. 281–298).
   Springer Berlin Heidelberg. doi:10.1007/978-3-642-25740-7\_13

