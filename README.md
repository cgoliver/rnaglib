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


## Setup: updating RNA structure database

Run the `rnaglib_download` script to fetch the database of annotated structures. By default it will download to `~/.rnaglib`.
Use the `--help` flag for more options.

```
$ rnaglib_download
```

In addition to analysing RNA data, RNAglib also distributes available parsed RNA structures.
Databases of annotated structures can be downloaded directly from Zenodo, either the non-redundant subset
[link](https://zenodo.org/records/7624873/files/rnaglib-all-1.0.0.tar.gz)
or all rna structures [link](https://zenodo.org/records/7624873/files/rnaglib-nr-1.0.0.tar.gz).
They can also be obtained through the provided command line utility `$ rnaglib_download -r all|nr`.

| Version | Date     | Total RNAs | Total Non-Redundant | Non-redundant version | `rnaglib` commit |
---------|----------|------------|---------------------|-----------------------|------------------|
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

Here is a snippet for a full data loading and model training of the **RNA Binding Site Detection** task. 
All tasks follow a similar pattern.

```python

from rnaglib.tasks import BindingSiteDetection
from rnaglib.transforms import GraphRepresentation

# Load the task data and annotations
ta = BindingSiteDetection('my_root')

# Select a data representation and framework (see docs for support of other data modalities and deep learning frameworks)

ta.dataset.add_representation(GraphRepresentation(framework='pyg'))

# Access the predefined splits

train_ind, val_ind, test_ind = ta.split()
train_set = ta.dataset.subset(train_ind)
val_set = ta.dataset.subset(val_ind)
test_set = ta.dataset.subset(test_ind)

```

All you need after this to implement a full training and evaluation loop is standard ML boilerplate. Have a look at one
of the [demos](https://github.com/cgoliver/rnaglib/blob/master/src/rnaglib/tasks/RNA_Site/demo.py) for a full example.

### Create your own ML tasks (**new**)

The `rnaglib.tasks` subpackage defines an abstract class to quickly implement machine learning tasks.
You can create a task from the databases available through rnaglib as well as custom annotations to open up your
challenge to the rest of the community.
If you implement a task we encourage you to submit a pull request.

To implement a task you must subclass `rnaglib.tasks.Task` and define the following methods:

* `default_splitter()`: a method that takes as input a dataset and returns train, validation, and test indices.
* `build_dataset()`: a method that returns a `rnaglib.RNADataset` object containing the RNAs needed for the task.

See [here](https://github.com/cgoliver/rnaglib/blob/master/src/rnaglib/tasks/RNA_Site/binding_site.py) for an example of
a full custom task implementation.

### Fetch and browse annotated RNA 3D structures

Current release contains annotations generated by x3dna-dssr as well as some additional ones that we added for all
available PDBs at the time of release.

Each RNA is stored as a networkx graph where nodes are residues and edges are backbone and base pairing edges.
The networkx graph object has graph-level, node-level and edge-level attributes.
[Here](https://rnaglib.readthedocs.io/en/latest/rna_ref.html) is a reference for all the annotations currently
available.

```python

from rnaglib.utils import available_pdbids
from rnaglib.utils import graph_from_pdbid

# load a graph containing annotations from a PDBID
rna = graph_from_pdbid('4v9i')

# you can get list of pdbids currently available in RNAglib
pdbids = available_pdbids()

# print(rna.graph)
{'dbn': {'all_chains': {'num_nts': 143, 'num_chars': 144,
                        'bseq': 'GCCCGGAUAGCUCAGUCGGUAGAGCAGGGGAUUGAAAAUCCCCGUGUCCUUGGUUCGAUUCCGAGUCUGGGCAC&CGGAUAGCUCAGUCGGUAGAGCAGGGGAUUGAAAAUCCCCGUGUCCUUGGUUCGAUUCCGAGUCCGGGC',
                        'sstr': '(((((((..((((.....[..)))).(((((.......))))).....(((((..]....))))))))))))..&((((..((((.....[..)))).(((((.......))))).....(.(((..]....))).)))))...',
                        'form': 'AAAAAA...AA...A.......AAA.AAAA.......A.AAA......AAAAA..A....AAAAAAAAAAAA.-&.AA...AA...A.......AAA.AAAA.......A.AAA......AAAAA..A....A...AAAA.A.-'}...,
```

### Annotate your own structures

You can extract Leontis-Westhof interactions and convert 3D structures to 2.5D graphs.
We wrap a fork of [fr3d-python](https://github.com/cgoliver/fr3d-python) to support this functionality.

```python
from rnaglib.prepare_data import fr3d_to_graph

G = fr3d_to_graph("../data/structures/1fmn.cif")
```

Warning: this method currently does not support non-standard residues. Support coming soon. Up to version 1.0.0 of the
RNA database were created using x3dna-dssr which do contain non-standard residues.

### Quick visualization of 2.5D graphs

We customize networkx graph drawing functionalities to give some convenient visualization of 2.5D base pairing networks.
This is not a dedicated visualization tool, it is only intended for quick debugging. We point you
to [VARNA]()https://varna.lisn.upsaclay.fr/ or [RNAscape](https://academic.oup.com/nar/article/52/W1/W354/7648766) for a
full-featured visualizer.

```python
from rnaglib.drawing import rna_draw

rna_draw(G, show=True, layout="spring")
```

![](https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/g.png)

### 2.5D graph comparison and alignment

When dealing with 3D structures as 2.5D graphs we support graph-level comparison through the graph edit distance.

```python
from rnaglib.ged import graph_edit_distance
from rnaglib.utils import graph_from_pdbid

G = graph_from_pdbid("4nlf")
print(graph_edit_distance(G, G))  # 0.0

```

### Testing library functionality

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

