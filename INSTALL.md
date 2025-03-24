
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
