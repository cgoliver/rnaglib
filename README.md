
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

