
<p align="center">
<img src="https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/rgl.png#gh-light-mode-only" width="40%">
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

We provide a visualization of what the graphs in this database contain.
A more detailed description of the data is presented in `/RNAGlib/prepare_data/README`.

![Example graph](https://raw.githubusercontent.com/cgoliver/rnaglib/master/images/rgl_fig.png)

## Data stats

| Version | Date | Total RNAs | Total Non-Redundant | Non-redundant version | `rnaglib` commit  |
----------|------|------------|---------------------|-----------------------|-------------------|
0.0.1     | 15-02-23 | 5759   | 1176                | 3.269                 |  5446ae2c         |
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

