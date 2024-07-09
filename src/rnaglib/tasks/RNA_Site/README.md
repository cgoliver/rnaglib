# RNA-Site

In this directory you can find the implementation of the `RNA-Site` task. Given an RNA structure, it predicts whether an individual node/residue is part of a binding site or not. Here, we implement this task on the entire `rnaglib` database as well as using a dataset and splits provided in the following paper:


> Hong Su, Zhenling Peng, and Jianyi Yang. Recognition of small molecule–rna binding sites using
rna sequence and structure. Bioinformatics, 37(1):36–42, 2021. <https://doi.org/10.1093/bioinformatics/btaa1092>

## Project Structure

This repository contains four files:

1. `demo.py`
2. `binding_site.py`
3. `benchmark_demo.py`
4. `benchmark_binding_site.py`

### demo.py

This file contains a demonstration of how to use the `RNA-Site` task, in the implementation of `binding_site.py` to train a simple model. It trains an RGCN and outputs some performance metrics. It can be easily expanded for further model development. It uses an RGCN implemented in the `learning` directory of this repo.

### binding_site.py

This is the task definition using `rnaglib`'s task API. It includes:
- Loading and preprocessing of the part of `rnaglib` database that has known ligands
- Defines choice of splitting strategy and through inheritance the model evaluation methods.

### benchmark_demo.py

This file contains a demonstration of how to use the `RNA-Site` task, in the implementation of `benchmark_binding_site.py`, using splits from Su et al. (2021) to train a simple model. It trains an RGCN and outputs some performance metrics. It can be easily expanded for further model development. It also uses an RGCN implemented in the `learning` directory of this repo.

### benchmark_binding_site.py

This is the task definition using `rnaglib`'s task API. It includes:
- Loading and preprocessing of the subset of `rnaglib` dataset that is used in Su et al. (2021).
- Defines the splitting strategy used in Su et al. (2021) and through inheritance defines the model evaluation methods.

## Usage

To train and evaluate the model, simply run: `python demo.py` or `python benchmark_demo.py` and use the `--frompickle` from if you want to use a precomptued task, saving some execution time.