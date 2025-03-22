# RNA-Site

In this directory you can find the implementation of the `RNA-Site` task.
Given an RNA structure, it predicts whether an individual node/residue is part of a binding site or not.
Here, we implement this task on the entire `rnaglib` database as well as using a dataset and splits provided in the
following paper:


> Hong Su, Zhenling Peng, and Jianyi Yang. Recognition of small molecule–rna binding sites using
> rna sequence and structure. Bioinformatics, 37(1):36–42, 2021. <https://doi.org/10.1093/bioinformatics/btaa1092>

## Project Structure

This repository contains four files:

1. `demo.py`
2. `binding_site.py`
3. `benchmark_demo.py`

### demo.py

This file contains a demonstration of how to use the `RNA-Site` task, in the implementation of `binding_site.py`
to train a simple model.
It trains an RGCN and outputs some performance metrics. It can be easily expanded for further model development.
It uses an RGCN implemented in the `learning` directory of this repo.

### binding_site.py

This is the task definitions of the `BindingSite` task using `rnaglib`'s task API. It includes:

- Loading and preprocessing of the part of `rnaglib` database that has known ligands
- Defines choice of splitting strategy and through inheritance the model evaluation methods.

A second task named `BenchmarkBindingSite` is being implemented. It has the same input and output variables as `BindingSite`
and also uses the RGCN implemented in the `learning` directory of this repo but uses the data and splits from Su et al. (2021)
to ensure fair benchmarking with their results.

### benchmark_demo.py

This file contains a demonstration of how to use the `RNA-Site` task, in the implementation
of `benchmark_binding_site.py`, using splits from Su et al. (2021) to train a simple model. It trains an RGCN and
outputs some performance metrics. It can be easily expanded for further model development. It also uses an RGCN
implemented in the `learning` directory of this repo.

## Usage

To train and evaluate the model, simply run: `python demo.py`