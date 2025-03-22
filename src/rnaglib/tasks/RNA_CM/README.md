# RNA-CM

In this directory you can find the implementation of the `RNA-CM` task. It provides a dataset of RNAs and their chemical
modifications and allows for a node level binary prediction of chemical modification. To our knowledge, this task has
not yet been introduced in the literature.

## Project Structure

This repository contains two files:

1. `demo.py`
2. `chemical_modification.py`

### demo.py

This file contains a demonstration of how to use the `RNA-CM` task to train a simple model. It trains an RGCN and
outputs some performance metrics. It can be easily expanded for further model development. It uses an RGCN implemented
in the `learning` directory of this repo.

### chemical_modification.py

This is the task definition using `rnaglib`'s task API. It includes:

- Loading and preprocessing of the entire `rnaglib` dataset.
- Definition of the splitting strategy and - through inheritance - the model evaluation methods.

## Usage

To train and evaluate the model, simply run: `python demo.py`
