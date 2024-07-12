# RBP_Node

In this directory you can find the implementation of the `RBP-Node` task.
This task is analagous to the `RNA-Site` task which allows the prediction of ligand binding sites at the node level,
except that here, RNA-binding protein binding sites are predicted.

## Project Structure

This repository contains two files:

1. `demo.py`
2. `protein_binding_site.py`

### demo.py

This file contains a demonstration of how to use the `RBP-Node` task to train a simple model.
It trains an RGCN and outputs some performance metrics.
It can be easily expanded for further model development.
It uses an RGCN implemented in the `learning` directory of this repo.

### protein_binding_site.py

This is the task definition using `rnaglib`'s task API. It includes:

- Loading and preprocessing of the entire `rnaglib` dataset except ribosomal RNA.
- Defining the choice of splitting strategy and through inheritance the model evaluation methods.

## Usage

To train and evaluate the model, simply run: `python demo.py` or `python demo.py --frompickle` if you want to use a
precomputed task, saving some execution time.