# RBP_Graph

In this directory you can find the implementation of the `RBP-Graph` task. It provides a dataset of RNAs exlcuding ribosomal RNAs and allows for the binary prediction of protein binding at the graph/RNA level. Since all RNAs in our dataset bind to proteins, further refinement of the dataset is necessary by the user to train a reasonable model, for example by providing a custom dataset, custom binding sites, or subetting the existing RNAs into pockets for pocket level classification.

## Project Structure

This repository contains two files:

1. `demo.py`
2. `protein_binding.py`

### demo.py

This file contains a demonstration of how to use the `RBP-Graph` task to train a simple model. It trains an RGCN and outputs some performance metrics. It can be easily expanded for further model development. It uses an RGCN implemented in the `learning` directory of this repo.

### protein_binding.py

This is the task definition using `rnaglib`'s task API. It includes:
- Loading and preprocessing of the entire `rnaglib` datase excluding ribosomal RNA.
- Defines choice of splitting strategy and through inheritance the model evaluation methods.

WIP: Please note that furhter processing of the dataset is necessary for model development.

## Usage

To train and evaluate the model, simply run: `python demo.py` or `python demo.py --frompickle` if you want to use a precomptued task, saving some execution time.