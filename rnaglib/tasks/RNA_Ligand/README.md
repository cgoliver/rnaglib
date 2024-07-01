# RNA-Ligand

In this directory you can find the implementation of the RNA-Ligand task. It provides a dataset of RNA pockets and their respective ligands and allows for ligand prediction. Here, we base our task on the Graph Matching Substitution Matrices introduced in:

> Pellizzoni, P., Oliver, C., Borgwardt, K. (2024). Structure- and Function-Aware Substitution Matrices via Learnable Graph Matching. In: Ma, J. (eds) Research in Computational Molecular Biology. RECOMB 2024. Lecture Notes in Computer Science, vol 14758. Springer, Cham. https://doi.org/10.1007/978-1-0716-3989-4_18

## Project Structure

This repository contains two files:

1. `demo.py`
2. `ligand_identity.py`

### demo.py

This file contains a demonstration of how to use the `RNA-Ligand` task to train a simple model. It trains an RCGN and outputs some performance metrics. It can be easily expanded for further model development. It uses the RCGN implemented in the `learning` directory of this repo.

### ligand_identity.py

This is the task definition using `rnaglib`'s task API. It includes:
- Loading and preprocessing of the dataset based on GMSM
- Adds relevant attributes to the classification tasks
- Defines choice of splitting strategy and through inheritance the model evaluation methods.
- Logging the training progress

## Usage

To train and evaluate the model, simply run: `python demo.py` or `python demo.py --frompickle` if you want to use a precomptued task, saving some execution time.