# RNA-IF

In this directory you can find the implementation of the `RNA-IF` task. Given an RNA structure it predicts nucleotide identity on the node level. Thereby, it allows model development for the inverse folding problem (structure to sequence). Here, we implement this task on the entire `rnaglib` database as well as using a dataset and splits provided in the following paper:


> Joshi, Chaitanya K., Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon V. Mathis, Alex Morehead, Rishabh Anand, and Pietro Liò. "gRNAde: Geometric Deep Learning for 3D RNA inverse design." bioRxiv (2024) <https://doi.org/10.1101/2024.03.31.587283>

The accuracy score of this task is equivalent to the sequence recovery metric that is used in the above paper and the inverse folding community.

## Project Structure

This repository contains four files:

1. `demo.py`
2. `inverse_folding.py`
3. `benchmark_demo.py`
4. `gRNAde.py`

### demo.py

This file contains a demonstration of how to use the `RNA-IF` task, in the implementation of `inverse_folding.py` to train a simple model. It trains a multiclass RGCN and outputs some performance metrics. It can be easily expanded for further model development. It uses an RGCN implemented in the `learning` directory of this repo.

### inverse_folding.py

This is the task definition using `rnaglib`'s task API. It includes:
- Loading and preprocessing of the entire `rnaglib` database.
- Creation of a dummy node attribute for node classification.
- Defines choice of splitting strategy and the model evaluation method.

### benchmark_demo.py

This file contains a demonstration of how to use the `RNA-Site` task, in the implementation of `benchmark_binding_site.py`, using splits from Joshi et al. (2024) to train a simple model. It trains an RGCN and outputs some performance metrics. It can be easily expanded for further model development. It also uses an RGCN implemented in the `learning` directory of this repo. This script takes a substantial amount of time to run.

### benchmark_binding_site.py

This is the task definition using `rnaglib`'s task API. It includes:
- Loading and preprocessing of the subset of `rnaglib` dataset that is used in Joshi et al. (2021).
- Creation of a dummy node attribute for node classification.
- Defines the splitting strategy used in Joshi et al. (2021) and the model evaluation methods.

## Usage

To train and evaluate the model, simply run: `python demo.py` or `python benchmark_demo.py` and use the `--frompickle` from if you want to use a precomptued task, saving some execution time.