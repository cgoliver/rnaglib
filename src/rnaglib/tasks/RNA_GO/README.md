# RNA-GO

In this directory you can find the implementation of the `RNA-GO` task, that is a close equivalent to the Go-terms
task for proteins as introduced by DeepFRI.

It provides a dataset of RNA along some GO annotations.
These annotations were obtained from the files produced by RFAM.
GO annotations with less than 50 or more than 1000 examples were discarded.

Please note that some RNA have more than one annotation.

## Project Structure

This repository contains two files:

1. `demo.py`
2. `rna_go.py`

### demo.py

This file contains a demonstration of how to use the `RNA-GO` task to train a simple model. It trains an RGCN and
outputs some performance metrics. It can be easily expanded for further model development. It uses an RGCN implemented
in the `learning` directory of this repo.

### rna_go.py

This is the task definition using `rnaglib`'s task API. It includes:

- Loading and preprocessing of the entire `rnaglib` dataset.
- Definition of the splitting strategy and - through inheritance - the model evaluation methods.

## Usage

To train and evaluate the model, simply run: `python demo.py`