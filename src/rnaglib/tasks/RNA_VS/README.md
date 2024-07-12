# RNA-IF

In this directory you can find the implementation of the `RNA-VS` task.
Given the structure of an RNA binding-site and of a small molecule, we train a model to predict a compatibility score.
This model can then be used as a virtual-screening tool, as introduced in:

> RNAmigos2: Fast and accurate structure-based RNA virtual screening with semi-supervised graph learning and large-scale
> docking data. Juan G. Carvajal-Patiño, Vincent Mallet, David Becerra, Luis Fernando Niño Vasquez, Carlos Oliver,
> Jérôme Waldispühl bioRxiv (2024) <https://doi.org/10.1101/2023.11.23.568394>

The metric used is the AuROC, which approximately equates the expected rank of active compounds of a binding site
among decoys.

In this task, binding sites were clustered into groups of homologous binding sites, usingRMscores (a structural
similarity method).
The groups were then split following a strict structural split introduced in the paper.

This task implementation deviates a bit from the general Task API, because it involves loaders returning more than an
RNA, and distinct dataloaders for training (returning pairs of RNA, compound) and testing (pairs of RNA, list of
compounds).

## Project Structure

This repository contains four files:

1. `demo.py`
2. `build_data.py`
3. `data.py`
4. `evaluate.py`
5. `ligands.py`
6. `model.py`
7. `task.py`

### demo.py

This file contains a demonstration of how to use the `RNA-VS` task, to train a simple model.
It trains a small RGCN encoder and outputs some performance metrics.
It can be easily expanded for further model development.

### build_data.py

This script encapsulates the data creation:

- Extracting the pockets and annotating their inner part with an 'in_pocket' feature.
- Computing all ligand representations as graphs and saving them in a pickle object

### data.py

This file contains the data loading logic, both for training and testing.

Since the binding pockets are clustered into groups, a single binding site is associated to several actives and decoys.
At train time, we choose to use a positive or negative pair with a balanced probability.
Then we sample an active (resp. inactive) from the corresponding group, and train on the resulting pair.

At test time, we use all actives and inactives associated to a binding site and rank them with our method.

### evaluate.py

Just a simple script to compute the auroc metric over a dataloader of pairs of (RNA binding site, list of compounds)

## ligands.py

Simple script containing the logic to encode a ligand as a graph. 
Right now, this only supports a DGL representation.
Support for other frameworks is on the todo, but feel free to PR.

## model.py

Script defining the model architecture for this task. 
Note that the final model should implement a `predict_ligands(pocket, ligands)` method (to rank compounds).

## task.py

The script combining all these methods together. 
It handles building the data and follows the general Task API: 
- task.get_split_loaders(dataset_kwargs=None, dataloader_kwargs=None)
- task.evaluate(model)

