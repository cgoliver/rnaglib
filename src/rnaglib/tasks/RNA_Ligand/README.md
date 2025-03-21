# RNA-Ligand

In this directory you can find the implementation of the `RNA-Ligand` task.
It provides a dataset of RNA pockets and their respective ligands and allows for ligand prediction.

## Project Structure

This repository contains four files:

1. `demo.py`
2. `ligand_identity.py`
3. `prepare_dataset.py`
4. `prepare_data.py`

### demo.py

This file contains a demonstration of how to use the `RNA-Ligand` task to train a simple model.
It trains an RGCN and outputs some performance metrics.
It can be easily expanded for further model development.
It uses an RGCN implemented in the `learning` directory of this repo.

### ligand_identity.py

This is the task definition using `rnaglib`'s task API. It includes:

- Loading and preprocessing of the dataset based on GMSM
- Adds relevant attributes to the classification tasks
- Defines choice of splitting strategy and through inheritance the model evaluation methods.

### prepare_dataset.py

This file contains the definition of the PrepareDataset class which is being called in ligand_identity.py in the data post-processing. Indeed, the filtering of the RNAs aiming at removing redundancy is a bit different for this tasks than for other tasks.

### prepare_data.py

This file contains the instructions to build the 2 JSON files bp_dict.json and ligands_dict.json which respectively indicate the splitting of the RNA residues into binding pockets and the ligand associated with each RNA residue. bp_dict.json and ligands_dict.json are required to run ligand_identity.py. By default, the files bp_dict.json and ligands_dict.json contained in data folder will be used.

## Usage

To train and evaluate the model, simply run: `python demo.py` if you want to use a precomputed task, saving some execution time.
