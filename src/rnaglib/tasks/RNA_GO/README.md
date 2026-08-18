# RNA-GO

In this directory you can find the implementation of the `RNA-GO` task, a close equivalent of the GO-term
prediction task for proteins introduced by DeepFRI and evaluated as in GearNet.

It provides a dataset of RNA chains labeled with Gene Ontology (GO) terms, one task per GO aspect
(`ontology="molecular_function"`, `"biological_process"`, or `"cellular_component"`). Annotations come from
Rfam's own family-to-GO curation (`rfam2go`), joined onto PDB chains via `Rfam.pdb`, then propagated up the GO
DAG under the true-path rule so that labels reflect shared function across families rather than acting as a
proxy for Rfam family identity. GO terms are then frequency-filtered (`min_count`/`max_frequency`, following
DeepFRI's `>50 non-redundant chains` cutoff, scaled down to the PDB's smaller RNA coverage) and de-duplicated
(`corr_threshold`) -- see the `RNAGo` class docstring in `rna_go.py` for the full rationale.

Please note that some RNA have more than one annotation. Evaluation follows DeepFRI/GearNet: `Fmax`
(protein/RNA-centric maximum F-score) is the default metric, alongside a pair-centric micro-averaged AUPR
(`auprc_pair`).

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