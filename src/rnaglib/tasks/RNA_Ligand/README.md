# RNA-Ligand

In this directory you can find the implementation of the `RNA-Ligand` task.
It provides a dataset of RNA pockets and their respective ligands and allows for ligand prediction.

## Project Structure

This repository contains five files:

1. `demo.py`
2. `ligand_identity.py`
3. `prepare_dataset.py`
4. `prepare_data.py`
5. `cluster_ligands.py`

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

The classification target is `ligand_cluster`: a Tanimoto-similarity cluster id (see `cluster_ligands.py` below),
not a hand-curated chemical family. Pockets whose ligand doesn't fall into one of the clusters in
`data/ligand_to_cluster.json` are dropped, keeping the task closed (no catch-all class).

### prepare_dataset.py

This file contains the definition of the PrepareDataset class which is being called in ligand_identity.py in the data post-processing. Indeed, the filtering of the RNAs aiming at removing redundancy is a bit different for this tasks than for other tasks.

### prepare_data.py

This file contains the instructions to build the 2 JSON files bp_dict.json and ligands_dict.json which respectively indicate the splitting of the RNA residues into binding pockets and the ligand associated with each RNA residue. bp_dict.json and ligands_dict.json are required to run ligand_identity.py. By default, the files bp_dict.json and ligands_dict.json contained in data folder will be used.

### cluster_ligands.py

Builds `data/ligand_to_cluster.json`, the file `ligand_identity.py` reads to assign each pocket its class. This
replaces the earlier scheme (`data/family_map.json`, a hand-curated mapping to 12 ChEBI/ClassyFire families, of
which only 3 had enough independent binding sites to be usable, and even then 88% of the final dataset fell into
one class).

Method: fingerprint every ligand (ECFP4 from `data/ligand_smiles.json`), cluster with complete-linkage hierarchical
clustering on 1-Tanimoto distance (bounds the *maximum* pairwise distance within a cluster — a real chemical
homogeneity guarantee, unlike a connected-components scheme where membership is only chain-connected through
intermediates), then iteratively merge clusters below a minimum size into their nearest neighbor as long as the
merge doesn't push mean intra-cluster similarity below a homogeneity floor. Unmergeable groups are dropped —
including, correctly, the old "amino acid / peptide" family, which turns out to not be a single Tanimoto cluster at
all (it mixed free amino acids, arginine mimetics, and peptide-like antibiotics under one ChEBI label).

The `n_min` cutoff operates on a *proxy*: raw per-ligand occurrence counts in `ligands_dict.json`, not the true
post-redundancy-removal (CD-Hit + US-align) pocket count, which is only known after actually building and
post-processing the task and is too expensive to compute inside a clustering sweep (pairwise structural alignment
over thousands of pockets). Re-tune `n_min`/`cut_dist`/`homogeneity_floor` once real post-redundancy class counts
are available from a full `LigandIdentification(recompute=True)` run, and re-run this script if they need
adjusting. `data/cluster_report.json` records, per cluster, its members, proxy weight, mean intra-cluster
similarity, and which legacy families its members used to carry, to sanity-check the clusters chemically.

This script does not cap or subsample any cluster's size — structural diversity within a class is left entirely to
the existing redundancy-removal / splitting pipeline (CD-Hit, US-align, `ClusterSplitter`).

## Usage

To train and evaluate the model, simply run: `python demo.py` if you want to use a precomputed task, saving some execution time.
