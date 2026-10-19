# RNA-Site

In this directory you can find the implementation of the `RNA-Site` task.
Given an RNA structure, it predicts whether each residue is part of a small-molecule binding site.
The task is implemented on the `rnaglib` database (`BindingSite`) and on the dataset and splits of:

> Hong Su, Zhenling Peng, and Jianyi Yang. Recognition of small molecule–rna binding sites using
> rna sequence and structure. Bioinformatics, 37(1):36–42, 2021. <https://doi.org/10.1093/bioinformatics/btaa1092>

## Project Structure

1. `binding_site.py`
2. `demo.py`
3. `benchmark_demo.py`

### binding_site.py

Task definitions using `rnaglib`'s task API:

- `BindingSite`: a residue is binding if any of its atoms lies within `cutoff` (default 6 Å) of a small-molecule
  ligand. The dataset contains the connected components of 15 to 500 residues with at least 10 binding residues
  (the HARIBOSS binding-site definition). Redundancy is removed on sequence (CD-Hit) and structure (US-align), and
  the split clusters structures by US-align similarity.
- `BenchmarkBindingSite`: same input and output, built on the chains and the TR60/TE18 split of Su et al. (2021);
  as in that benchmark, ion-binding residues also count as binding.

Residues are labelled from the deposited structures only: a residue that binds a ligand in another deposition of the
same RNA is labelled as non-binding.

### demo.py

Trains an RGCN (from the `learning` directory of this repo) on `BindingSite` and reports test metrics.

### benchmark_demo.py

Same as `demo.py`, on `BenchmarkBindingSite`.

## Usage

To train and evaluate the model, run `python demo.py` or `python benchmark_demo.py`.
