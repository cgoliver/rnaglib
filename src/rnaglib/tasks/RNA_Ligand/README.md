# RNA-Ligand

In this directory you can find the implementation of the `RNA-Ligand` task (`LigandIdentification`, `rna_ligand`).
Given the structure of an RNA binding pocket, it predicts which group of ligands it binds (pocket-level multi-class
classification).

## Project Structure

1. `ligand_identity.py`
2. `demo.py`
3. `data/`

### ligand_identity.py

Task definition using `rnaglib`'s task API:

- **Pockets** are built directly from all-atom graphs (`graph_path`, built with `atom_coords_to_store=["all_atom"]`):
  each ligand copy with at least `min_rna_contacts` RNA residues within 4 Å defines one pocket, the RNA residues
  within `pocket_cutoff` of it.
- **Labels** are curated ligand groups describing the recognition mode (`data/ligand_groups.json`). Non-specific
  binders (buffers, polyamines, ...) are left out, and so are nucleotides and nucleotide cofactors by default.
- **Each binding site is kept once**: duplicate copies and depositions are removed, and ribosomal sites are matched
  across organisms through E. coli rRNA numbering (`data/ecoli_rrna_reference.json`). At most
  `max_sites_per_molecule` sites are kept per RNA molecule.
- **Split**: every cluster of similar RNA molecules (US-align and sequence identity) stays within a single split,
  while every class is balanced across splits (`MoleculeClusterStratifiedSplitter`).
- **Sample weights** give every binding site the same weight within its class, whatever its number of sequence
  variants.

See the module docstring for the full definition. Building the task requires US-align.

### demo.py

Trains an RGCN (from the `learning` directory of this repo) and reports test metrics.

## Usage

To train and evaluate the model, set `graph_path` in `demo.py` and run `python demo.py`.
