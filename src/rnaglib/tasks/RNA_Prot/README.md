# RNA-Prot

In this directory you can find the implementation of the `RNA-Prot` task.
This task is analogous to `RNA-Site`: it predicts, for each residue, whether it is part of a protein-binding
interface.

## Project Structure

1. `protein_binding_site.py`
2. `demo.py`

### protein_binding_site.py

Task definition using `rnaglib`'s task API (`ProteinBindingSite`):

- The dataset contains the connected components (backbone and base-pair edges) of 15 to 500 residues, which leaves
  out large ribosomal RNAs, with at least one protein-binding and one non-binding residue.
- Redundancy is removed on sequence (CD-Hit) and structure (US-align), and the split clusters structures by
  US-align similarity.
- `exclude_chaperone_contacts=True` relabels as non-binding the residues contacted only by crystallization
  chaperones (U1A RRM domain, Fab/Fv frameworks; see `CHAPERONE_CHAINS`), which are not biological partners of the
  RNA.

### demo.py

Trains an RGCN (from the `learning` directory of this repo) and reports test metrics.

## Usage

To train and evaluate the model, run `python demo.py`.
