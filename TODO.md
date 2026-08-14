# TODO

## Clean up scratch files and generated task artifacts

Untracked cruft currently sitting in the working tree:

- Scratch scripts: `src/rnaglib/tasks/RNA_Ligand/toto.py`, `src/rnaglib/tasks/RNA_Prot/toto.py`, `src/rnaglib/tasks/RNA_CM/toto.py`
- Generated dataset build directories: `src/rnaglib/tasks/RNA_CM/RNA_CM/`, `src/rnaglib/tasks/RNA_GO/RNA_GO/`, `src/rnaglib/tasks/RNA_GO/RNA_GO_random/`, `src/rnaglib/tasks/RNA_IF/RNA_IF/`, `src/rnaglib/tasks/RNA_Ligand/RNA_Ligand/`, `src/rnaglib/tasks/RNA_Prot/RNA_Prot_final/`, `src/rnaglib/tasks/RNA_Site/RNA_Site/`
- `rna_site.tar.gz` at repo root

Need to: decide what's worth keeping vs deleting, remove the rest, and add a CLAUDE.md rule (drafted, not yet added) against leaving scratch scripts/build artifacts under `src/`.

## Add test/lint workflow section to CLAUDE.md

Document how to run tests (`pytest`, `testpaths=tests`, `pythonpath=src`) and lint (`ruff check`, `select=ALL` minus `ANN`) in CLAUDE.md.

First review/clean up the existing suite under `tests/`: `test_data_loading.py`, `test_features_computer.py`, `test_prepare_data.py`, `test_representations.py`, `test_splitters.py`, `test_tasks.py`, `test_transforms.py`, `test_utils.py`.

Deferred by user on 2026-07-31, pending a separate pass on the tests themselves.

## Publish RNA_CM debug data version 3.0.0 to Zenodo

`ChemicalModification.version` was bumped to `"3.0.0"` in `01477f63` (restrict annotated modifications to natural
modifications) but no `rnaglib-{debug,nr,all}-3.0.0.tar.gz` exists on Zenodo record `14930728` yet (all return 404).
`tests/test_tasks.py::test_ChemicalModification` and `::test_eval` are skipped until the v3 data is deployed.

## RNA_Ligand family_map.json is missing ~2/3 of referenced ligand codes

`src/rnaglib/tasks/RNA_Ligand/data/family_map.json` maps only 121 of the 345 distinct PDB chemical component codes
referenced in `data/ligands_dict.json` (224 unmapped). Several unmapped codes are known members of families already
present in the map, e.g. T1C (tigecycline, tetracycline), SCM (spectinomycin), HGR (hygromycin A), PCY (pactamycin),
LUJ (aminoglycoside-like sugar conjugate) — plus at least two antibiotic classes with no bucket at all yet: CLM
(chloramphenicol, amphenicol) and ANM (anisomycin, pyrrolidine antibiotic).

This isn't just a test-data gap: in production (non-debug) `LigandIdentification.process()` silently drops any pocket
whose ligand isn't in the map, so `aminoglycoside`/`tetracycline` classes are under-represented in the real task
dataset today. Needs a curation pass (domain call, not a mechanical fix) to extend `family_map.json` — scope
separately.
