In this folder, we define global variables that are useful for RNA represented as a graph.

We define an extended version of the isostericity matrix that includes the possibility to 
use the isostericity values for directed graphs.

We also define the major key mappings as used by convention in our data.

Before parsing mmCIF files, a <mark>modifications_cache.json</mark> must be generated. This file contains conversions from <mark>three_letter_code</mark> to <mark>one_letter_code</mark> for modified residues. To do this, we start by dowloading the latest version of <mark>components.cif</mark> from the [Chemical Component Dictionary](https://www.wwpdb.org/data/ccd), and running:
```
python3 rnaglib/config/generate_modifications_cache.py path/to/components.cif rnaglib/config/cache/modifications_cache.json
```
