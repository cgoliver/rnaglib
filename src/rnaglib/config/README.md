In this folder, we define global variables that are useful for RNA represented as a graph.

We define an extended version of the isostericity matrix that includes the possibility to 
use the isostericity values for directed graphs.

We also define the major key mappings as used by convention in our data.

Before parsing mmCIF files, a modifications_cache.json must be generated. This file contains conversions from three_letter_code to one_letter_code for modified residues. To do this, we start by dowloading the latest version of components.cif from the Chemical Component Dictionary, and running:
```
python rnaglib/config/generate_modifications_cache.py path/to/components.cif rnaglib/config/cache/modifications_cache.json
```
