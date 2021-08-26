# Data README

# 1. Data Preparation
Networkx graphs which are sliced into portions containing RNA interfaces and their respective complement counterparts. Graphs are Augmented Base Pairing Networks. Here is an example of one overlayed on a PDB structure. Backbones are in white, canonical Watson-Crick bonds are in green and non-canonical bonds are in red.


![RNA motif binding to CMC ligand](images/structures/1qvg_graphandchimera.png)

To generate this data:
1. Retrieve a representative set of RCSB PDB structures.
2. Find all interfaces within structures.
3. Slice native RNA graphs into interface and complement parts.

The `prepare_data` package contains all the scripts to do these tasks. The process can take some time so alternatively the following pre-built datasets can be downloaded from MEGA:

|	Dataset 	    |Graphs | Edges| Nodes  |Avg. Nodes | Avg. Edges|Links|
|---------------------------|-------:|------:|--------:|-------:|-----------:|-------|
|ALL                         |2679   | 447225 | 641968  |166.9|239.6|[link](https://mega.nz/file/hX4CVRib#ukoA6xaHdY14Vf9CIY7CXlFtycAmBIk16j6Oa65yJZo)|
|ALL complement              |9034   | 195395 | 228261 | 21.6 |25.3||
|RNA-Protein                 |2750   | 411487 | 587961 | 149.6|213.8|[link](https://mega.nz/file/9WpXHSab#JMtoU3RU-SZRqf4d34n3LRPHQNm2DqSwoIj5EUFtWrw)|
|RNA-Protein complement      |8265   | 241611 | 322324 | 29.2 |39.0||
|RNA-RNA                     |2737   | 59333  | 79116  | 21.7 |28.9|[link](https://mega.nz/file/xHpl3CIK#miMH5dVsqpLmJGmQSuR3qLCPhNmpXFOEVIhYKiOESuo)|
|RNA-RNA complement          |2483   | 55001  | 70551  | 22.2 |28.4||
|RNA-Small\_Mol.             |166    | 981    | 1004   | 5.9 |6.0|[link](https://mega.nz/file/lDhhTQZR#ovE1oZw1s6bolMA-AkMoljf6i4fV5Ih3yBme5RgjOGY)|
|RNA-Small\_Mol. complement  |140    | 973    | 1038   | 7.0|7.4||
|RNA-Ion                     |572    | 3490   | 3764   | 6.1  |6.6|[link](https://mega.nz/file/cGpFXC6J#NTOQ97TRmY9dzFfx3aYxk-ifJykxms1JQvfNGoSAj3A)|
|RNA-Ion complement          |493    | 3691   | 3993   | 7.5  |8.1||

## 1.1 Retrieve a Representative Set of PDB Structures
To avoid redundancies in the training data the BGSU representative set of RNAs are used.
They can be downloaded from [here](http://rna.bgsu.edu/rna3dhub/nrlist/release/3.145) [1]

Make a directory to store the structures

`mkdir data/structures`

Then run the following command to retrieve the PDB structures from the RCSB database

`python prepare_data/retrieve_structures.py <BGSU file> data/structures`

## 1.2 Find Interfaces in the PDB structures and Slice their RNA graphs
Make a directory for the native graphs and the interface graphs

`mkdir data/graphs`

`mkdir data/graphs/interfaces`

`mkdir data/graphs/native`

Download the set of native RNA graphs from [here]() and extract the compressed files into the `native` directory.

Now run `prepare_data/main.py` to find all the interfaces and slice the graphs. This process will take a few hours.

`python prepare_data/main.py data/graphs/interfaces`

#### Note
- The an optional parameter `-t` can be added to specify the RNA interaction type. The default is all but can be any of `rna protein ion ligand`. Use a string in quotations seperated by spaces for multple interaction types.
- Once the PDB interfaces are found, if you would like to run the script again use `-interface_list_input interface_residues_list.csv` option to use the interfaces computed from previous call and speed up execution.


## Associated Repositories:
[VeRNAl](https://github.com/cgoliver/vernal)

[RNAMigos](https://github.com/cgoliver/RNAmigos)

# References
1. Leontis, N. B., & Zirbel, C. L. (2012). Nonredundant 3D Structure Datasets for RNA Knowledge Extraction and Benchmarking. In RNA 3D Structure Analysis and Prediction N. Leontis & E. Westhof (Eds.), (Vol. 27, pp. 281–298). Springer Berlin Heidelberg. doi:10.1007/978-3-642-25740-7\_13


## Graph Format

Each graph contains structure information for one model of a PDB entry
containing at least one RNA chain.

Graphs are stored in `JSON` node-link format which can be loaded by
[networkx](https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.node_link_data.html#networkx.readwrite.json_graph.node_link_data).

All data comes from the output of `x3dna-dssr` which can be downloaded
[here](https://x3dna.org/) and our custom interface extraction tools.


Data sources to integrate:

* DSSP for protein structure info
* GO annotations
* RFAM annotations
* PDB metadata

### Creating a graph

TODO: will need to make a script that calls the DSSR annotator and Jonathan's
scripts.

### Loading a graph

Graphs are dumped as JSONs in the node-link format.

```python
import json
from networkx.readwrite.json_graph import node_link_graph

G  = node_link_graph(open('path/to/graph', 'r'))
```

### Graph

TODO

### Nodes 

#### Node IDs

Node IDs are strings in the form `[pdb id].[chain name].[residue number]`.

#### Node Data

To access node data dictionary:

```python
G.nodes[<node_id>]
```
These are the keys in the node data dictionary: 

* `'index'`: (int), relative index along chain starting at 1 (e.g. `1`)
* `'index_chain'`: (int) 26 (e.g. `26`)
* `'chain_name'`: (str), name of chain. (e.g. `A`)
* `'nt_resnum'`: (int), residue number according to PDB. (e.g. `101`)
* `'nt_name'`: (str), nucleotide name  (e.g. `G`)
* `'nt_code'`: (str), (e.g. `'U'`)
* `'nt_id'`: (str) unique nucleotide ID generated by DSSR. `<chain name>.<nt name><nt_resnum>` (e.g. `'A.U42'`),
* `'nt_type'`: (str) molecule type of residue (e.g. `'RNA'`)
* `'dbn':` (str) dot-bracket notation for the residue (e.g. `')'`)
* `'summary'`: (str) additional residue info (e.g. `"anti,~C3'-endo,BI,canonical,non-pair-contact,helix,stem,coaxial-stack"`)
* `'alpha'`: (float) base angle in degrees `[-180, 180]`.
* `'beta'`: (float)  base angle
* `'gamma'`: (float) base angle 
* `'delta'`: (float) base angle 
* `'epsilon'`: (float)
* `'zeta'`: (float) 
* `'epsilon_zeta'`: (float)
* `'bb_type':` (str) Backbone type 'BI',
* `'chi'`: (float) 
* `'glyco_bond'`: str (e.g.   `'anti'`
* `'C5prime_xyz': (list), 5' Carbon xyz coordinates (e.g.  `[-1.343, 8.453, 1.288]`)
* `'P_xyz'`: (list) Phosphate coordinates.
* `'form'`: (str) (e.g. `'A'`)  classification of a dinucleotide step comprising the bp above the given designation and the bp that follows it. Types include 'A', 'B' or 'Z' for the common A-, B- and Z- form helices, '.' for an unclassified step , and 'x' for a step without a continuous backbone.
* `'ssZp'`: (float)  (e.g. `4.41`),
* `'Dp'`: (float) (e.g. `4.404`)
* `'splay_angle'`: (float)  (e.g. `21.6`),
* `'splay_distance'`: (float) (e.g. `3.612`)
* `'splay_ratio':` (float) (e.g. `0.199`)
* `'eta'`: (float) (e.g. `169.652`),
* `'theta':` -167.457,
* `'eta_prime'`: (float) (e.g. `-176.189`)
* `'theta_prime':` (float) (e.g. `-167.27`)
* `'eta_base'`: (float)   (e.g. `-135.681`)
* `'theta_base'`: (float) (e.g. `-141.003`)
* `'v0'`: (float) (e.g `8.194`)
* `'v1'`: (float) (e.g. `-28.393`),
* `'v2'`: (float) 
* `'v3'`: (float) 
* `'v4'`: (float) 
* `'amplitude'`: (float)
* `'phase_angle'`: (float) 
* `'puckering'`: (str) (e.g. `"C3'-endo"`)
* `'sugar_class':` (str) (e.g. `"~C3'-endo"`)
* `'bin'`: (str) (e.g. `'33t'`) ( name of the 12 bins based on [ delta (i -1) , delta , gamma ], where delta (i -1) and delta can be either 3 ( for C3 '- endo sugar ) or 2 ( for C2 '- endo ) and gamma can be p/t/ m ( for gauche +/ trans / gauche - conformations , respectively ) (2 x2x3 =12 combinations : `33p` , `33t` , ... `22m`); `'inc'` refers to incomplete cases (i .e., with missing torsions ) , and `'trig'` to triages ( i.e., with torsion angle outliers ),\[1\]
* `'cluster'`: (str) (e.g. `'1c'`) (2-char suite name, for one of 53 reported clusters (46 certain and 7 wannabes ) , `'__'` for incomplete cases , and `'!!'` for outliers),\[1\]
* `'suiteness'`: (float) (measure of conformer - match quality ( low to high in range 0 to 1) ) \[1\]
* `'filter_rmsd'`: (float)
* `'frame':` (dict)  e.g. (`{'rmsd': 0.006, 'origin': [-4.856, 8.564, -1.171], 'x_axis': [0.922, 0.386, -0.006], 'y_axis': [0.098, -0.25, -0.963], 'z_axis': [-0.374, 0.888, -0.269], 'quaternion': [0.592, -0.781, -0.155, 0.122]}`
* `'sse'`: (dict) Secondary structure info (e.g. residue inside third hairpin `{'sse': 'hairpin-3'}`)
* `'binding_protein'`: (dict) RNA-Protein interface. If no interface found, `None`. Else, dictionary (e.g. `{'nt-aa': 'C-arg', 'nt': 'A.C37', 'aa': 'A.ARG47', 'Tdst': '6.62', 'Rdst': '-114.00', 'Tx': '-1.15', 'Ty': '1.89', 'Tz': '6.23', 'Rx': '-53.57', 'Ry': '19.41', 'Rz': '-103.42', 'sse': 'a-helix'}`)
* `binding_ion`: (string) molecule ID of ion if residue is at a binding site (otherwise `None`) (e.g. `'Ca'`)
* `binding_small-molecule`: (string) molecule ID of small molecule if residue is at a binding site (otherwise `None`) (e.g. `'SAM'`)

#### References
\[1\] [Richardson et al. (2008): "RNA backbone: consensus all-angle conformers and modular string nomenclature (an RNA Ontology Consortium contribution). RNA, 14(3):465-481](https://rnajournal.cshlp.org/content/14/3/465.short)


### Edge data

Each edge also has an attribute dictionary:

```python
G.edges[(<node_1>, <node_2>)]
```

* `'index'`: (int) Index of edge in DSSR ordering.
* `'nt1'`: (str) DSSR nucleotide ID of first base (e.g. `'A.G17'`)
* `'nt2'`: (str) DSSR nucleotide ID of second base (e.g. `'A.G29'`)
* `'bp'` (str): Nucleotide identity of paired residues (e.g. `'G-C'`)
* `'name'`: (str) (e.g. `'WC'`)
* `'Saenger'`: (str) Saenger base pairing category (e.g. `'19-XIX'`),
* `'LW'`: (str) Leontis-Westhof base pair geometry category (e.g. `'cWW'`)
* `'DSSR'`: (str) Custom DSSR base pair geometry category (e.g. `'cW-W'`)

