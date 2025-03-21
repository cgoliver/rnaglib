What is an RNA 2.5D structure?
----------------------------------

RNA 2.5D structures are discrete graph-based representations of atomic coordinates derived
from techniques such as X-ray crystallography and NMR. This type of representation encodes
all possible base pairing interactions which are known to be crucial for understanding RNA function.


.. |Example graph| image:: https://jwgitlab.cs.mcgill.ca/cgoliver/rnaglib/-/raw/zenodo/images/1qvg_graphandchimera.png

|Example graph|

**Why use RNA 2.5D data?**

The benefit is twofold. When dealing with RNA 3D data, a representation centered on
base pairing is a very natural prior which has been shown to carry important signals for
complex interactions, and can be directly interpreted.
Second, adopting graph representations lets us take advantage of many powerful algorithmic tools
such as graph neural networks and graph kernels.

**What type of functional data is included?**

The graphs are annotated with graph, node, and edge-level attributes.
These include, but are not limited to:

-  Secondary structure 
-  Protein binding 
-  Small molecule binding 
-  Chemical modifications 
-  3-D coordinates
-  Leontis-westhof base pair geometry classification 

We provide a visualization of what the graphs in this database contain.
A more detailed description of the data is presented in :doc:`../data_reference/rna_ref`.

