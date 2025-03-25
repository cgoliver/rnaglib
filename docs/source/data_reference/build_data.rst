Building 3D RNA Databases
----------------------------------------

The (`prepare_data`) contains all the necessary code to build databases of annotated RNA 3D structures, and the user interfaces with it through the `rnaglib_prepare_data` command line script.
Database creation follows the following steps:

* Fetching the raw RNA structures from either:
        * RCSB PDB Databank (accepts the `--nr` flag to only use structures in the [BGSU Representative Set](https://www.bgsu.edu/research/rna/databases/non-redundant-list.html)
        * A local user-defined folder
* For each structure, run fr3d-python.
* Store fr3d-python output in a networkx Graph object
* Include addidtional annotations to the graph (e.g. small molecule binding
  info, chemical modifications, etc.)


Quickstart
~~~~~~~~~~~~~

Print the help message:

::

        $ rnaglib_prepare_data -h

To run a quick debug build with default values:

::

        $ rnaglib_prepare_data -s structures/ --tag first_build -o builds/ -d

Data versioning
~~~~~~~~~~~~~~~~~~~~~~~~~

The optional argument `--tag` is used to name the folder containing the final output.
For our distributions we use `rnaglib-<'all' or 'nr'><'-annotated' or ''><'-chopped' or ''>-<version>` depending on the build options.
We distribute data builds with all available RNAs and assign `all` to the tag, and non-redundant structures according to the [BGSU Representative Set](https://www.bgsu.edu/research/rna/databases/non-redundant-list.html).

Output 
~~~~~~~~~~~~~~~~~~~~~~~~~

After running the `--debug` test run above, your `./builds/` folder will contain a single sub-folder called `./builds/graphs` with 10 `.json` files and a file `./builds/graphs/errors.csv`. Each of these JSONs contains the annotated RNAs and the CSV contains a list of RNAs that failed to build and the failure reason.

Data building options
~~~~~~~~~~~~~~~~~~~~~~~~~

* `--nr` only outputs RNAs from the non-redundant set from BGSU

Hosting
~~~~~~~~
