import os
import tempfile

from Bio.PDB.PDBList import PDBList

from rnaglib.prepare_data import build_graph_from_cif
from rnaglib.utils import get_default_download_dir, load_graph


def rna_from_pdbid(
    pdbid,
    version="2.0.0",
    annotated=False,
    chop=False,
    redundancy="nr",
    download_dir=None,
        verbose=True,
):
    """Fetch an annotated graph with a PDBID.

    :param pdbid: PDB id to fetch
    :param version: database version to query
    :param graph_dir: path containing annotated graphs

    :return: RNA dictionary object.
    """

    tag = f"rnaglib-{redundancy}-{version}{'-chop' if chop else ''}{'-' + 'annotated' if annotated else ''}"

    graph_path = None

    # Try in look into the existing data, we need to check for both annotated and graphs, as well as in each dl
    dl_dir = get_default_download_dir()
    graph_path = os.path.join(dl_dir, "datasets", tag, "graphs", f"{pdbid.lower()}.json")
    if not os.path.exists(graph_path):
        print(
            """The required pdb was not found in existing default downloads.
Fetching PDB and annotating...
If you want to use a local graph,
pass a path to the  `graph_dir` argument. """
        )
        pl = PDBList()
        with tempfile.TemporaryDirectory() as tmpdir:
            if download_dir is None:
                pdir = tmpdir
            cif_path = pl.retrieve_pdb_file(pdbid, pdir=pdir, file_format="mmCif")
            graph = build_graph_from_cif(cif_path, None)

    else:
        if verbose:
            print("Loading graph from local database...")
        graph = {"rna": load_graph(graph_path)}
    return graph
