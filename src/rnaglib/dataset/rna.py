import os
import tempfile

from Bio.PDB.PDBList import PDBList

from rnaglib.prepare_data import build_graph_from_cif
from rnaglib.utils import get_default_download_dir, load_graph, dump_json, load_graph


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

class RNA:
    def __init__(self, rna_dict: dict = None, pdbid: str = None, path: str = None, multigraph: bool = False):
        # check that only one of the three is provided
        self.multigraph = multigraph

        if sum([rna_dict is not None, pdbid is not None, path is not None]) != 1:
            raise ValueError("Only one of rna_dict, pdbid, or path must be provided")

        if rna_dict is not None:
            self.from_dict(rna_dict)
        elif pdbid is not None:
            self.from_pdbid(pdbid)
        elif path is not None:
            self.from_path(path)
        else:
            raise ValueError("No valid input provided")

    def from_dict(self, rna_dict: dict):
        for k, v in rna_dict.items():
            if k == 'rna':
                for attr, val in rna_dict['rna'].graph.items():
                    setattr(self, k, v)
            else:
                setattr(self, k, v)

    def from_pdbid(self, pdbid: str):
        rna_dict = rna_from_pdbid(pdbid, multigraph=self.multigraph)
        self.from_dict(rna_dict)

    def to_dict(self):
        return self.rna_dict

    def save(self, path: str):
        dump_json(path, self.rna_dict)
        pass

    def from_path(self, path: str):
        self.rna_dict = load_graph(path, multigraph=self.multigraph)
        self.from_dict(self.rna_dict)
