import sys
import os
import traceback
import json
import pickle
import requests
import urllib
from pathlib import Path
from typing import Optional, List, Tuple
from collections import defaultdict

import requests
import warnings
import pandas as pd

import tarfile
import zipfile

import networkx as nx
from Bio.PDB.PDBList import PDBList


def dump_json(filename, graph):
    """
    Just a shortcut to dump a json graph more compactly.

    :param filename: The dump name
    :param graph: The graph to dump
    """
    # This is important for nx versionning retrocompatibility
    try:
        from networkx.readwrite import json_graph

        g_json = json_graph.node_link_data(graph)
    except Exception as e:
        g_json = nx.node_link_data(graph, edges="links")

    with open(filename, "w") as f:
        json.dump(g_json, f, indent=2)


def load_json(filename):
    """
    Just a shortcut to load a json graph more compactly.

    :param filename: The dump name

    :return: The loaded graph
    """
    with open(filename, "r") as f:
        js_graph = json.load(f)
    # This is important for nx versionning retrocompatibility
    try:
        from networkx.readwrite import json_graph

        out_graph = json_graph.node_link_graph(js_graph)
    except Exception as e:
        out_graph = nx.node_link_graph(js_graph, edges="links")
    return out_graph


def load_graph(filename):
    """
    This is a utility function that supports loading from json or pickle.
    Sometimes, the pickle also contains rings in the form of a node dict,
    in which case the rings are added into the graph

    :param filename: json or pickle filename

    :return: networkx DiGraph object
    """
    if filename.endswith("json"):
        return load_json(filename)
    elif filename.endswith("p"):
        pickled = pickle.load(open(filename, "rb"))
        # Depending on the data versionning, the object contained in the pickles is
        # - a graph with noderings in the nodes
        # - a dict {graph: , rings: }
        if isinstance(pickled, dict):
            graph = pickled["graph"]
            # rings is a dict of dict {ring_type : {node : ring}}
            rings = pickled["rings"]
            for ring_type, noderings in rings.items():
                nx.set_node_attributes(
                    G=graph, name=f"{ring_type}_annots", values=noderings
                )
        else:
            graph = pickled
        return graph

    else:
        raise NotImplementedError("We have not implemented this data format yet")


def get_name_extension(filename, permissive=False):
    if filename.endswith(".json"):
        fname, extension = filename[:-5], filename[-5:]
    elif filename.endswith(".p"):
        fname, extension = filename[:-2], filename[-2:]
    else:
        if permissive:
            fname, extension = filename, None
        else:
            raise NotImplementedError("We have not implemented this data format yet")
    return fname, extension


def get_all_existing(
        dataset_path: os.PathLike, all_rnas: Optional[List[str]] = None
) -> Tuple[List[str], str]:
    """
    Return list of graph IDs in a given dataset directory in sorted() order. If you pass ``all_rnas``
    as a list of, returns the graph IDs in ``all_rnas`` that have a matching file in the dataset folder.

    :param dataset_path: where all the RNA JSONs are stored.
    :param all_rnas: list of RNA names to search for (e.g. ``'1aju'`` will match ``'1aju.json'`` in ``dataset_path``.
    :return: List of filenames in ``dataset_path``
    """
    _, extension = get_name_extension(os.listdir(dataset_path)[0])

    # By default, return a sorted listdir
    if all_rnas is None:
        return [Path(g).stem for g in sorted(os.listdir(dataset_path))], extension

    # Filter out existing ones, and print message if there is a difference
    existing_all_rnas = [g_name for g_name in all_rnas if os.path.exists(Path(dataset_path) / f"{g_name}{extension}")]
    size_diff = len(all_rnas) - len(existing_all_rnas)
    if size_diff > 0:
        print(f"{size_diff} graphs were missing from {dataset_path} compared to asked graphs")
    return existing_all_rnas, extension


def get_default_download_dir():
    """
    Get the absolute path to the download directory.
    """
    dirname = os.path.join(os.path.expanduser("~"), ".rnaglib/")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        os.makedirs(os.path.join(dirname, "downloads"))
        os.makedirs(os.path.join(dirname, "datasets"))
    return dirname


def download(url, path=None, overwrite=True, retries=5, verify_ssl=True, log=True):
    """
    Download a given URL.

    Codes borrowed from mxnet/gluon/utils.py

    :param url: URL to download.
    :param path:  Destination path to store downloaded file. By default stores to the current directory
     with the same name as in url.
    :param overwrite: Whether to overwrite the destination file if it already exists.
        By default always overwrites the downloaded file.
    :param retries: The number of times to attempt downloading in case of failure or non 200 return codes.
    :param verify_ssl: bool, default True. Verify SSL certificates.
    :param log:  bool, default True Whether to print the progress for download

    :return: The file path of the downloaded file.

    """
    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, (
            "Can't construct file-name from this URL. "
            "Please set the `path` option manually."
        )
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised."
        )

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print("Downloading %s from %s..." % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                total_length = int(r.headers.get("content-length"))
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, "wb") as f:
                    dl = 0
                    for chunk in r.iter_content(chunk_size=1024):
                        dl += len(chunk)
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            done = int(50 * dl / total_length)
                            sys.stdout.write(
                                "\r[%s%s] %.2f %%"
                                % (
                                    "=" * done,
                                    " " * (50 - done),
                                    dl / total_length * 100,
                                )
                            )
                            sys.stdout.flush()
                break
            except Exception as e:
                print(e)
                print(traceback.print_exc())
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print(
                            "download failed, retrying, {} attempt{} left".format(
                                retries, "s" if retries > 1 else ""
                            )
                        )

    return fname


def download_name_generator(
        version="1.0.0", redundancy="nr", annotated=False, record="7624873", debug=False
):
    """
    This returns the zenodo URL given dataset choices.

    :param version: Which data version to use.
    :param redundancy: Whether we want all RNA structures or just a filtered set
    :param annotated: Whether to include pre-computed annotation for each node with information
        to be used by kernel functions

    """
    # Generic name
    if debug:
        return f"https://github.com/cgoliver/rnaglib/raw/master/examples/rnaglib-debug-{version}.tar.gz"

    # Find remote url and get download link
    # full = https://zenodo.org/records/7624873/files/rnaglib-all-1.0.0.tar.gz?download=1
    if annotated:
        if version == "1.0.0":
            print("Annotated version for v 1.0.0 not available. Try a higher version")
            return None
        return f"https://zenodo.org/records/{record}/files/rnaglib-{redundancy}-{version}-annotated.tar.gz?download=1"
    else:
        return f"https://zenodo.org/records/{record}/files/rnaglib-{redundancy}-{version}.tar.gz?download=1"


def download_graphs(
        redundancy="nr",
        version="1.0.0",
        annotated=False,
        chop=False,
        overwrite=False,
        data_root=None,
        verbose=False,
        get_pdbs=False,
        debug=False,
):
    """
    Based on the options, get the right data from the latest release and put it in download_dir.

    :param redundancy: Whether to include all RNAs or just a non-redundant set as defined by BGSU
    :param annotated: Whether to include graphlet annotations in the graphs. This will also create a hashing directory and table
    :param overwrite: To overwrite existing data
    :param download_dir: Where to save this data. Defaults to ~/.rnaglib/
    :param store_pdbs: Whether to store assocaited PDBs in ~/.rnaglib/structures

    :return: the path of the data along with its hashing.

    """
    # Get the correct names for the download option and download the correct files
    hashing_path = None
    if data_root is None:
        data_root = get_default_download_dir()

    if debug:
        tag = f"rnaglib-debug-{version}"
    else:
        tag = f"rnaglib-{redundancy}-{version}{'-chop' if chop else ''}{'-' + 'annotated' if annotated else ''}"
    url = download_name_generator(
        redundancy=redundancy, version=version, annotated=annotated, debug=debug
    )
    dl_path = Path(data_root) / "downloads" / Path(tag + ".tar.gz")
    data_path = Path(data_root) / "datasets"

    if not os.path.exists(data_path / tag) or overwrite:
        print(
            "Required database not found, launching a download. This should take about a minute"
        )
        print(f"Fetching {url}")
        print(f"Downloading to : {dl_path}")
        print(f"Saving to : {data_path}")
        download(path=dl_path, url=url)
        # Expand the compressed files at the right location
        with tarfile.open(dl_path) as tar_file:
            tar_file.extractall(path=data_path)

        if get_pdbs:
            print("Fetching PDB structures")
            pdb_path = data_path / tag / "structures"
            pdb_path.mkdir(parents=True, exist_ok=True)
            rna_list = [Path(p).stem for p in os.listdir(data_path / tag / "graphs")]
            update_RNApdb(pdb_path, rna_list=rna_list, nr_only=redundancy == "nr")

    else:
        print(f"Database was found and not overwritten")
    return os.path.join(data_root, "datasets", tag)


def available_pdbids(
        graph_dir=None,
        version="1.0.0",
        chop=False,
        annotated=False,
        redundancy="nr",
        debug=False,
):
    if debug:
        tag = f"rnaglib-debug-{version}"
    else:
        tag = f"rnaglib-{redundancy}-{version}{'-chop' if chop else ''}{'-' + 'annotated' if annotated else ''}"
    if graph_dir is None:
        dl_dir = get_default_download_dir()
        graph_path = os.path.join(dl_dir, "datasets", tag, "graphs")
        if not os.path.exists(graph_path):
            print(
                f"Data build {graph_path} download not found. Use rnaglib_download to fetch"
            )
            return None
    else:
        graph_path = graph_dir

    return [os.path.splitext(g)[0] for g in os.listdir(graph_path)]


def graph_from_pdbid(
        pdbid,
        graph_dir=None,
        version="1.0.0",
        annotated=False,
        chop=False,
        redundancy="nr",
        graph_format="json",
):
    """Fetch an annotated graph with a PDBID.

    :param pdbid: PDB id to fetch
    :param graph_dir: path containing annotated graphs
    :param graph_format: which format to load (JSON, or networkx)
    """

    tag = f"rnaglib-{redundancy}-{version}{'-chop' if chop else ''}{'-' + 'annotated' if annotated else ''}"

    if graph_format == "nx":
        graph_name = os.path.join(pdbid.lower() + ".nx")
    elif graph_format == "json":
        graph_name = os.path.join(pdbid.lower() + ".json")
    else:
        raise ValueError(f"Invalid graph format {graph_format}. Use NetworkX or JSON.")

    graph_path = None

    # Try in look into the existing data, we need to check for both annotated and graphs, as well as in each dl
    if graph_dir is None:
        dl_dir = get_default_download_dir()
        graph_path = os.path.join(dl_dir, "datasets", tag, "graphs", graph_name)
        if not os.path.exists(graph_path):
            print(
                "The required pdb was not found in existing default downloads, "
                "please provide a path to look for the graph"
            )
            return None
    else:
        graph_path = os.path.join(graph_dir, graph_name)

    graph = load_graph(graph_path)
    return graph


def get_rna_list(nr_only=False):
    """
    Fetch a list of PDBs containing RNA from RCSB API.

    """
    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                "operator": "greater",
                "value": 0,
            },
        },
        "request_options": {"results_verbosity": "compact", "return_all_hits": True},
        "return_type": "entry",
    }

    payload = json.dumps(payload)
    encoded_query = urllib.parse.quote(payload)
    url = f"https://search.rcsb.org/rcsbsearch/v2/query?json={encoded_query}"
    r = requests.get(url)
    try:
        response_dict = json.loads(r.text)
        ids = [p.lower() for p in response_dict["result_set"]]
        if nr_only:
            nr_chains = [c.lower() for c in get_NRchains("4.0A")]
            ids = [pdbid.lower() for pdbid in ids if pdbid in nr_chains]
    except Exception as e:
        print(f"An error occured when querying RCSB or BGSU Atlas")
        print(r.text)
        print(e)
        exit()
    return ids


def get_NRlist(resolution):
    """
    Get non-redudant RNA list from the BGSU website

    :param resolution: minimum rseolution to apply
    """

    base_url = "http://rna.bgsu.edu/rna3dhub/nrlist/download/rna"
    release = "current"  # can be replaced with a specific release id, e.g. 0.70
    # release = '3.186'
    url = "/".join([base_url, release, resolution])
    print(url)

    df = pd.read_csv(url, header=None)

    repr_set = []
    for ife in df[1]:
        repr_set.append(ife)

    return repr_set


def load_csv(input_file, quiet=False):
    """
    Load a csv of from rna.bgsu.edu of representative set

    :param input_file: path to csv file
    :param quiet: set to true to turn off warnings
    :return repr_set: list of equivalence class RNAs
    """
    NRlist = []
    with open(input_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                NRlist.append(row[1])
            except csv.Error as e:
                if not quiet:
                    print(f"Warning error {e} found when trying to parse row: \n {row}")

    return NRlist


def parse_NRlist(NRlist):
    """
    Parse NR BGSU csv file for a list of non-redundant RNA chains
    list can be downloaded from:
        http://rna.bgsu.edu/rna3dhub/nrlist

    :param NRlist: Set of representative RNAs output (see load_csv())

    :return: set of non-redundant RNA chains (tuples of (structure, model, chain))
    """

    NRchains = defaultdict(set)

    # split into each IFE (Integrated Functional Element)
    for representative in NRlist:
        items = representative.split("+")
        for entry in items:
            pbid, model, chain = entry.split("|")
            NRchains[pbid.lower()].add(chain)

    return NRchains


def get_NRchains(resolution):
    """
    Get a map of non redundant IFEs (integrated functional elements) from
    rna.bgsu.edu/rna3dhub/nrlist

    :param resolution: (string) one of
    [1.0A, 1.5A, 2.0A, 2.5A, 3.0A, 3.5A, 4.0A, 20.0A]
    :return: Dictionary, keys=PDB IDs, Values=(set) Chain IDs
    """

    NR_list = get_NRlist(resolution)
    return parse_NRlist(NR_list)


def update_RNApdb(pdir, nr_only=True, rna_list=None):
    """
    Download a list of RNA containing structures from the PDB
    overwrite exising files

    :param pdbdir: path containing downloaded PDBs
    :param rna_list: list of PDBIDs to download

    :returns rna: list of PDBIDs that were fetched.
    """
    print(f"Updating PDB mirror in {pdir}")
    # Get a list of PDBs containing RNA
    if not rna_list is None:
        rna = rna_list
    else:
        rna = set(get_rna_list(nr_only=nr_only))

    print(rna)
    pl = PDBList()

    # If not fully downloaded before, download all structures
    if len(os.listdir(pdir)) < 2000:
        pl.download_pdb_files(rna, pdir=pdir, overwrite=True)
    else:
        added, mod, obsolete = pl.get_recent_changes()
        # Download new and modded entries
        new_rna = rna.intersection(set(added).union(set(mod)))
        pl.download_pdb_files(new_rna, pdir=pdir, overwrite=True)

        # Remove Obsolete entries
        obsolete_dir = os.path.join(pdir, "obsolete")
        if not os.path.exists(obsolete_dir):
            os.mkdir(obsolete_dir)
        for cif in os.listdir(pdir):
            if cif[-8:-4].upper() in set(obsolete):
                os.rename(os.path.join(pdir, cif), os.path.join(obsolete_dir, cif))

    return rna


def get_Ribochains():
    """
    Get a list of all PDB structures containing RNA and have the text 'ribosome'

    :return: dictionary, keys=pbid, value='all'
    """
    q1 = Attr("rcsb_entry_info.polymer_entity_count_RNA") >= 1
    q2 = TextQuery("ribosome")

    query = q1 & q2

    results = set(query())

    # print("textquery len: ", len(set(q2())))
    # print("RNA query len: ", len(set(q1())))
    # print("intersection len: ", len(results))
    return set(query())


def get_NonRibochains():
    """
    Get a list of all PDB structures containing RNA
    and do not have the text 'ribosome'

    :return: dictionary, keys=pbid, value='all'
    """
    q1 = Attr("rcsb_entry_info.polymer_entity_count_RNA") >= 1
    q2 = TextQuery("ribosome")

    return set(q1()).difference(set(q2()))


def get_Custom(text):
    """
    Get a list of all PDB structures containing RNA
    and do not have the text 'ribosome'

    :return: dictionary, keys=pbid, value='all'
    """
    q1 = Attr("rcsb_entry_info.polymer_entity_count_RNA") >= 1
    q2 = TextQuery(text)

    query = q1 & q2

    return set(query())


if __name__ == "__main__":
    # tmp_path = '../../examples/2du5.json'
    # g = load_json(tmp_path)
    # print(g.nodes())
    default = get_default_download_dir()
    print(default)
    graph_from_pdbid("4nlf")
