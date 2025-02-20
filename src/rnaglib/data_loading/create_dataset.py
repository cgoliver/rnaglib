import os
from networkx import set_node_attributes
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import numpy as np

from rnaglib.utils import download_graphs, load_graph, dump_json
from rnaglib.utils.graph_io import get_all_existing, get_name_extension

if TYPE_CHECKING:
    from rnaglib.data_loading.features import FeaturesComputer


def database_to_dataset_loop(
    all_rnas_db,
    db_path,
    rna_filter: Callable = None,
    nt_filter: Callable = None,
    pre_transforms: Callable = None,
    features_computer: "FeaturesComputer" = None,
):
    """Iterates through database, applying filters and annotations"""
    from tqdm import tqdm as tqdm

    rna_list = []

    for rna_filename in tqdm(all_rnas_db):
        rna_filename += ".json"
        rna_path = os.path.join(db_path, rna_filename)
        rna = load_graph(rna_path)

        # Remove whole systems
        if rna_filter is not None and not rna_filter(rna):
            continue

        # Apply a chunking function to whole RNAs
        if nt_filter is not None:
            subgs = []

            for subg in nt_filter(rna):
                subgs.append(subg)
        else:
            subgs = [rna]

        # Apply a per graph/subgraph function
        if pre_transforms is not None:
            for subg in subgs:
                pre_transforms({"rna": subg})

        # Add a 'name' field to the graphs if annotator did not put one.
        rna_name, rna_extension = get_name_extension(rna_filename)
        for i, subg in enumerate(subgs):
            if subg.name == "":
                if len(subgs) == 1:
                    subg.name = rna_name
                else:
                    subg.name = f"{rna_name}_{i}"
        assert len(subgs) == len(set([rna.name for rna in subgs])), (
            "When adding several subgraphs in nt_filter," " make sure to use unique names for each subgraphs"
        )

        # Remove useless keys
        if features_computer is not None:
            subgs = [features_computer.remove_useless_keys(subg) for subg in subgs]

        rna_list.extend(subgs)
    return rna_list


def database_to_dataset(
    dataset_path=None,
    recompute=False,
    all_rnas=None,
    return_rnas=True,
    pre_transforms=None,
    nt_filter=None,
    rna_filter=None,
    features_computer=None,
    db_path=None,
    all_rnas_db=None,
    version="1.0.0",
    download_dir=None,
    redundancy="nr",
    annotated=False,
    debug=False,
):
    """
    Function to
    :param dataset_path: Path to an already saved dataset, skips dataset creation if loaded.
    :param recompute: Boolean if we should recompute
    :param all_rnas: A list of file names if we're using precomputed data
    :param return_rnas: If we're using existing files, shall we load them or just the file names ?

    :param db_path: The original database directory to produce our data from. If unset, further params are used. (below)
    :param all_rnas_db: If we want to only precompute over only a subset of the db
    :param redundancy: To use all graphs or just the non-redundant set.
    :param download_dir: If one changed the default download directory of rglib
    :param version: Version of the dataset to use (default='1.0.0')
    :param annotated: To use for pretraining

    :param nt_filter: Callable which takes as input an RNA dictionary and filters out some nt
    :param pre_transform: Callable which takes as input an RNA dictionary and adds new key-value pairs.
    :param rna_filter: Callable which takes as input an RNA dictionary and returns whether we should keep it.
    """
    # If this corresponds to a dataset that was precomputed already, just return the graphs
    if dataset_path is not None and os.path.exists(dataset_path) and not recompute:
        existing_all_rnas, extension = get_all_existing(dataset_path=dataset_path, all_rnas=all_rnas)
        if return_rnas:
            rnas = [load_graph(os.path.join(dataset_path, g_name + extension)) for g_name in existing_all_rnas]
            for rna, name in zip(rnas, existing_all_rnas):
                rna.name = name
        else:
            rnas = None
        return dataset_path, existing_all_rnas, rnas

    # Set up the original data to build the dataset from
    # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched

    # By default, we set hashing to None and potential node sim should be specified when creating
    # the node_sim function. Then, if a download occurs and no hashing was provided to the loader, the hashing used
    # is the one fetched by the downloading process to ensure it matches the data we iterate over.
    # TODO, check annotations and pretraining still work
    if db_path is None:
        db_path = download_graphs(
            redundancy=redundancy,
            version=version,
            annotated=annotated,
            data_root=download_dir,
            debug=debug,
        )[0]

    if all_rnas_db is None:
        all_rnas_db = [f.split(".")[0] for f in os.listdir(db_path)]

    # If no constructions args are given, just return the graphs
    if rna_filter is None and nt_filter is None and pre_transforms is None and features_computer is None:
        rnas = [load_graph(Path(db_path) / g_name + ".json") for g_name in all_rnas_db]
        return rnas

    # If some constructions args are given, launch processing.
    rnas = database_to_dataset_loop(
        all_rnas_db=all_rnas_db,
        db_path=db_path,
        rna_filter=rna_filter,
        nt_filter=nt_filter,
        pre_transforms=pre_transforms,
        features_computer=features_computer,
    )
    all_rnas_name = [rna.name for rna in rnas]
    """
    if dataset_path is not None:
        os.makedirs(dataset_path, exist_ok=True)
        for i, rna in enumerate(rnas):
            dump_json(os.path.join(dataset_path, f"{rna.name}.json"), rna)
    """
    return dataset_path, all_rnas_name, rnas


def nt_filter_split_chains(x, rna_id_to_chains):
    pdb_id = x.graph["pdbid"][0].lower()
    chains = rna_id_to_chains[pdb_id]
    for chain in chains:
        wrong_chain_nodes = [node for node in list(x) if chain != node.split(".")[1]]
        subgraph = x.copy()
        subgraph.remove_nodes_from(wrong_chain_nodes)
        subgraph.name = f"{pdb_id}_{chain}"
        yield subgraph


def annotator_add_embeddings(x, embs_path=None):
    if embs_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        embs_path = os.path.join(script_dir, "../data/rnafm_chain_embs")
    all_chains = {node.rsplit(".", 1)[0] for node in x.nodes}
    all_chain_embs = {}
    for chain in all_chains:
        chain_embs = np.load(os.path.join(embs_path, f"{chain}.npz"))
        all_chain_embs.update(chain_embs)
    # needs to be list or won't be json serialisable
    embeddings = {node: all_chain_embs[node].tolist() for node, nodedata in x.nodes.items()}
    set_node_attributes(x, embeddings, "embeddings")
