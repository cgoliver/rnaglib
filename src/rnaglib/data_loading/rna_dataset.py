import os
import sys
from pathlib import Path
from typing import Optional, Union, Literal

import copy
import torch
import networkx as nx

from rnaglib.utils import download_graphs, load_graph, dump_json
from rnaglib.data_loading.features import FeaturesComputer


def get_all_existing(dataset_path, all_graphs=None):
    # Only use queried graph, by default listdir
    all_graphs = sorted(os.listdir(dataset_path)) if all_graphs is None else all_graphs

    # Filter out existing ones, and print message if there is a difference
    all_graphs_path = [os.path.join(dataset_path, g_name) for g_name in all_graphs]
    existing_all_graphs_path = [g_path for g_path in all_graphs_path if os.path.exists(g_path)]
    size_diff = len(all_graphs) - len(existing_all_graphs_path)
    if size_diff > 0:
        print(f"{size_diff} graphs were missing from {dataset_path} compared to asked graphs")
    return existing_all_graphs_path


def build_dataset_loop(all_graphs_db, db_path, rna_filter=None, nt_filter=None, annotator=None,
                       features_computer: FeaturesComputer = None):
    """ Iterates through database, applying filters and annotations"""
    from tqdm import tqdm as tqdm
    graph_list = []

    for graph_name in tqdm(all_graphs_db):
        g_path = os.path.join(db_path, graph_name)
        g = load_graph(g_path)

        # Remove whole systems
        if not rna_filter(g):
            continue

        # Apply a chunking function to whole RNAs
        if not nt_filter is None:
            subgs = []

            for subg in nt_filter(g):
                subgs.append(subg)
        else:
            subgs = [g]

        # Apply a per graph/subgraph function
        if not annotator is None:
            for subg in subgs:
                annotator(subg)

        # Remove useless keys
        if features_computer is not None:
            subgs = [features_computer.remove_useless_keys(subg) for subg in subgs]

        graph_list.extend(subgs)
    return graph_list


def build_dataset(dataset_path=None, recompute=False, all_graphs=None,
                  annotator=None, nt_filter=None, rna_filter=None, features_computer=None,
                  db_path=None, all_graphs_db=None,
                  version='1.0.0', download_dir=None, redundancy='nr', annotated=False):
    """
    Function to
    :param dataset_path: Path to an already saved dataset, skips dataset creation if loaded.
    :param recompute: Boolean if we should recompute
    :param all_graphs: A list of file names if we're using precomputed data

    :param db_path: The original database directory to produce our data from. If unset, further params are used. (below)
    :param all_graphs_db: If we want to only precompute over only a subset of the db
    :param redundancy: To use all graphs or just the non-redundant set.
    :param download_dir: If one changed the default download directory of rglib
    :param version: Version of the dataset to use (default='1.0.0')
    :param annotated: To use for pretraining

    :param nt_filter: Callable which takes as input an RNA dictionary and filters out some nt
    :param annotator: Callable which takes as input an RNA dictionary and adds new key-value pairs.
    :param rna_filter: Callable which takes as input an RNA dictionary and returns whether we should keep it.
    """
    if not recompute and dataset_path is not None and os.path.exists(dataset_path):
        existing_all_graphs_path = get_all_existing(dataset_path=dataset_path, all_graphs=all_graphs)
        rnas = [load_graph(g_path) for g_path in existing_all_graphs_path]
        return rnas
        # return dataset_path, all_graphs

    # Set up the original data to build the dataset from
    # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched

    # By default, we set hashing to None and potential node sim should be specified when creating
    # the node_sim function. Then, if a download occurs and no hashing was provided to the loader, the hashing used
    # is the one fetched by the downloading process to ensure it matches the data we iterate over.
    # TODO, check annotations and pretraining still work
    if db_path is None:
        db_path = download_graphs(redundancy=redundancy,
                                  version=version,
                                  annotated=annotated,
                                  data_root=download_dir)
        db_path = os.path.join(db_path, 'graphs')

    all_graphs_db = os.listdir(db_path) if all_graphs_db is None else all_graphs_db

    # If no constructions args are given, just return the graphs
    if rna_filter is None and nt_filter is None and annotator is None and features_computer is None:
        rnas = [load_graph(os.path.join(db_path, g_name)) for g_name in all_graphs]
        return rnas

    # If some constructions args are given, launch processing.
    if rna_filter is None:
        rna_filter = lambda x: True
    rnas = build_dataset_loop(all_graphs_db=all_graphs_db, db_path=db_path,
                              rna_filter=rna_filter, nt_filter=nt_filter,
                              annotator=annotator, features_computer=features_computer)
    if dataset_path is not None:
        os.makedirs(dataset_path, exist_ok=True)
        for i, rna in enumerate(rnas):
            dump_json(os.path.join(dataset_path, f"{i}.json"), rna)
    return rnas
    # TODO this is broken, it should iterate over graphs or something
    # self.available_pdbids = [g.split(".")[0].lower() for g in self.all_graphs]


class RNADataset:
    """
        This class is the main object to hold the core RNA data annotations.
        The ``RNAglibDataset.all_rnas`` object is a generator networkx objects that hold all the annotations for each RNA in the dataset.
        You can also access individual RNAs on-disk with ``RNAGlibDataset()[idx]`` or ``RNAGlibDataset().get_pdbid('1b23')``
    """

    def __init__(self,
                 rnas=None,
                 dataset_path=None,
                 all_graphs=None,
                 # TODO add in_memory field to load on the fly
                 representations=None,
                 features_computer=None):
        """
        :param representations: List of `rnaglib.Representation` objects to apply to each item.
        :param dataset_path: The path to the folder containing the graphs.
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use
        """
        if rnas is None:
            if dataset_path is None:
                # By default, use non redundant (nr), v1.0.0 dataset of rglib
                dataset_path = download_graphs()
                dataset_path = os.path.join(dataset_path, 'graphs')

            # One can restrict the number of graphs to use
            #     TODO make the role of all_graphs clearer/refactor the dataset saving to make it more explicit
            existing_all_graphs_path = get_all_existing(dataset_path=dataset_path, all_graphs=all_graphs)
            rnas = [load_graph(g_path) for g_path in existing_all_graphs_path]
        self.rnas = rnas

        # Now that we have the raw data setup, let us set up the features we want to be using:
        if features_computer is None:
            features_computer = FeaturesComputer()
        self.features_computer = features_computer

        # Finally, let us set up the list of representations that we will be using
        if representations is None:
            self.representations = []
        elif not isinstance(representations, list):
            self.representations = [representations]
        else:
            self.representations = representations

    @classmethod
    def from_args(cls, representations=None, features_computer=None, **dataset_build_params):
        rnas = build_dataset(features_computer=features_computer, **dataset_build_params)
        return cls(representations=representations,
                   features_computer=features_computer,
                   rnas=rnas)
        # dataset_path=data.dataset_path,
        # all_graphs=data.all_graphs)

    def __len__(self):
        return len(self.rnas)

    def __getitem__(self, idx):
        """ Fetches one RNA and converts it from raw data to a dictionary
        with representations and annotations to be used by loaders """

        rna_graph = self.rnas[idx]
        rna_dict = {'rna': rna_graph}
        features_dict = self.features_computer.compute_features(rna_dict)

        # apply representations to the res_dict
        # each is a callable that updates the res_dict
        for rep in self.representations:
            rna_dict[rep.name] = rep(rna_graph, features_dict)
        return rna_dict

    def add_representation(self, representation):
        self.representations.append(representation)

    def remove_representation(self, name):
        self.representations = [representation for representation in self.representations
                                if representation.name != name]

    def subset(self, list_of_ids):
        """
        Create another dataset with only the specified graphs

        :param list_of_graphs: a list of graph names
        :return: A graphdataset
        """
        subset = copy.deepcopy(self)
        subset.rnas = [self.rnas[i] for i in list_of_ids]
        # TODO: also subset available pdbids and all graphs
        return subset

    def save(self, dump_path):
        """ Save a local copy of the dataset"""
        for i, rna in enumerate(self.rnas):
            dump_json(os.path.join(dump_path, f"{i}.json"), rna)

    def get_pdbid(self, pdbid):
        """ Grab an RNA by its pdbid """
        # TODO fix by subclassing to get a PDBRNADataset ?
        return self.__getitem__(self.all_graphs.index(pdbid.lower()))


if __name__ == '__main__':
    from rnaglib.representations import GraphRepresentation

    features_computer = FeaturesComputer(nt_features='nt_code', nt_targets='binding_protein')
    graph_rep = GraphRepresentation(framework='dgl')
    all_graphs = ['1a9n.json', '1b23.json', '1b7f.json', '1csl.json', '1d4r.json', '1dfu.json', '1duq.json',
                  '1e8o.json', '1ec6.json', '1et4.json']

    # # First case
    # supervised_dataset = RNADataset(all_graphs=all_graphs,
    #                                 features_computer=features_computer,
    #                                 representations=[graph_rep])
    # g1 = supervised_dataset[0]
    # a = list(g1['rna'].nodes(data=True))[0][1]

    # This instead uses from_args, hence features_computer is called during dataset preparation, which saves spaces
    # supervised_dataset = RNADataset.from_args(all_graphs_db=all_graphs,
    #                                           features_computer=features_computer,
    #                                           representations=[graph_rep])
    # g2 = supervised_dataset[0]
    # b = list(g2['rna'].nodes(data=True))[0][1]

    # Test dumping/loading
    # This instead uses from_args, hence features_computer is called during dataset preparation, which saves spaces
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(script_dir, "../data/test")
    rnas = build_dataset(all_graphs_db=all_graphs,
                         dataset_path=dataset_path,
                         features_computer=features_computer)
    supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep, all_graphs=all_graphs)
    g2 = supervised_dataset[0]
    b = list(g2['rna'].nodes(data=True))[0][1]
    a = 1
