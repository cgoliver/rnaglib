import os
import sys

import copy
import torch
import networkx as nx

from rnaglib.utils import build_node_feature_parser
from rnaglib.utils import download_graphs
from rnaglib.utils import load_graph
from rnaglib.utils import dump_json


class RNADataset:
    """ 
        This class is the main object to hold the core RNA data annotations.
        The ``RNAglibDataset.all_rnas`` object is a generator networkx objects that hold all the annotations for each RNA in the dataset.
        You can also access individual RNAs on-disk with ``RNAGlibDataset()[idx]`` or ``RNAGlibDataset().get_pdbid('1b23')``

    """

    def __init__(self,
                 db_path=None,
                 saved_dataset=None,
                 version='1.0.0',
                 download_dir=None,
                 redundancy='nr',
                 all_graphs=None,
                 representations=None,
                 rna_features=None,
                 nt_features=None,
                 bp_features=None,
                 rna_targets=None,
                 nt_targets=None,
                 bp_targets=None,
                 custom_encoders=None,
                 annotated=False,
                 verbose=False,
                 annotator=None,
                 nt_filter=None,
                 rna_filter=None,
                 ):
        """


        :param representations: List of `rnaglib.Representation` objects to apply to each item.
        :param db_path: The path to the folder containing the graphs. If node_sim is not None, this data should be annotated
        :param saved_dataset: Path to an already saved dataset, skips dataset creation if loaded.
        :param version: Version of the dataset to use (default='0.0.0')
        :param redundancy: To use all graphs or just the non redundant set.
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use
        :param rna_filter: Callable which accepts an RNA dictionary and returns a new RNA dictionary with fewer nodes.
        :param annotator: Callable which takes as input an RNA dictionary and adds new key-value pairs.

        """

        # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched
        # By default, we set hashing to None and potential node sim should be specified when creating
        # the node_sim function.
        # Then if a download occurs and no hashing was provided to the loader, the hashing used is the one
        # fetched by the downloading process to ensure it matches the data we iterate over.
        if representations is None:
            self.representations = []
        else:
            self.representations = representations

        self.db_path = db_path

        # DB_path corresponds to all available RNA graphs in rnaglib
        if db_path is None:
            self.db_path = download_graphs(redundancy=redundancy,
                                           version=version,
                                           annotated=annotated,
                                           data_root=download_dir,
                                           )

            self.db_path = os.path.join(self.db_path, 'graphs')

        # One can restrict the number of graphs to use
        if all_graphs is None:
            self.all_graphs = sorted(os.listdir(self.db_path))
        else:
            self.all_graphs = all_graphs

        # Maybe we precomputed subsets of the db already or we want to; this is what saved_dataset is here for
        self.saved_dataset = saved_dataset

        self.rna_features = rna_features
        self.rna_targets = rna_targets
        self.nt_features = nt_features
        self.nt_targets = nt_targets
        self.bp_features = bp_features
        self.bp_targets = bp_targets

        self.node_features_parser = build_node_feature_parser(self.nt_features,
                                                              custom_encoders=custom_encoders
                                                              )
        self.node_target_parser = build_node_feature_parser(self.nt_targets)

        self.input_dim = self.compute_dim(self.node_features_parser)
        self.output_dim = self.compute_dim(self.node_target_parser)

        self.available_pdbids = [g.split(".")[0].lower() for g in self.all_graphs]

        if rna_filter is None:
            self.rna_filter = lambda x: True
        else:
            self.rna_filter = rna_filter

        self.nt_filter = nt_filter

        self.annotator = annotator

        self.rnas = self._build_dataset()

    def __len__(self):
        return len(self.rnas)

    def _build_dataset(self):
        if not self.saved_dataset is None:
            return [load_graph(os.path.join(self.saved_dataset, g_name)) \
                    for g_name in os.listdir(self.saved_dataset)]
        else:
            return self.build_dataset()
        pass

    def build_dataset(self):
        """ Iterates through database, applying filters and annotations"""
        graph_list = []
        for graph_name in self.all_graphs:
            g_path = os.path.join(self.db_path, graph_name)
            g = load_graph(g_path)

            if not self.rna_filter(g):
                continue
            if not self.nt_filter is None:
                subgs = []

                for subg in self.nt_filter(g):
                    subgs.append(subg)
            else:
                subgs = [g]
            if not self.annotator is None:
                for subg in subgs:
                    self.annotator(subg)
            graph_list.extend(subgs)
        return graph_list

    def save(self, dump_path):
        """ Save a local copy of the dataset"""
        for i, rna in enumerate(self.rnas):
            dump_json(os.path.join(dump_path, f"{i}.json"), rna)

    def __getitem__(self, idx):
        """ Fetches one RNA and converts it from raw data to a dictionary
        with representations and annotations to be used by loaders """

        rna_graph = self.rnas[idx]

        rna_dict = {'rna': rna_graph}
        features_dict = self.compute_features(rna_dict)
        # apply representations to the res_dict
        # each is a callable that updates the res_dict
        for rep in self.representations:
            rna_dict[rep.name] = rep(rna_graph, features_dict)
        return rna_dict

    def select(self):
        return [self[i] for i in range(len(self))]

    def add_representation(self, representation):
        self.representations.append(representation)

    def remove_representation(self, name):
        self.representations = [representation for representation in self.representations if
                                representation.name != name]

    def subset(self, list_of_ids):
        """
        Create another dataset with only the specified graphs

        :param list_of_graphs: a list of graph names
        :return: A graphdataset
        """
        subset = copy.deepcopy(self)
        subset.rnas = [self.rnas[i] for i in list_of_ids]
        return subset

    def get_pdbid(self, pdbid):
        """ Grab an RNA by its pdbid """
        return self.__getitem__(self.available_pdbids.index(pdbid.lower()))

    def get_nt_encoding(self, g, encode_feature=True):
        """

        Get targets for graph g
        for every node get the attribute specified by self.node_target
        output a mapping of nodes to their targets

        :param g: a nx graph
        :param encode_feature: A boolean as to whether this should encode the features or targets
        :return: A dict that maps nodes to encodings
        
        """
        node_encodings = {}
        node_parser = self.node_features_parser if encode_feature else self.node_target_parser

        if len(node_parser) == 0:
            return None

        for node, attrs in g.nodes.data():
            all_node_feature_encoding = list()
            for i, (feature, feature_encoder) in enumerate(node_parser.items()):
                try:
                    node_feature = attrs[feature]
                    node_feature_encoding = feature_encoder.encode(node_feature)
                except KeyError:
                    node_feature_encoding = feature_encoder.encode_default()
                all_node_feature_encoding.append(node_feature_encoding)
            node_encodings[node] = torch.cat(all_node_feature_encoding)
        return node_encodings

    def compute_dim(self, node_parser):
        """
        Based on the encoding scheme, we can compute the shapes of the in and out tensors

        :return:

        """
        if len(node_parser) == 0:
            return 0
        all_node_feature_encoding = list()
        for i, (feature, feature_encoder) in enumerate(node_parser.items()):
            node_feature_encoding = feature_encoder.encode_default()
            all_node_feature_encoding.append(node_feature_encoding)
        all_node_feature_encoding = torch.cat(all_node_feature_encoding)
        return len(all_node_feature_encoding)

    def compute_features(self, rna_dict):
        """ Add 3 dictionaries to the `rna_dict` wich maps nts, edges, and the whole graph
        to a feature vector each. The final converter uses these to include the data in the
        framework-specific object.

        """

        graph = rna_dict['rna']
        features_dict = {}

        # Get Node labels
        if len(self.node_features_parser) > 0:
            feature_encoding = self.get_nt_encoding(graph, encode_feature=True)
            features_dict['nt_features'] = feature_encoding
        if len(self.node_target_parser) > 0:
            target_encoding = self.get_nt_encoding(graph, encode_feature=False)
            features_dict['nt_targets'] = target_encoding
        return features_dict
