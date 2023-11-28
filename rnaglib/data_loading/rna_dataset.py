import os
import sys

import copy
import torch
import networkx as nx

from rnaglib.utils import build_node_feature_parser
from rnaglib.utils import download_graphs
from rnaglib.utils import load_graph


class RNADataset:
    """ 
        This class is the main object to hold the core RNA data annotations.
        The ``RNAglibDataset.all_rnas`` object is a generator networkx objects that hold all the annotations for each RNA in the dataset.
        You can also access individual RNAs on-disk with ``RNAGlibDataset()[idx]`` or ``RNAGlibDataset().get_pdbid('1b23')``

    """

    def __init__(self,
                 data_path=None,
                 version='1.0.0',
                 download_dir=None,
                 redundancy='nr',
                 all_graphs=None,
                 representations=(),
                 rna_features=None,
                 nt_features=None,
                 bp_features=None,
                 rna_targets=None,
                 nt_targets=None,
                 bp_targets=None,
                 annotated=False,
                 verbose=False):
        """


        :param representations: List of `rnaglib.Representation` objects to apply to each item.
        :param data_path: The path to the folder containing the graphs. If node_sim is not None, this data should be annotated
        :param version: Version of the dataset to use (default='0.0.0')
        :param redundancy: To use all graphs or just the non redundant set.
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use

        """

        # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched
        # By default, we set hashing to None and potential node sim should be specified when creating
        # the node_sim function.
        # Then if a download occurs and no hashing was provided to the loader, the hashing used is the one
        # fetched by the downloading process to ensure it matches the data we iterate over.
        self.representations = representations
        self.data_path = data_path

        if data_path is None:
            self.data_path = download_graphs(redundancy=redundancy,
                                             version=version,
                                             annotated=annotated,
                                             data_root=download_dir,
                                             )

            self.data_path = os.path.join(self.data_path, 'graphs')
        if all_graphs is not None:
            self.all_graphs = all_graphs
        else:
            self.all_graphs = sorted(os.listdir(self.data_path))

        self.rna_features = rna_features
        self.rna_targets = rna_targets
        self.nt_features = nt_features
        self.nt_targets = nt_targets
        self.bp_features = bp_features
        self.bp_targets = bp_targets

        self.node_features_parser = build_node_feature_parser(self.nt_features)
        self.node_target_parser = build_node_feature_parser(self.nt_targets)

        self.input_dim = self.compute_dim(self.node_features_parser)
        self.output_dim = self.compute_dim(self.node_target_parser)

        self.available_pdbids = [g.split(".")[0].lower() for g in self.all_graphs]

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        """ Fetches one RNA and converts it from raw data to a dictionary
        with representations and annotations to be used by loaders """

        g_path = os.path.join(self.data_path, self.all_graphs[idx])
        rna_graph = load_graph(g_path)
        rna_dict = {'rna_name': self.all_graphs[idx],
                    'rna': rna_graph,
                    'path': g_path
                    }
        features_dict = self.compute_features(rna_dict)
        # apply representations to the res_dict
        # each is a callable that updates the res_dict
        for rep in self.representations:
            rna_dict[rep.name] = rep(rna_graph, features_dict)
        return rna_dict

    def add_representation(self, representation):
        self.representations.append(representation)

    def remove_representation(self, name):
        self.representations = [representation for representation in self.representations if
                                representation.name != name]

    def subset(self, list_of_graphs):
        """
        Create another dataset with only the specified graphs

        :param list_of_graphs: a list of graph names
        :return: A graphdataset
        """
        subset = copy.deepcopy(self)
        subset.all_graphs = list(set(list_of_graphs).intersection(set(self.all_graphs)))
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
