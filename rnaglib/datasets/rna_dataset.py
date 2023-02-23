import os
import sys

import copy
import torch
import networkx as nx

from rnaglib.utils import build_node_feature_parser
from rnaglib.utils import download_graphs
from rnaglib.utils import load_graph

class RNADataset:
    def __init__(self,
                 data_path=None,
                 version='0.0.0',
                 download_dir=None,
                 redundancy='nr',
                 all_graphs=None,
                 representations=None,
                 annotated=False,
                 node_features='nt_code',
                 node_target=None,
                 verbose=False):
        """
        This class is the main object to hold the core RNA data annotations.
        The `RNAglibDataset.all_rnas` object is a generator networkx objects that hold all the annotations for each RNA in the dataset.
        You can also access individual RNAs on-disk with `RNAGlibDataset()[idx]` or `RNAGlibDataset().get_pdbid('1b23')`

        :param representations: List of `rnaglib.Representation` objects to apply to each item.
        :param data_path: The path to the folder containing the graphs. If node_sim is not None, this data should be annotated
        :param version: Version of the dataset to use (default='0.0.0')
        :param redundancy: To use all graphs or just the non redundant set.
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use
        :return:
        """

        # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched
        # By default, we set hashing to None and potential node sim should be specified when creating
        # the node_sim function.
        # Then if a download occurs and no hashing was provided to the loader, the hashing used is the one
        # fetched by the downloading process to ensure it matches the data we iterate over.
        self.data_path = data_path
        self.representations = representations
        if data_path is None:
            self.data_path = download_graphs(redundancy=redundancy,
                                             version=version,
                                             annotated=annotated,
                                             data_root=download_dir,
                                           )

            self.graphs_path = os.path.join(self.data_path, 'graphs')
        if all_graphs is not None:
            self.all_graphs = all_graphs
        else:
            self.all_graphs = sorted(os.listdir(self.graphs_path))

        self.node_features = [node_features] if isinstance(node_features, str) else node_features
        self.node_target = [node_target] if isinstance(node_target, str) else node_target
        self.node_features_parser = build_node_feature_parser(self.node_features)
        self.node_target_parser = build_node_feature_parser(self.node_target)

        self.available_pdbids = [g.split(".")[0].lower() for g in self.all_graphs]
        self.rnas = (self[i] for i in range(len(self.available_pdbids)))

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        """ Fetches one RNA and converts it from raw data to a dictionary
        with representations and annotations to be used by loaders """

        g_path = os.path.join(self.graphs_path, self.all_graphs[idx])
        rna_dict = {'rna_name': self.all_graphs[idx],
                    'rna': load_graph(g_path),
                    'path': g_path
                    }
        # apply representations to the res_dict
        # each is a callable that updates the res_dict
        [r(rna_dict) for r in self.representations]
        return rna_dict


    def get_pdbid(self, pdbid):
        """ Grab an RNA by its pdbid """
        return self.get(self.available_pdbids.index(pdbid.lower()))


    def get_node_encoding(self, g, encode_feature=True):
        """

        Get targets for graph g
        for every node get the attribute specified by self.node_target
        output a mapping of nodes to their targets

        :param g: a nx graph
        :param encode_feature: A boolean as to whether this should encode the features or targets
        :return: A dict that maps nodes to encodings
        """
        targets = {}
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
            targets[node] = torch.cat(all_node_feature_encoding)
        return targets

