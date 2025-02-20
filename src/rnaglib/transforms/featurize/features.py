"""Cast annotations to feature tensors."""

from typing import Dict, Union, List, TYPE_CHECKING, Literal

import torch
import networkx as nx

from rnaglib.config import NODE_FEATURE_MAP, EDGE_FEATURE_MAP
from rnaglib.transforms import Transform


class FeaturesComputer(Transform):
    """
    This class takes as input an RNA in the networkX form
    and computes the ``features_dict`` which maps node IDs to a tensor of features.
    The ``features_dict`` contains keys: ``'nt_features'``for node features,
    ``'nt_targets'`` for node-level prediction targets. In :class:`~rnaglib.data_loading.RNADataset` construction,
    the ``FeaturesComputer.compute_features()`` method is called during the ``RNADataset`` ``__getitem__()`` call.

    :param nt_features: List of keys to use as node features, choose from the `dataset[i]['rna']` node attributes dictionary.
    :param nt_targets: List of keys to use as node features, choose from the `dataset[i]['rna']` node attributes dictionary.
    :param rna_features:
    :param rna_targets:
    :param bp_features:
    :param bp_targets:
    :param post_transform:
    :param extra_useful_keys:
    """

    def __init__(
        self,
        nt_features: Union[List, str] = None,
        nt_targets: Union[List, str] = None,
        rna_features: Union[List, str] = None,
        rna_targets: Union[List, str] = None,
        bp_features: Union[List, str] = None,
        bp_targets: Union[List, str] = None,
        extra_useful_keys: Union[List, str] = None,
        custom_encoders: dict = None,
    ):

        self.rna_features_parser = self.build_feature_parser(rna_features, custom_encoders=custom_encoders)
        self.rna_targets_parser = self.build_feature_parser(rna_targets, custom_encoders=custom_encoders)
        self.node_features_parser = self.build_feature_parser(nt_features, custom_encoders=custom_encoders)
        self.node_targets_parser = self.build_feature_parser(nt_targets, custom_encoders=custom_encoders)

        # This is only useful when using a FeatureComputer to create a dataset, and avoid removing important features
        # of the graph that are not used during loading
        self.extra_useful_keys = extra_useful_keys

        # experimental
        self.rna_features = rna_features
        self.rna_targets = rna_targets
        self.bp_features = bp_features
        self.bp_targets = bp_targets

    def add_feature(
        self,
        feature_names=None,
        custom_encoders=None,
        input_feature=True,
        feature_level: Literal["rna", "residue"] = "residue",
    ):
        """
        Update the input/output feature selector with either an extra available named feature or a custom encoder
        :param feature_names: Name of the input feature to add
        :param transforms: A Transform object to compute new features with
        :param feature_level: If featureis RNA-level ('rna`) or residue-level (`residue`)
        :param input_feature: Set to true to modify the input feature encoder, false for the target one
        :return: None
        """
        # Select the right node_parser and update it

        if feature_level == "residue":
            old_parser = self.node_features_parser if input_feature else self.node_target_parser
        elif feature_level == "rna":
            old_parser = self.rna_features_parser if input_feature else self.rna_target_parser
        else:
            raise ValueError(f"Invalid feature level {feature_level}, must be 'rna' or 'residue'")

        new_parser = self.build_feature_parser(asked_features=feature_names, custom_encoders=custom_encoders)
        old_parser.update(new_parser)

    def remove_feature(self, feature_name=None, input_feature=True):
        """
        Update the input/output feature selector with either an extra available named feature or a custom encoder
        :param feature_name: Name of the input feature to remove
        :param input_feature: Set to true to modify the input feature encoder, false for the target one
        :return: None
        """
        if not isinstance(feature_name, list):
            feature_name = [feature_name]

        # Select the right node_parser and update it
        node_parser = self.node_features_parser if input_feature else self.node_target_parser
        filtered_node_parser = {k: node_parser[k] for k in node_parser if not k in feature_name}
        if input_feature:
            self.node_features_parser = filtered_node_parser
        else:
            self.node_target_parser = filtered_node_parser

    @staticmethod
    def compute_dim(node_parser):
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

    @property
    def input_dim(self):
        return self.compute_dim(self.node_features_parser)

    @property
    def output_dim(self):
        return self.compute_dim(self.node_target_parser)

    def remove_useless_keys(self, rna_graph):
        """
        Copy the original graph to only retain keys relevant to this FeaturesComputer
        :param rna_graph:
        :return:
        """
        useful_keys = set(self.node_features_parser.keys()).union(set(self.node_target_parser.keys()))
        if self.extra_useful_keys is not None:
            useful_keys = useful_keys.union(set(self.extra_useful_keys))
        cleaned_graph = nx.DiGraph(name=rna_graph.name)
        cleaned_graph.add_edges_from(rna_graph.edges(data=True))
        for key in useful_keys:
            val = nx.get_node_attributes(rna_graph, key)
            nx.set_node_attributes(cleaned_graph, name=key, values=val)
        return cleaned_graph

    @staticmethod
    def encode_rna(g: nx.Graph, parser):
        """
        Simply apply the rna encoding functions in ``parser`` for all features.
        Then use torch.cat over the result to get a tensor for each node in the graph.

        :param g: a nx graph
        :param node_parser: {feature_name : encoder}
        :return: A dict that maps nodes to encodings

        """

        if len(parser) == 0:
            return None

        all_feature_encoding = list()
        for i, (feature, feature_encoder) in enumerate(parser.items()):
            try:
                feature_encoding = feature_encoder.encode(g.graph[feature])
            except KeyError:
                feature_encoding = feature_encoder.encode_default()
            all_feature_encoding.append(feature_encoding)
        encodings = torch.cat(all_feature_encoding) if len(all_feature_encoding) > 1 else all_feature_encoding[0]
        return encodings

    @staticmethod
    def encode_nodes(g: nx.Graph, node_parser):
        """
        Simply apply the node encoding functions in node_parser to each node in the graph
        Then use torch.cat over the result to get a tensor for each node in the graph.

        :param g: a nx graph
        :param node_parser: {feature_name : encoder}
        :return: A dict that maps nodes to encodings

        """

        node_encodings = {}
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

    def build_feature_parser(
        self,
        asked_features: Union[List, str] = None,
        custom_encoders: dict = None,
        feature_map: dict = None,
    ) -> dict:
        """
        This function will load the predefined feature maps available globally.
        Then for each of the features in 'asked feature', it will return an encoder object for each of the asked features
        in the form of a dict {asked_feature : EncoderObject}

        If some keys don't exist, will raise an Error. However if some keys are present but problematic,
        this will just cause a printing of the problematic keys
        :param asked_features: A list of string keys that are present in the encoder
        :param transforms: Transform objects to compute extra features
        :param feature_map: Dictionary mapping feature key to an Encoder() object.
        :return: A dict {asked_feature : EncoderObject}
        """

        if asked_features is None:
            return {}

        # default to node-feature map
        if feature_map is None:
            feature_map = {**NODE_FEATURE_MAP, **EDGE_FEATURE_MAP}
        else:
            feature_map = feature_map.copy()
        # Build an asked list of features, with no redundancies
        asked_features = [] if asked_features is None else asked_features
        if not isinstance(asked_features, list):
            asked_features = [asked_features]
        # Make a non redundant list that keeps the features order
        nr_asked_feature = []
        for item in asked_features:
            if item not in nr_asked_feature:
                nr_asked_feature.append(item)

        # attach the transform's encoder
        if custom_encoders is not None:
            for feature, encoder in custom_encoders.items():
                feature_map[feature] = encoder

        # Update the map {key:encoder} and ensure every asked feature is in this encoding map.
        if any([feature not in feature_map for feature in asked_features]):
            problematic_keys = tuple([feature for feature in asked_features if feature not in feature_map])
            raise ValueError(f"{problematic_keys} were asked as a feature or target but do not exist")

        # Filter out None encoder functions, we don't know how to encode those...
        encoding_features = [feature for feature in asked_features if feature_map[feature] is not None]
        if len(encoding_features) < len(asked_features):
            unencodable_keys = [feature for feature in asked_features if feature_map[feature] is None]
            print(f"{unencodable_keys} were asked as a feature or target but do not exist")

        # Finally, keep only the relevant keys to include in the encoding dict.
        subset_dict = {k: feature_map[k] for k in encoding_features}
        return subset_dict

    def build_edge_feature_parser(self, asked_features=None):
        raise NotImplementedError

    def forward(self, rna_dict: Dict):
        """
        Add 3 dictionaries to the `rna_dict` wich maps nts, edges, and the whole graph
        to a feature vector each. The final converter uses these to include the data in the
        framework-specific object.
        """

        features_dict = {}
        if len(self.rna_features_parser) > 0:
            rna_feature_encoding = self.encode_rna(rna_dict["rna"], parser=self.rna_features_parser)
            features_dict["rna_features"] = rna_feature_encoding

        if len(self.rna_targets_parser) > 0:
            rna_targets_encoding = self.encode_rna(rna_dict["rna"], parser=self.rna_targets_parser)
            features_dict["rna_targets"] = rna_targets_encoding

        # Get Node labels
        if len(self.node_features_parser) > 0:
            feature_encoding = self.encode_nodes(
                rna_dict["rna"],
                node_parser=self.node_features_parser,
            )
            features_dict["nt_features"] = feature_encoding
        if len(self.node_targets_parser) > 0:
            target_encoding = self.encode_nodes(
                rna_dict["rna"],
                node_parser=self.node_targets_parser,
            )
            features_dict["nt_targets"] = target_encoding
        return features_dict
