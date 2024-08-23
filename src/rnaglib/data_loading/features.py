import torch
import networkx as nx
from rnaglib.utils import build_node_feature_parser


class FeaturesComputer:
    """
    This class can be seen as a Transforms class, taking as input an RNA in the networkX form,

    """

    def __init__(self,
                 nt_features=None,
                 nt_targets=None,
                 custom_encoders_features=None,
                 custom_encoders_targets=None,
                 rna_features=None,
                 rna_targets=None,
                 bp_features=None,
                 bp_targets=None,
                 misc_encoder=None,
                 extra_useful_keys=None,
                 ):
        self.node_features_parser = build_node_feature_parser(nt_features,
                                                              custom_encoders=custom_encoders_features)
        self.node_target_parser = build_node_feature_parser(nt_targets,
                                                            custom_encoders=custom_encoders_targets)

        # For all special cases, this is what to use
        self.misc_encoder = misc_encoder

        # This is only useful when using a FeatureComputer to create a dataset, and avoid removing important features
        # of the graph that are not used during loading
        self.extra_useful_keys = extra_useful_keys

        # experimental
        self.rna_features = rna_features
        self.rna_targets = rna_targets
        self.bp_features = bp_features
        self.bp_targets = bp_targets

    def add_feature(self, feature_names=None, custom_encoders=None, input_feature=True):
        """
        Update the input/output feature selector with either an extra available named feature or a custom encoder
        :param feature_names: Name of the input feature to add
        :param custom_encoders: A dict containing {named_feature: custom encoder}
        :param input_feature: Set to true to modify the input feature encoder, false for the target one
        :return: None
        """
        # Select the right node_parser and update it
        node_parser = self.node_features_parser if input_feature else self.node_target_parser
        new_node_parser = build_node_feature_parser(asked_features=feature_names,
                                                    custom_encoders=custom_encoders)
        node_parser.update(new_node_parser)

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
    def encode_nodes(g, node_parser):
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

    def compute_features(self, rna_graph):
        """
        Add 3 dictionaries to the `rna_dict` wich maps nts, edges, and the whole graph
        to a feature vector each. The final converter uses these to include the data in the
        framework-specific object.
        """

        features_dict = {}
        # Get Node labels
        if len(self.node_features_parser) > 0:
            feature_encoding = self.encode_nodes(rna_graph, node_parser=self.node_features_parser)
            features_dict['nt_features'] = feature_encoding
        if len(self.node_target_parser) > 0:
            target_encoding = self.encode_nodes(rna_graph, node_parser=self.node_target_parser)
            features_dict['nt_targets'] = target_encoding
        if self.misc_encoder is not None:
            self.misc_encoder(rna_graph, features_dict)
        return features_dict
