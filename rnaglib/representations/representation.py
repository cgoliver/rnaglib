import torch

from rnaglib.utils import build_node_feature_parser

FRAMEWORKS = ['dgl', 'torch', 'pyg', 'nx']

class Representation:
    """ Callable object that accepts a raw RNA networkx object
    and returns a representation of it (e.g. graph, voxel, point cloud)
    along with necessary nucleotide / base pair features """
    def __init__(self,
                 rna_features=None,
                 nt_features=None,
                 bp_features=None,
                 rna_targets=None,
                 nt_targets=None,
                 bp_targets=None,
                 framework='dgl'
                 ):

        self.check_framework(self.framework)
        self.framework = framework

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


    def __call__(self, rna_dict):
        """ This function is applied to each RNA in the dataset and updates
        `rna_dict`"""
        self.compute_features(rna_dict)
        self.call(rna_dict)

    def call(self, rna_dict):
        raise NotImplementedError

    def get_nt_encoding(self, g, encode_feature=True):
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
        framework-specific object."""

        graph = rna_dict['rna']

        # Get Node labels
        node_attrs_toadd = list()
        if len(self.node_features_parser) > 0:
            feature_encoding = self.get_nt_encoding(graph, encode_feature=True)
            rna_dict['nt_features'] = feature_encoding
        if len(self.node_target_parser) > 0:
            target_encoding = self.get_nt_encoding(graph, encode_feature=False)
            rna_dict['nt_targets'] = target_encoding

    def check_framework(self, framework):
        assert framework in self.frameworks, f"Framework {framework} not supported for this representation. Choose one of {self.frameworks}."
