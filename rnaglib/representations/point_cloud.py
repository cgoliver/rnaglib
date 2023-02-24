import torch
import numpy as np

from rnaglib.representations import Representation


def get_point_cloud_dict(rna_graph, features_dict, sort=False):
    """
    This is factored out because this computation is also used by the voxel based representation.

    :param rna_graph:
    :param features_dict:
    :return:
    """
    node_names = []
    res_dict = {'point_cloud_coords': []}
    if "nt_features" in features_dict:
        res_dict['point_cloud_feats'] = []
    if "nt_targets" in features_dict:
        res_dict['point_cloud_targets'] = []

    node_iterator = rna_graph.nodes.data()
    node_iterator = sorted(node_iterator) if sort else node_iterator
    for node, attrs in node_iterator:
        node_names.append(node)
        node_coords = attrs['C5prime_xyz']
        node_coords = torch.as_tensor(np.array(node_coords, dtype=float))
        res_dict['point_cloud_coords'].append(node_coords)
        if "nt_features" in features_dict:
            res_dict['point_cloud_feats'].append(features_dict['nt_features'][node])
        if "nt_targets" in features_dict:
            res_dict['point_cloud_targets'].append(features_dict['nt_targets'][node])

    # for key, value in res_dict.items():
    #     print(key, [val.shape for val in value])
    stacked_res_dict = {key: torch.stack(value, dim=0) for key, value in res_dict.items()}
    stacked_res_dict['point_cloud_nodes'] = node_names
    return stacked_res_dict


class PointCloudRepresentation(Representation):
    """ Converts and RNA into a voxel based representation """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        pass

    def __call__(self, rna_graph, features_dict):
        return get_point_cloud_dict(rna_graph=rna_graph, features_dict=features_dict)

    @property
    def name(self):
        return "point_cloud"

    def batch(self, samples):
        """
        Just return the name of the representation

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        raise NotImplementedError
