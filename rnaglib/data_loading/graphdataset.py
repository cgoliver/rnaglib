import os
import sys

import copy
import networkx as nx
import numpy as np
from sklearn.gaussian_process.kernels import RBF

import torch
from torch.utils.data import Dataset
import dgl

from rnaglib.utils import download_graphs
from rnaglib.utils import load_graph
from rnaglib.data_loading.feature_maps import build_node_feature_parser
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL


def get_bins(coords, spacing, padding, xyz_min=None, xyz_max=None):
    """
    Compute the 3D bins from the coordinates
    """
    if xyz_min is None:
        xm, ym, zm = np.nanmin(coords, axis=0) - padding
    else:
        xm, ym, zm = xyz_min - padding
    if xyz_max is None:
        xM, yM, zM = np.nanmax(coords, axis=0) + padding
    else:
        xM, yM, zM = xyz_max + padding

    # print(xm)
    # print(xM)
    # print(spacing)
    xi = np.arange(xm, xM, spacing)
    yi = np.arange(ym, yM, spacing)
    zi = np.arange(zm, zM, spacing)
    return xi, yi, zi


def just_one(coord, xi, yi, zi, sigma, feature, total_grid, use_multiprocessing=False):
    """

    :param coord: x,y,z
    :param grid:
    :param sigma:
    :return:
    """
    #  Find subgrid
    nx, ny, nz = xi.size, yi.size, zi.size

    bound = int(4 * sigma)
    x, y, z = coord
    binx = np.digitize(x, xi)
    biny = np.digitize(y, yi)
    binz = np.digitize(z, zi)
    min_bounds_x, max_bounds_x = max(0, binx - bound), min(nx, binx + bound)
    min_bounds_y, max_bounds_y = max(0, biny - bound), min(ny, biny + bound)
    min_bounds_z, max_bounds_z = max(0, binz - bound), min(nz, binz + bound)

    X, Y, Z = np.meshgrid(xi[min_bounds_x: max_bounds_x],
                          yi[min_bounds_y: max_bounds_y],
                          zi[min_bounds_z:max_bounds_z],
                          indexing='ij')
    X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()

    #  Compute RBF
    rbf = RBF(sigma)
    subgrid = rbf(coord, np.c_[X, Y, Z])
    subgrid = subgrid.reshape((max_bounds_x - min_bounds_x,
                               max_bounds_y - min_bounds_y,
                               max_bounds_z - min_bounds_z))

    # Broadcast the feature throughout the local grid.
    subgrid = subgrid[None, ...]
    feature = feature[:, None, None, None]
    subgrid_feature = subgrid * feature

    #  Add on the first grid
    if not use_multiprocessing:
        total_grid[:, min_bounds_x: max_bounds_x, min_bounds_y: max_bounds_y,
        min_bounds_z:max_bounds_z] += subgrid_feature
    else:
        return min_bounds_x, max_bounds_x, min_bounds_y, max_bounds_y, min_bounds_z, max_bounds_z, subgrid_feature


def gaussian_blur(coords, xi, yi, zi, features=None, sigma=1., use_multiprocessing=False):
    """

    :param coords: (n_points, 3)
    :param xi:
    :param yi:
    :param zi:
    :param features: (n_points, dim) or None
    :param sigma:
    :param use_multiprocessing:
    :return:
    """

    nx, ny, nz = xi.size, yi.size, zi.size
    features = np.ones((len(coords), 1)) if features is None else features
    feature_len = features.shape[1]
    total_grid = np.zeros(shape=(feature_len, nx, ny, nz))

    if use_multiprocessing:
        import multiprocessing
        args = [(coord, xi, yi, zi, sigma, features[i], None, True) for i, coord in enumerate(coords)]
        pool = multiprocessing.Pool()
        grids_to_add = pool.starmap(just_one, args)
        for min_bounds_x, max_bounds_x, min_bounds_y, max_bounds_y, min_bounds_z, max_bounds_z, subgrid in grids_to_add:
            total_grid[:, min_bounds_x: max_bounds_x, min_bounds_y: max_bounds_y, min_bounds_z:max_bounds_z] += subgrid
    else:
        for i, coord in enumerate(coords):
            just_one(coord, feature=features[i], xi=xi, yi=yi, zi=zi, sigma=sigma, total_grid=total_grid)
    return total_grid


def get_grid(coords, features=None, spacing=2, padding=3, xyz_min=None, xyz_max=None, sigma=1.):
    """
    Generate a grid from the coordinates
    :param coords: (n,3) array
    :param features: (n,k) array
    :param spacing:
    :param padding:
    :param xyz_min:
    :param xyz_max:
    :param sigma:
    :return:
    """
    xi, yi, zi = get_bins(coords, spacing, padding, xyz_min, xyz_max)
    grid = gaussian_blur(coords, xi, yi, zi, features=features, sigma=sigma)
    return grid


class GraphDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 version='0.0.0',
                 download_dir=None,
                 redundancy='nr',
                 chop=False,
                 all_graphs=None,
                 node_features='nt_code',
                 node_target=None,
                 return_type=('graph'),
                 node_simfunc=None,
                 hashing_path=None,
                 edge_map=GRAPH_KEYS['edge_map'][TOOL],
                 label='LW',
                 verbose=False):
        """
        This class is the main object for graph data loading. One can simply ask for feature and the appropriate data
        will be fetched.

        :param data_path: The path to the folder containing the graphs. If node_sim is not None, this data should be annotated
        :param hashing_path: If node_sim is not None, we need hashing tables. If the path is not automatically created
        (ie the data was downloaded manually) one should input the path to the hashing.
        :param download_dir: When one fetches the data, one can choose where to dl it.
        By default, it will go to ~/.rnaglib/
        :param redundancy: To use all graphs or just the non redundant set.
        :param chop: if we want full graphs or chopped ones for learning on smaller chunks
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use
        :param node_features: node features to include, stored in one tensor in order given by user,
        for example : ('nt_code','is_modified')
        :param node_target: node targets to include, stored in one tensor in order given by user
        for example : ('binding_protein', 'binding_small-molecule')
        :param return_type: string or iterable of a subset from {'graph', 'point_cloud', 'voxel'}. This determines what
        is included in the output from get_item and serves as keys to the dictionnary.
        :param edge_map: Necessary to build the one hot mapping from edge labels to an id
        :param label: The label to use for the edges in the graph
        :param node_simfunc: The node comparison object as defined in kernels/node_sim to use for the embeddings.
         If None is selected, this will just return graphs
        :return:
        """

        # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched
        # By default, we set hashing to None and potential node sim should be specified when creating
        # the node_sim function.
        # Then if a download occurs and no hashing was provided to the loader, the hashing used is the one
        # fetched by the downloading process to ensure it matches the data we iterate over.
        self.data_path = data_path
        self.hashing_path = hashing_path
        if data_path is None:
            self.data_path = download_graphs(redundancy=redundancy,
                                             version=version,
                                             annotated=node_simfunc is not None,
                                             data_root=download_dir,
                                           )

            self.graph_path = os.path.join(self.data_path, 'graphs')
        if all_graphs is not None:
            self.all_graphs = all_graphs
        else:
            self.all_graphs = sorted(os.listdir(self.graph_path))

        self.return_type = [return_type] if isinstance(return_type, str) else return_type

        # If queried, add node features and node targets
        self.node_features = [node_features] if isinstance(node_features, str) else node_features
        self.node_target = [node_target] if isinstance(node_target, str) else node_target

        self.node_features_parser = build_node_feature_parser(self.node_features)
        self.node_target_parser = build_node_feature_parser(self.node_target)

        self.input_dim = self.compute_dim(self.node_features_parser)
        self.output_dim = self.compute_dim(self.node_target_parser)

        self.node_simfunc = None
        if 'graph' in self.return_type:
            # This is len() so we have to add the +1
            self.label = label
            self.edge_map = edge_map
            self.num_edge_types = max(self.edge_map.values()) + 1
            if verbose:
                print(f"Found {self.num_edge_types} relations")
            # If it is not None, add a node comparison tool
            self.node_simfunc, self.level = self.add_node_sim(node_simfunc=node_simfunc)

    def __len__(self):
        return len(self.all_graphs)

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

    def get_nx_graph(self, idx):
        """
        Load the correct graph and embed its features and/or target into one_hot vectors.
        Return this graph along with a list containing a subset of {'features', 'target'} based on what was to encode.
        :param idx:
        :return:
        """
        g_path = os.path.join(self.graph_path, self.all_graphs[idx])
        graph = load_graph(g_path)

        # Get Node labels
        node_attrs_toadd = list()
        if len(self.node_features_parser) > 0:
            feature_encoding = self.get_node_encoding(graph, encode_feature=True)
            nx.set_node_attributes(graph, name='features', values=feature_encoding)
            node_attrs_toadd.append('features')
        if len(self.node_target_parser) > 0:
            target_encoding = self.get_node_encoding(graph, encode_feature=False)
            nx.set_node_attributes(graph, name='target', values=target_encoding)
            node_attrs_toadd.append('target')
        return graph, node_attrs_toadd

    def as_tensor(self, graph, node_feature, sort=False):
        """
        Flattens graph attributes as a tensor : (num_node, dim_feature)
        :param graph:
        :param node_feature:
        :param sort:
        :return:
        """
        iterator = graph.nodes.data()
        iterator = sorted(iterator) if sort else iterator
        tensor_list = list()
        for node, attrs in iterator:
            feat = attrs[node_feature]
            if not isinstance(feat, torch.Tensor):
                feat = torch.as_tensor(np.array(feat, dtype=float))
            tensor_list.append(feat)
        tensor = torch.stack(tensor_list, dim=0)
        return tensor

    def fix_buggy_edges(self, graph, strategy='remove'):
        """
        Sometimes some edges have weird names such as t.W representing a fuzziness.
        We just remove those as they don't deliver a good information

        :param graph:
        :param strategy: How to deal with it : for now just remove them.
        In the future maybe add an edge type in the edge map ?
        :return:
        """
        if strategy == 'remove':
            # Filter weird edges for now
            to_remove = list()
            for start_node, end_node, nodedata in graph.edges(data=True):
                if nodedata[self.label] not in self.edge_map:
                    to_remove.append((start_node, end_node))
            for start_node, end_node in to_remove:
                graph.remove_edge(start_node, end_node)
        else:
            raise ValueError(f'The edge fixing strategy : {strategy} was not implemented yet')
        return graph

    def shuffle(self):
        np.random.shuffle(self.all_graphs)

    def add_node_sim(self, node_simfunc):
        if node_simfunc is not None:
            if node_simfunc.method in ['R_graphlets', 'graphlet', 'R_ged']:
                if self.hashing_path is not None:
                    node_simfunc.add_hashtable(self.hashing_path)
                level = 'graphlet_annots'
            else:
                level = 'edge_annots'
        else:
            node_simfunc, level = None, None
        return node_simfunc, level

    def update_node_sim(self, node_simfunc):
        """
        This function is useful because the default_behavior is changed compared to above :
            Here if None is given, we don't remove the previous node_sim function

        :param node_simfunc: A nodesim.compare function
        :return:
        """
        if node_simfunc is not None:
            if node_simfunc.method in ['R_graphlets', 'graphlet', 'R_ged']:
                if self.hashing_path is not None:
                    node_simfunc.add_hashtable(self.hashing_path)
                level = 'graphlet_annots'
            else:
                level = 'edge_annots'
            self.node_simfunc, self.level = node_simfunc, level

    def subset(self, list_of_graphs):
        """
        Create another dataset with only the specified graphs

        :param list_of_graphs: a list of graph names
        :return: A graphdataset
        """
        subset = copy.deepcopy(self)
        subset.all_graphs = list(set(list_of_graphs).intersection(set(self.all_graphs)))
        return subset

    def __getitem__(self, idx):
        res_dict = {'graph_name': self.all_graphs[idx]}
        graph, node_attrs_toadd = self.get_nx_graph(idx)
        res_dict['num_nodes'] = len(graph)

        if 'point_cloud' in self.return_type or 'voxel' in self.return_type:
            # More robust to get C5 ? P is sometimes None
            coord_tens = self.as_tensor(graph, 'C5prime_xyz')
            # coord_tens = self.as_tensor(graph, 'P_xyz')
            if "features" in node_attrs_toadd:
                feat_tens = self.as_tensor(graph, 'features')
            if "target" in node_attrs_toadd:
                target_tens = self.as_tensor(graph, 'target')

            # If we need to return the point cloud computations
            if 'point_cloud' in self.return_type:
                res_dict['node_coords'] = coord_tens
                if "features" in node_attrs_toadd:
                    res_dict['node_feats'] = feat_tens
                if "target" in node_attrs_toadd:
                    res_dict['node_targets'] = target_tens

            # If we need voxels, let's do the computations. Once again it's tricky to get the right dimensions.
            if 'voxel' in self.return_type:
                to_embed = []
                if "features" in node_attrs_toadd:
                    to_embed.append(feat_tens)
                if "target" in node_attrs_toadd:
                    to_embed.append(target_tens)

                if len(to_embed) == 0:
                    features = None
                else:
                    if len(to_embed) == 2:
                        features = torch.hstack(to_embed)
                    else:
                        features = to_embed[0]
                    features = features.numpy()  # TODO : port in torch to avoid back and forth
                coords = coord_tens.numpy()
                voxel_representation = get_grid(coords=coords, features=features)
                voxel_representation = torch.from_numpy(voxel_representation)

                # Just retrieve a one-hot
                if features is None:
                    res_dict['voxel_feats'] = voxel_representation
                if "features" in node_attrs_toadd:
                    res_dict['voxel_feats'] = voxel_representation[:self.input_dim]
                if "target" in node_attrs_toadd:
                    res_dict['voxel_target'] = voxel_representation[-self.output_dim:]

        if 'graph' in self.return_type:
            graph = self.fix_buggy_edges(graph=graph)

            # Get Edge Labels
            edge_type = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                         (nx.get_edge_attributes(graph, self.label)).items()}
            nx.set_edge_attributes(graph, name='edge_type', values=edge_type)
            # Careful ! When doing this, the graph nodes get sorted.
            g_dgl = dgl.from_networkx(nx_graph=graph,
                                      edge_attrs=['edge_type'],
                                      node_attrs=node_attrs_toadd)
            res_dict['graph'] = g_dgl

            if self.node_simfunc is not None:
                ring = list(sorted(graph.nodes(data=self.level)))
                res_dict['ring'] = ring
        return res_dict
