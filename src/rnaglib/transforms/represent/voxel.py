import os
import sys

import copy
import networkx as nx
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import torch

from .representation import Representation
from .point_cloud import get_point_cloud_dict


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


class VoxelRepresentation(Representation):
    """
    Converts RNA into a voxel based representation
    """

    def __init__(self, spacing=2, padding=3, sigma=1., **kwargs):
        super().__init__(**kwargs)
        self.spacing = spacing
        self.padding = padding
        self.sigma = sigma

    def __call__(self, rna_graph, features_dict):
        # If we need voxels, let's do the computations.
        # We redo the point cloud computations that are fast compared to voxels
        point_cloud_dict = get_point_cloud_dict(rna_graph, features_dict, sort=False)

        point_cloud_coords = point_cloud_dict['point_cloud_coords']

        output_dim = 0
        if "point_cloud_feats" in point_cloud_dict:
            stacked_feats = point_cloud_dict['point_cloud_feats']
            input_dim = stacked_feats.shape[1]
        # If no features are provided, use a one hot encoding
        else:
            stacked_feats = torch.ones(size=(len(point_cloud_coords), 1))
            input_dim = 1
        to_embed = [stacked_feats]
        if "point_cloud_targets" in point_cloud_dict:
            stacked_targets = point_cloud_dict['point_cloud_targets']
            output_dim = stacked_targets.shape[1]
            to_embed.append(stacked_targets)

        if output_dim > 0:
            features = torch.hstack((stacked_feats, stacked_targets))
        else:
            features = stacked_feats

        features = features.numpy()  # TODO : port in torch to avoid back and forth
        coords = point_cloud_coords.numpy()
        voxel_representation = get_grid(coords=coords, features=features,
                                        spacing=self.spacing, padding=self.padding, sigma=self.sigma)
        voxel_representation = torch.from_numpy(voxel_representation)

        res_dict = {'voxel_feats': voxel_representation[:input_dim]}
        if output_dim > 0:
            res_dict['voxel_target'] = voxel_representation[-output_dim:]
        return res_dict

    @property
    def name(self):
        return "voxel"

    def batch(self, samples):
        """
        Batch a list of voxel samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        voxel_batch = {}
        for key, value in samples[0].items():
            voxel_batch[key] = [sample[key] for sample in samples]
        return voxel_batch
