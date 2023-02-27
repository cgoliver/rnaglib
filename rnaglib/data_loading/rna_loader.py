import os
import sys

from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import dgl

DGL_VERSION = dgl.__version__
if DGL_VERSION < "0.8":
    from dgl.dataloading.pytorch import EdgeDataLoader
else:
    from dgl.dataloading import DataLoader as DGLDataLoader

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.kernels.node_sim import k_block_list
from rnaglib.config.graph_keys import EDGE_MAP_RGLIB_REVERSE
from rnaglib.data_loading.rna_dataset import RNADataset
from rnaglib.data_loading.get_statistics import DEFAULT_INDEX
from rnaglib.representations import RingRepresentation


class Collater:
    def __init__(self, dataset):
        """
        Wrapper for collate function, so we can use different node similarities.
            We cannot use functools.partial as it is not picklable so incompatible with Pytorch loading
        :param node_simfunc: A node comparison function as defined in kernels, to optionally return a pairwise
        comparison of the nodes in the batch
        :param max_size_kernel: If the node comparison is not None, optionnaly only return a pairwise
        comparison between a subset of all nodes, of size max_size_kernel
        :param hstack: If True, hstack point cloud return
        :return: a picklable python function that can be called on a batch by Pytorch loaders
        """
        self.dataset = dataset

    def collate(self, samples):
        """
        New format that iterates through the possible keys returned by get_item

        The graphs are batched, the rings are compared with self.node_simfunc and the features are just put into a list.
        :param samples:
        :return: a dict
        """
        batch = dict()
        for representation in self.dataset.representations:
            representation_samples = [sample.pop(representation.name) for sample in samples]
            batched_representation = representation.batch(representation_samples)
            batch[representation.name] = batched_representation
        remaining_keys = set(samples[0].keys())
        for key in remaining_keys:
            batch[key] = [sample[key] for sample in samples]
        return batch


def split_list_in_fractions(list_to_split, split_train=0.7, split_valid=0.85):
    copy_list = list_to_split.copy()
    random.shuffle(copy_list)

    train_index, valid_index = int(split_train * len(copy_list)), int(split_valid * len(copy_list))

    train_list = copy_list[:train_index]
    valid_list = copy_list[train_index:valid_index]
    test_list = copy_list[valid_index:]
    return train_list, valid_list, test_list


def split_dataset_in_fractions(dataset, split_train=0.7, split_valid=0.85):
    """
    Just randomly split a dataset
    :param dataset:
    :param split_train:
    :param split_valid:
    :return:
    """
    indices = list(range(len(dataset)))
    train_indices, valid_indices, test_indices = split_list_in_fractions(indices,
                                                                         split_train=split_train,
                                                                         split_valid=split_valid)

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)
    return train_set, valid_set, test_set


def get_multitask_split(node_targets, graph_index=DEFAULT_INDEX, fractions=(0.15, 0.15)):
    """
    :param node_targets: A subset of {'binding_protein', 'binding_small-molecule', 'is_modified', 'binding_ion'}
    :param graph_index: should be the opened output of the previous function a dict of dict of dict.
    :param target_fraction: The fraction of each task to have in the test set

    Correctly splitting the data for multitasking is hard,
    For instance in a triangle situation AB,BC,CD,DA : we can split in halves along each dimension but not along
    two at the same time
    This is a very basic, greedy version of data splitting for multi task where we first count the amount of nodes
     for each attrs and we then fill a test split.

    :return: the splits in the form of a list of graphs.
    """
    # First count all occurences :
    total_counts = defaultdict(int)
    for graph, graph_attrs in graph_index.items():
        for graph_attrs_name, graph_attrs_counter in graph_attrs.items():
            if graph_attrs_name in node_targets:
                # Maybe there is something to be made here, but usually it's just absent from the encoding
                # So summing all values in counter makes sense
                total_counts[graph_attrs_name] += sum(graph_attrs_counter.values())

    # Then iterate again and stop after having filled all queried sets following the queried fractions
    splits_to_fill = [set() for _ in range(len(fractions))]
    filled_splits = []
    fractions = list(fractions)
    current_fraction = fractions.pop()
    currrent_split = splits_to_fill.pop()

    query_attrs_insplit = defaultdict(int)
    copy_query_attrs = node_targets.copy()

    for graph, graph_attrs in graph_index.items():
        for graph_attrs_name, graph_attrs_counter in graph_attrs.items():
            if graph_attrs_name in copy_query_attrs:
                # Now add this graph and update the splits
                currrent_split.add(graph)
                # total_nodes_in_split += len(graph.nodes()) TODO get the number of nodes per graph
                query_attrs_insplit[graph_attrs_name] += sum(graph_attrs_counter.values())
                attrs_fraction = float(query_attrs_insplit[graph_attrs_name]) / total_counts[graph_attrs_name]
                if attrs_fraction > current_fraction:
                    copy_query_attrs.remove(graph_attrs_name)
        # If we found everything we needed for this split, save it and reset relevant variables
        if len(copy_query_attrs) == 0:
            filled_splits = [list(currrent_split)] + filled_splits
            query_attrs_insplit = defaultdict(int)
            copy_query_attrs = node_targets.copy()
            if len(fractions) == 0:
                break
            current_fraction = fractions.pop()
            currrent_split = splits_to_fill.pop()

    all_but_train = set().union(*filled_splits)
    train_split = set(graph_index.keys()) - all_but_train
    all_splits = [list(train_split)] + filled_splits
    return all_splits


def get_single_task_split(node_target, graph_index=DEFAULT_INDEX, split_train=0.7, split_valid=0.85):
    all_list = []
    for graph, graph_attrs in graph_index.items():
        if node_target in graph_attrs:
            all_list.append(graph)

    train_graphs, valid_graphs, test_graphs = split_list_in_fractions(all_list,
                                                                      split_train=split_train,
                                                                      split_valid=split_valid)

    return train_graphs, valid_graphs, test_graphs


def split_dataset(dataset, split_train=0.7, split_valid=0.85):
    node_targets = [f"node_{target}" for target in dataset.nt_targets]
    # 1st strategy : if we are looking for a single property : subset the graphs that contain at least a node with this
    # property and make a random split among these.
    if len(node_targets) == 1:
        train_split, validation_split, test_split = get_single_task_split(node_targets[0],
                                                                          split_train=split_train,
                                                                          split_valid=split_valid)
        train_set = dataset.subset(train_split)
        validation_set = dataset.subset(validation_split)
        test_set = dataset.subset(test_split)
        return train_set, validation_set, test_set

    # 2nd strategy : for multitask objective, we could also subset, but a random split could end up with one of
    # the categories missing from a set
    fractions = (1 - split_valid, split_valid - split_train)
    train_split, validation_split, test_split = get_multitask_split(node_targets=node_targets,
                                                                    fractions=fractions)

    train_set = dataset.subset(train_split)
    validation_set = dataset.subset(validation_split)
    test_set = dataset.subset(test_split)
    return train_set, validation_set, test_set


def get_loader(dataset,
               batch_size=5,
               num_workers=0,
               split=True,
               split_train=0.7,
               split_valid=0.85,
               verbose=False):
    collater = Collater(dataset=dataset)
    if not split:
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size,
                            num_workers=num_workers, collate_fn=collater.collate)
        return loader

    else:
        train_set, valid_set, test_set = split_dataset(dataset, split_train=split_train, split_valid=split_valid)

        if verbose:
            print(f"training items: ", len(train_set))
        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collater.collate)
        valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=batch_size,
                                  num_workers=num_workers, collate_fn=collater.collate)
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size,
                                 num_workers=num_workers, collate_fn=collater.collate)
        return train_loader, valid_loader, test_loader


def get_inference_loader(list_to_predict,
                         data_path=None,
                         dataset=None,
                         batch_size=5,
                         num_workers=20,
                         **kwargs):
    """
    This is to just make an inference over a list of graphs.
    """
    if (dataset is None and data_path is None) or (dataset is not None and data_path is not None):
        raise ValueError("To create an inference loader please provide either an existing dataset or a data path")
    if dataset is None:
        dataset = RNADataset(data_path=data_path, **kwargs)
    subset = dataset.subset(list_to_predict)
    collater = Collater()
    train_loader = DataLoader(dataset=subset,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=collater.collate)
    return train_loader


class EdgeLoaderGenerator:
    def __init__(self,
                 graph_loader,
                 inner_batch_size=50,
                 sampler_layers=2,
                 neg_samples=1):
        """
        This turns a graph dataloader or dataset into an edge data loader generator.
        It needs to be reinitialized every epochs because of the double iteration pattern

        Iterates over batches of base pairs and generates negative samples for each.
        Negative sampling is just uniform for the moment (eventually we should change it to only sample
        edges at a certain backbone distance.

        timing :
        - num workers should be used to load the graphs not in the inner loop
        - The inner batch size yields huge speedups (probably generating all MFGs is tedious)

        :param graph_loader: A GraphLoader or GraphDataset. We will iterate over its graphs and then over its basepairs
        :param inner_batch_size: The amount of base-pairs to sample in each batch on each graph
        :param sampler_layers: The size of the neighborhood
        :param neg_samples: The number of negative sample to use per positive ones
        """
        self.graph_loader = graph_loader
        self.neg_samples = neg_samples
        self.sampler_layers = sampler_layers
        self.inner_batch_size = inner_batch_size
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.sampler_layers)
        self.negative_sampler = dgl.dataloading.negative_sampler.Uniform(self.neg_samples)
        self.eloader_args = {
            'shuffle': False,
            'batch_size': self.inner_batch_size,
            'negative_sampler': self.negative_sampler
        }

    @staticmethod
    def get_base_pairs(g):
        """
        Get edge IDS of edges in a base pair (non-backbone or unpaired).

        :param g: networkx graph
        :return: list of ids
        """
        eids = []
        for ind, e in enumerate(g.edata['edge_type']):
            if EDGE_MAP_RGLIB_REVERSE[e.item()][0] != 'B':
                eids.append(e)
        return eids

    def get_edge_loader(self):
        """
        Simply get the loader for one epoch. This needs to be called at each epoch

        :return: the edge loader
        """

        if DGL_VERSION < 1.8:
            from dgl.dataloading.pytorch import EdgeDataLoader
            edge_loader = (EdgeDataLoader(g_batched, self.get_base_pairs(g_batched), self.sampler, **self.eloader_args)
                           for g_batched, _ in self.graph_loader)
        else:
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                self.sampler,
                negative_sampler=self.negative_sampler)
            edge_loader = (DGLDataLoader(g_batched,
                                         self.get_base_pairs(g_batched),
                                         sampler,
                                         shuffle=False,
                                         batch_size=self.inner_batch_size)
                           for g_batched, _ in self.graph_loader)
        return edge_loader


class DefaultBasePairLoader:
    def __init__(self,
                 dataset=None,
                 data_path=None,
                 batch_size=5,
                 inner_batch_size=50,
                 sampler_layers=2,
                 neg_samples=1,
                 num_workers=4,
                 **kwargs):
        """
        Just a default edge base pair loader that deals with the splits

        :param dataset: A GraphDataset we want to loop over for base-pair prediction
        :param data_path: Optionnaly, we can use a data path to create a default GraphDataset
        :param batch_size: The desired batch size (number of whole graphs)
        :param inner_batch_size:The desired inner batch size (number of sampled edge in a batched graph)
        :param sampler_layers: The size of the neighborhood
        :param neg_samples: The number of negative sample to use per positive ones
        :param num_workers: The number of cores to use for loading
        """
        # Create default loaders
        if dataset is None:
            dataset = GraphDataset(data_path=data_path, **kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.g_train, self.g_val, self.g_test = get_loader(self.dataset,
                                                           batch_size=self.batch_size,
                                                           num_workers=self.num_workers)

        # Get the inner loader parameters
        self.inner_batch_size = inner_batch_size
        self.neg_samples = neg_samples
        self.sampler_layers = sampler_layers

    def get_data(self):
        train_loader = EdgeLoaderGenerator(graph_loader=self.g_train, inner_batch_size=self.inner_batch_size,
                                           sampler_layers=self.sampler_layers,
                                           neg_samples=self.neg_samples).get_edge_loader()
        val_loader = EdgeLoaderGenerator(graph_loader=self.g_val, inner_batch_size=self.inner_batch_size,
                                         sampler_layers=self.sampler_layers,
                                         neg_samples=self.neg_samples).get_edge_loader()
        test_loader = EdgeLoaderGenerator(graph_loader=self.g_test, inner_batch_size=self.inner_batch_size,
                                          sampler_layers=self.sampler_layers,
                                          neg_samples=self.neg_samples).get_edge_loader()

        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    pass
    node_features = ['nt_code', "alpha", "C5prime_xyz", "is_modified"]
    # node_features = None
    node_target = ['binding_ion', 'binding_protein']
    # node_target = None
    # node_simfunc = SimFunctionNode(method='R_1', depth=2)
    node_simfunc = None

    torch.random.manual_seed(42)

    from rnaglib.representations import GraphRepresentation, RingRepresentation
    from rnaglib.data_loading import RNADataset
    from rnaglib.kernels import node_sim

    # GET THE DATA GOING
    graph_rep = GraphRepresentation(framework='dgl')
    ring_rep = RingRepresentation(node_simfunc=node_simfunc, max_size_kernel=None)

    toy_dataset = RNADataset(
        representations=[graph_rep, ring_rep],
        annotated=True,
        nt_features=node_features,
        nt_targets=node_target)
    train_loader, validation_loader, test_loader = get_loader(dataset=toy_dataset,
                                                              batch_size=3,
                                                              num_workers=0)

    for i, batch in enumerate(train_loader):
        for k, v in batch.items():
            if 'voxel' in k:
                print(k, [value.shape for value in v])
        if i > 10:
            break
        # if not i % 20: print(i)
        pass
