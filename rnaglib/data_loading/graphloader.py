import os
import sys

from collections import defaultdict
import numpy as np
import random

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
from rnaglib.data_loading.graphdataset import GraphDataset
from rnaglib.data_loading.get_statistics import DEFAULT_INDEX


class Collater:
    def __init__(self, node_simfunc=None, max_size_kernel=None):
        """
        Wrapper for collate function, so we can use different node similarities.
            We cannot use functools.partial as it is not picklable so incompatible with Pytorch loading
        :param node_simfunc: A node comparison function as defined in kernels, to optionally return a pairwise
        comparison of the nodes in the batch
        :param max_size_kernel: If the node comparison is not None, optionnaly only return a pairwise
        comparison between a subset of all nodes, of size max_size_kernel
        :return: a picklable python function that can be called on a batch by Pytorch loaders
        """
        self.node_simfunc = node_simfunc
        self.max_size_kernel = max_size_kernel

    @staticmethod
    def collate_rings(list_of_rings, node_simfunc, max_size_kernel=None):
        # we need to flatten the list and then use the kernels :
        # The rings is now a list of list of tuples
        # If we have a huge graph, we can sample max_size_kernel nodes to avoid huge computations,
        # We then return the sampled ids

        flat_rings = list()
        for ring in list_of_rings:
            flat_rings.extend(ring)
        if max_size_kernel is None or len(flat_rings) < max_size_kernel:
            # Just take them all
            node_ids = [1 for _ in flat_rings]
        else:
            # Take only 'max_size_kernel' elements
            node_ids = [1 for _ in range(max_size_kernel)] + \
                       [0 for _ in range(len(flat_rings) - max_size_kernel)]
            random.shuffle(node_ids)
            flat_rings = [node for i, node in enumerate(flat_rings) if node_ids[i] == 1]
        K = k_block_list(flat_rings, node_simfunc)
        return torch.from_numpy(K).detach().float(), node_ids

    def collate(self, samples):
        """
        New format that iterates through the possible keys returned by get_item

        The graphs are batched, the rings are compared with self.node_simfunc and the features are just put into a list.
        :param samples:
        :return: a dict
        """
        # Exceptionnal treatment for batching graphs and rings.
        # Otherwise, return a list of individual embeddings (concatenation is one liner)
        batch = dict()
        batch_keys = set(samples[0].keys())
        if 'graph' in batch_keys:
            batched_graph = dgl.batch([sample['graph'] for sample in samples])
            batch['graphs'] = batched_graph

        if 'ring' in batch_keys:
            K, node_ids = self.collate_rings([sample['ring'] for sample in samples], self.node_simfunc,
                                             self.max_size_kernel)
            batch['node_similarities'] = (K, node_ids)

        for key in batch_keys - {'graph', 'ring'}:
            batch[key] = [sample[key] for sample in samples]
        return batch


def split_in_fractions(list_to_split, split_train=0.7, split_valid=0.85):
    copy_list = list_to_split.copy()
    random.shuffle(copy_list)

    train_index, valid_index = int(split_train * len(copy_list)), int(split_valid * len(copy_list))

    train_list = copy_list[:train_index]
    valid_list = copy_list[train_index:valid_index]
    test_list = copy_list[valid_index:]
    return train_list, valid_list, test_list


def full_split(dataset, split_train=0.7, split_valid=0.85):
    """
    Just randomly split a dataset
    :param dataset:
    :param split_train:
    :param split_valid:
    :return:
    """
    indices = list(range(len(dataset)))
    train_indices, valid_indices, test_indices = split_in_fractions(indices,
                                                                    split_train=split_train,
                                                                    split_valid=split_valid)

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)
    return train_set, valid_set, test_set


def get_multitask_split(node_targets, graph_index=DEFAULT_INDEX, target_fraction=0.2):
    """
    :param node_target: A subset of {'binding_protein', 'binding_small-molecule', 'is_modified', 'binding_ion'}
    :param seed: Should be set to the default zero
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
    query_attrs_insplit = defaultdict(int)
    # total_nodes_in_split = 0
    copy_query_attrs = node_targets.copy()
    selected_graphs = set()
    # Then iterate again and stop after reaching the threshold.
    for graph, graph_attrs in graph_index.items():
        for graph_attrs_name, graph_attrs_counter in graph_attrs.items():
            if graph_attrs_name in copy_query_attrs:
                # Now add this graph and update the splits
                selected_graphs.add(graph)
                # total_nodes_in_split += len(graph.nodes()) TODO get the number of nodes per graph
                query_attrs_insplit[graph_attrs_name] += sum(graph_attrs_counter.values())
                attrs_fraction = float(query_attrs_insplit[graph_attrs_name]) / total_counts[graph_attrs_name]
                if attrs_fraction > target_fraction:
                    copy_query_attrs.remove(graph_attrs_name)
        # If we found everything we needed
        if len(copy_query_attrs) == 0:
            break

    train_split, test_split = set(graph_index.keys()) - selected_graphs, selected_graphs
    return train_split, test_split


def get_single_task_split(node_target, graph_index=DEFAULT_INDEX, split_train=0.7, split_valid=0.85):
    all_list = []
    for graph, graph_attrs in graph_index.items():
        if node_target in graph_attrs:
            all_list.append(graph)

    train_graphs, valid_graphs, test_graphs = split_in_fractions(all_list,
                                                                 split_train=split_train,
                                                                 split_valid=split_valid)

    return train_graphs, valid_graphs, test_graphs


def meaningful_split_dataset(dataset, split_train=0.7, split_valid=0.85):
    node_targets = [f"node_{target}" for target in dataset.node_target]
    # 1st strategy : if we are looking for a single property : subset the graphs that contain at least a node with this
    # property and make a random split among these.
    if len(node_targets) == 1:
        train_split, validation_split, test_split = get_single_task_split(node_targets[0])
        train_set = dataset.subset(train_split)
        validation_set = dataset.subset(validation_split)
        test_set = dataset.subset(test_split)
        return train_set, validation_set, test_set

    # 2nd strategy : for multitask objective, we could also subset, but a random split could end up with one of
    # the categories missing from a set
    train_split, test_split = get_multitask_split(node_targets=node_targets,
                                                  target_fraction=split_train)

    train_set = dataset.subset(train_split)
    test_set = dataset.subset(test_split)
    return train_split, None, test_set  # TODO : implement validation using similar strategy


def get_loader(dataset,
               batch_size=5,
               num_workers=0,
               max_size_kernel=None,
               split=True,
               verbose=False):
    collater = Collater(dataset.node_simfunc, max_size_kernel=max_size_kernel)
    if not split:
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size,
                            num_workers=num_workers, collate_fn=collater.collate)
        return loader

    else:

        train_set, valid_set, test_set = meaningful_split_dataset(dataset)
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
                         data_path,
                         dataset=None,
                         batch_size=5,
                         num_workers=20,
                         **kwargs):
    """
    This is to just make an inference over a list of graphs.
    """
    if dataset is None:
        dataset = GraphDataset(data_path=data_path, **kwargs)
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
    node_target = ['binding_ion']
    # node_target = None
    # node_simfunc = SimFunctionNode(method='R_1', depth=2)
    node_simfunc = None

    torch.random.manual_seed(42)

    # GET THE DATA GOING
    toy_dataset = GraphDataset(
        # data_path='data/graphs/all_graphs',
        node_features=node_features,
        node_target=node_target,
        return_type='voxel',
        node_simfunc=node_simfunc)
    train_loader, validation_loader, test_loader = get_loader(dataset=toy_dataset,
                                                              batch_size=2,
                                                              num_workers=0)

    for i, batch in enumerate(train_loader):
        for k, v in batch.items():
            if 'voxel' in k:
                print(k, [value.shape for value in v])
        if i > 10:
            break
        # if not i % 20: print(i)
        pass
