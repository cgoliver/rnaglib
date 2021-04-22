import pickle
import sys
import os
from tqdm import tqdm
import networkx as nx
import dgl
import numpy as np
import torch

import os
import sys
import random

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from torch.utils.data import Dataset, DataLoader, Subset
from kernels.node_sim import SimFunctionNode, k_block_list, simfunc_from_hparams, EDGE_MAP
from utils import graph_io

import time


class GraphDataset(Dataset):
    def __init__(self,
                 edge_map,
                 node_simfunc=None,
                 annotated_path='../data/annotated/samples',
                 force_undirected=False,
                 label='LW'
                 ):
        """

        :param edge_map: Necessary to build the one hot mapping from edge labels to an id
        :param node_simfunc: Similarity function defined in kernels/node_sim
        :param annotated_path: The path of the data. If node_sim is not None, this data should be annotated
        (if not, it will annotate the data which is a long process)
                        TODO : do this annotate if not, probably with input confirmation
        :param debug:
        :param shuffled:
        :param force_directed: Whether we want to force the use of undirected graphs from a directed data set.
        Otherwise the directed attribute is observed from the data at hands.
        :param label: The label to use
        """

        self.path = annotated_path
        self.all_graphs = sorted(os.listdir(annotated_path))
        self.label = label

        # To ensure that we don't have a discrepancy between the attribute directed and the graphs :
        #   Since the original data is directed, it does not make sense to ask to build directed graphs
        #   from the undirected set.
        #   If directed graphs are what one wants, one should use the directed annotation rather than the undirected.
        sample_path = os.path.join(self.path, self.all_graphs[0])
        graph = graph_io.load_json(sample_path)
        self.directed = nx.is_directed(graph)
        self.force_undirected = force_undirected

        self.level = None
        self.node_simfunc, self.level = self.add_node_sim(node_simfunc=node_simfunc)

        self.edge_map = edge_map
        # This is len() so we have to add the +1
        self.num_edge_types = max(self.edge_map.values()) + 1
        print(f"Found {self.num_edge_types} relations")

    def __len__(self):
        return len(self.all_graphs)

    def add_node_sim(self, node_simfunc):
        if node_simfunc is not None:
            if node_simfunc.method in ['R_graphlets', 'graphlet', 'R_ged']:
                level = 'graphlet_annots'
            else:
                level = 'edge_annots'
        else:
            node_simfunc, level = None, None
        return node_simfunc, level

    def __getitem__(self, idx):
        time_start = time.perf_counter()
        g_path = os.path.join(self.path, self.all_graphs[idx])
        graph = graph_io.load_json(g_path)

        # We can go from directed to undirected
        if self.force_undirected:
            graph = nx.to_undirected(graph)

        # This is a weird call but necessary for DGL as it only deals
        #   with undirected graphs that have both directed edges
        graph = graph.to_directed()

        # Filter weird edges for now
        to_remove = list()
        for start_node, end_node, nodedata in graph.edges(data=True):
            if nodedata[self.label] not in self.edge_map:
                to_remove.append((start_node, end_node))
        for start_node, end_node in to_remove:
            graph.remove_edge(start_node, end_node)

        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, self.label)).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
        print(f' Everything up to dgl graph took : {time.perf_counter() - time_start}')

        if self.node_simfunc is not None:
            ring = list(sorted(graph.nodes(data=self.level)))
            return g_dgl, ring
        else:
            return g_dgl, 0


def collate_wrapper(node_simfunc=None, max_size_kernel=None):
    """
        Wrapper for collate function so we can use different node similarities.

        We cannot use functools.partial as it is not picklable so incompatible with Pytorch loading
    """
    if node_simfunc is not None:
        def collate_block(samples):
            # The input `samples` is a list of tuples (graph, ring).
            graphs, rings = map(list, zip(*samples))

            # DGL makes batching by making all small graphs a big one with disconnected components
            # We keep track of those
            batched_graph = dgl.batch(graphs)
            len_graphs = [graph.number_of_nodes() for graph in graphs]

            # Now compute similarities, we need to flatten the list and then use the kernels :
            # The rings is now a list of list of tuples
            # If we have a huge graph, we can sample max_size_kernel nodes to avoid huge computations,
            # We then return the sampled ids
            flat_rings = list()
            node_ids = list()
            for ring in rings:
                if max_size_kernel is None or len(ring) < max_size_kernel:
                    # Just take them all
                    node_ids.extend([1 for _ in ring])
                    flat_rings.extend(ring)
                else:
                    # Take only 'max_size_kernel' elements
                    graph_node_id = [1 for _ in range(max_size_kernel)] + [0 for _ in
                                                                           range(len(ring) - max_size_kernel)]
                    random.shuffle(graph_node_id)
                    node_ids.extend(graph_node_id)
                    flat_rings.extend([node for i, node in enumerate(ring) if graph_node_id[i] == 1])
            K = k_block_list(flat_rings, node_simfunc)
            return batched_graph, torch.from_numpy(K).detach().float(), len_graphs, node_ids
    else:
        def collate_block(samples):
            # The input `samples` is a list of pairs
            #  (graph, label).
            graphs, _ = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            len_graphs = [graph.number_of_nodes() for graph in graphs]
            return batched_graph, len_graphs
    return collate_block


class Loader:
    def __init__(self,
                 annotated_path='../data/annotated/samples/',
                 batch_size=5,
                 num_workers=20,
                 edge_map=EDGE_MAP,
                 node_simfunc=None,
                 max_size_kernel=None,
                 directed=True,
                 split=True):
        """

        :param annotated_path:
        :param batch_size:
        :param num_workers:
        :param debug:
        :param shuffled:
        :param node_simfunc: The node comparison object to use for the embeddings. If None is selected,
        will just return graphs
        :param max_graphs: If we use K comptutations, we need to subsamble some nodes for the big graphs
        or else the k computation takes too long
        :param hparams:
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = GraphDataset(annotated_path=annotated_path,
                                    node_simfunc=node_simfunc,
                                    edge_map=edge_map,
                                    directed=directed)
        self.max_size_kernel = max_size_kernel
        self.directed = directed
        self.node_simfunc = node_simfunc
        self.num_edge_types = self.dataset.num_edge_types
        self.split = split

    def get_data(self):
        collate_block = collate_wrapper(self.node_simfunc, max_size_kernel=self.max_size_kernel)
        if not self.split:
            loader = DataLoader(dataset=self.dataset, shuffle=True, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=collate_block)
            return loader

        else:
            n = len(self.dataset)
            indices = list(range(n))
            # np.random.shuffle(indices)

            np.random.seed(0)
            split_train, split_valid = 0.7, 0.85
            train_index, valid_index = int(split_train * n), int(split_valid * n)

            train_indices = indices[:train_index]
            valid_indices = indices[train_index:valid_index]
            test_indices = indices[valid_index:]

            train_set = Subset(self.dataset, train_indices)
            valid_set = Subset(self.dataset, valid_indices)
            test_set = Subset(self.dataset, test_indices)

            print(f"training items: ", len(train_set))
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                     num_workers=self.num_workers, collate_fn=collate_block)
            return train_loader, valid_loader, test_loader


class InferenceLoader(Loader):
    def __init__(self,
                 list_to_predict,
                 annotated_path,
                 batch_size=5,
                 num_workers=20,
                 edge_map=EDGE_MAP,
                 directed=True):
        super().__init__(
            annotated_path=annotated_path,
            batch_size=batch_size,
            num_workers=num_workers,
            edge_map=edge_map,
            directed=directed
        )
        self.dataset.all_graphs = list_to_predict
        self.dataset.path = annotated_path
        print(len(list_to_predict))

    def get_data(self):
        collate_block = collate_wrapper(None)
        train_loader = DataLoader(dataset=self.dataset,
                                  shuffle=False,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=collate_block)
        return train_loader


def loader_from_hparams(annotated_path, hparams, list_inference=None):
    """
        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
    """
    if list_inference is None:
        node_simfunc = simfunc_from_hparams(hparams)
        loader = Loader(annotated_path=annotated_path,
                        batch_size=hparams.get('argparse', 'batch_size'),
                        num_workers=hparams.get('argparse', 'workers'),
                        edge_map=hparams.get('edges', 'edge_map'),
                        node_simfunc=node_simfunc)
        return loader

    loader = InferenceLoader(list_to_predict=list_inference,
                             annotated_path=annotated_path,
                             batch_size=hparams.get('argparse', 'batch_size'),
                             num_workers=hparams.get('argparse', 'workers'),
                             edge_map=hparams.get('edges', 'edge_map'))
    return loader


if __name__ == '__main__':
    pass
    annotated_path = os.path.join(script_dir, "..", "data", "annotated", "samples")
    simfunc_r1 = SimFunctionNode('R_1', 2)
    loader = Loader(annotated_path=annotated_path,
                    num_workers=0,
                    split=False,
                    directed=False,
                    node_simfunc=simfunc_r1)
    train_loader = loader.get_data()
    for graph, K, lengths in train_loader:
        print('graph :', graph)
        # print('K :', K)
        # print('length :', lengths)
