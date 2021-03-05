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

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from torch.utils.data import Dataset, DataLoader, Subset
from kernels.node_sim import SimFunctionNode, k_block_list, simfunc_from_hparams, EDGE_MAP
from utils import graph_io


class GraphDataset(Dataset):
    def __init__(self,
                 edge_map,
                 node_simfunc=None,
                 annotated_path='../data/annotated/samples',
                 debug=False,
                 shuffled=False,
                 directed=True,
                 force_directed=False,
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
        :param directed: Whether we want to use directed graphs
        :param force_directed: If we ask for directed graphs from undirected graphs, this will raise an Error as
        we should rather be using directed annotations that are more rich (for instance we get the BB direction)
        :param label: The label to use
        """


        self.path = annotated_path
        self.all_graphs = sorted(os.listdir(annotated_path))
        self.label = label
        self.directed = directed
        self.node_simfunc = node_simfunc

        if node_simfunc is not None:
            if self.node_simfunc.method in ['R_graphlets', 'graphlet', 'R_ged']:
                self.level = 'graphlet_annots'
            else:
                self.level = 'edge_annots'
            self.depth = self.node_simfunc.depth
        else:
            self.level = None
            self.depth = None

        self.edge_map = edge_map
        # This is len() so we have to add the +1
        self.num_edge_types = max(self.edge_map.values()) + 1
        print(f"Found {self.num_edge_types} relations")

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        g_path = os.path.join(self.path, self.all_graphs[idx])
        graph = graph_io.load_json(g_path)

        # We can go from directed to undirected
        if self.directed and not nx.is_directed(graph):
            raise ValueError(f"The loader is asked to produce a directed graph from {g_path} that is undirected")
        if not self.directed:
            graph = nx.to_undirected(graph)

        # This is a weird call but necessary for DGL as it only deals
        #   with undirected graphs that have both directed edges
        # The error raised above ensures that we don't have a discrepancy *
        #   between the attribute directed and the graphs :
        #   One should not explicitly ask to make the graphs directed in the learning as it is done by default but when
        #   directed graphs are what we want, we should use the directed annotation rather than the undirected.
        graph = nx.to_directed(graph)
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, self.label)).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])

        if self.node_simfunc is not None:
            ring = list(sorted(graph.nodes(data=self.level)))
            return g_dgl, ring
        else:
            return g_dgl, 0


def collate_wrapper(node_simfunc=None):
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
            rings = [item for ring in rings for item in ring]
            K = k_block_list(rings, node_simfunc)
            return batched_graph, torch.from_numpy(K).detach().float(), len_graphs
    else:
        def collate_block(samples):
            # The input `samples` is a list of pairs
            #  (graph, label).
            graphs, _ = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            len_graphs = [graph.number_of_nodes() for graph in graphs]
            return batched_graph, [1 for _ in samples], len_graphs
    return collate_block


class Loader():
    def __init__(self,
                 annotated_path='../data/annotated/samples/',
                 batch_size=5,
                 num_workers=20,
                 debug=False,
                 shuffled=False,
                 edge_map=EDGE_MAP,
                 node_simfunc=None,
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
        :param hparams:
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = GraphDataset(annotated_path=annotated_path,
                                    debug=debug,
                                    shuffled=shuffled,
                                    node_simfunc=node_simfunc,
                                    edge_map=edge_map,
                                    directed=directed)

        self.directed = directed
        self.node_simfunc = node_simfunc
        self.num_edge_types = self.dataset.num_edge_types

        self.split = split

    def get_data(self):
        if not self.split:
            collate_block = collate_wrapper(self.node_simfunc)

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

            collate_block = collate_wrapper(self.node_simfunc)

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
    annotated_path = os.path.join("..", "data", "annotated", "samples")
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
