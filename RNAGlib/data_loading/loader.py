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
from collections import defaultdict

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from torch.utils.data import Dataset, DataLoader, Subset
from kernels.node_sim import SimFunctionNode, k_block_list, simfunc_from_hparams, EDGE_MAP
from utils import graph_io

FEATURE_MAPS ={
        'nt_code' : {k:v for v,k in enumerate(['A','U','C','G','P', 'c', 'a', 'u', 't', 'g'])},
        'nt_name' : {k:v for v,k in enumerate(['A','U','C','G','PSU', 'ATP', 'UR3', '2MG', '4OC', 'CCC', 'GDP', 'M2G', '5MC','7MG', 'MA6', 'GTP','G46', 'CBV','OMG', 'OMU', '5MU', '6MZ', 'RSP', 'G48', 'OMC', 'A44', '4SU','U36', 'H2U', 'CM0', 'I', 'C43', '1MA', 'A23'])},
        'form' : {k:v for v,k in enumerate(['A','-','B','Z','.','x'])},
        'dbn' : {k:v for v,k in enumerate(['(',')','{','}','<','>','&','.','[',']'])},
        'bb_type' : {'--':0, 'BI':1, 'BII':2},
        'glyco_bond' : {'--':0, 'anti':1, 'syn':2},
        'puckering' : {k:v for v,k in enumerate(["C3'-endo","C2'-endo","C3'-exo","C2'-exo","C4'-exo", "C1'-exo", "04'-exo","O4'-endo", "C1'-endo", "C4'-endo", "O4'-exo"])},
        'sugar_class' : {"~C3'-endo":0, "~C2'-endo":1, '--':3},
        'bin' : {k:v for v,k in enumerate(['33t','33p','33m','32t','32p','32m','23t','23p','23m','22t','22p','inc','trig', '22m']) },
        'cluster' : {b:n for n, b in enumerate(['1a', '1m', '1L', '&a', '7a', '3a', '9a', '1g', '7d', '3d', '5d', '1e', '1c', '1f', '6j', '1b', '1{', '3b', '1z', '5z', '7p', '1t', '5q', '1o', '7r', '2a', '4a', '0a', '#a', '4g', '6g', '8d', '4d', '6d', '2h', '4n', '0i', '6n', '6j', '2{', '4b', '0b', '4p', '6p', '4s', '2o', '5n', '5p', '5r', '3g', '2g', '__', '!!', '1[', '5j', '0k', '2z', '2u', '2['])},
        'sse' : {s:n for s, n in enumerate(['hairpin_1', 'hairpin_3', 'buldge_1'])}
        }

def dict_union(a, b):
    """
    performs union operation on two dictionaries of sets
    """
    c = {k:a[k].union(b[k]) for k in set(a.keys()).intersection(set(b.keys()))}
    for k in (set(b.keys()) - set(c.keys())): c[k] = b[k]
    for k in (set(a.keys()) - set(c.keys())): c[k] = a[k]

    for k,v in c.items():
        print(f'\nkey: {k}\tset:')
        print(v)

    print('\nNEXT\n')
    return c

class GraphDataset(Dataset):
    def __init__(self,
                 edge_map,
                 node_simfunc=None,
                 annotated_path='../data/annotated/samples',
                 directed=True,
                 label='LW',
                 node_features=None,
                 node_target=None
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
        :param node_features: node features to include, stored in one tensor in order given by user
        """

        self.path = annotated_path
        self.all_graphs = sorted(os.listdir(annotated_path))
        self.label = label
        self.directed = directed
        self.node_features = node_features
        self.node_target = node_target
        self.level = None
        self.unmapped_values = defaultdict(set)
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
        g_path = os.path.join(self.path, self.all_graphs[idx])
        graph = graph_io.load_json(g_path)

        # We can go from directed to undirected
        if self.directed and not nx.is_directed(graph):
            raise ValueError(f"The loader is asked to produce a directed graph from {g_path} that is undirected")
        if not self.directed:
            graph = nx.to_undirected(graph)

        # This is a weird call but necessary for DGL as it only deals
        #   with undirected graphs that have both directed edges
        # The error raised above ensures that we don't have a discrepancy
        #   between the attribute directed and the graphs :
        #   One should not explicitly ask to make the graphs directed in the learning as it is done by default but when
        #   directed graphs are what we want, we should use the directed annotation rather than the undirected.
        graph = graph.to_directed()

        # Filter weird edges for now
        to_remove = list()
        for start_node, end_node, nodedata in graph.edges(data=True):
            if nodedata[self.label] not in self.edge_map:
                to_remove.append((start_node, end_node))
        for start_node, end_node in to_remove:
            graph.remove_edge(start_node, end_node)

        # Get Edge Labels
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, self.label)).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        # Get Node labels
        if self.node_features is not None:
            feature_vector, unmapped_value = self.get_node_features(graph)
            # self.unmapped_values = dict_union(unmapped_value, self.unmapped_values)
            nx.set_node_attributes(graph, name='features',values=feature_vector)
        if self.node_target is not None:
            nx.set_node_attributes(graph, name='target', values=self.get_node_targets(graph))

        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'],
                node_attrs=['features', 'target'])

        if self.node_simfunc is not None:
            ring = list(sorted(graph.nodes(data=self.level)))
            return g_dgl, ring
        else:
            return g_dgl, 0

    def get_node_targets(self, g):
        """
        Get targets for graph g
        for every node get the attribute specified by self.node_target
        output a mapping of nodes to their targets
        """
        targets = {}

        for node, attrs in g.nodes.data():
            if 'binding' in self.node_target:
                if attrs[self.node_target] is None:
                    targets[node] = 0
                else:
                    targets[node] = 1
            else:
                try:
                    targets[node] = float(attrs[self.node_target])
                except ValueError:
                        try:
                            feats_flt[i] = FEATURE_MAPS[feature][attrs[feature]]
                        except KeyError as e:
                            raise Exception('ERROR: Cannot convert node target "{self.node_target}" to float')

            if 'cluster' == self.node_target:
                targets[node] = targets[node] * attrs['suiteness']

            targets[node] = torch.tensor(targets[node], dtype=torch.float32)

        return targets

    def get_node_features(self, g):
        """
        Get node attributes from g selected from self.node_features
        transform into feature tensor
        enumerate str features with FEATURE_MAPS
        output mapping of nodes to features tensor
        """
        junk_attrs = ['index_chain', 'chain_name', 'nt_resnum', 'nt_id', 'nt_type', 'summary', 'C5prime_xyz', 'P_xyz', 'frame', 'is_modified']

        feats = {}
        unmapped_value = defaultdict(set)

        for node, attrs in g.nodes.data():
            if self.node_features == 'all':
                self.node_features = list( set(attrs.keys()) - set(junk_attrs) )
            feats_flt = torch.zeros(len(self.node_features))
            for i, feature in enumerate(self.node_features):
                try:
                    feats_flt[i] = float(attrs[feature])
                except TypeError:
                    # feats_flt[i] = float('NaN')
                    continue
                except ValueError:
                    if 'binding' in feature:
                        if attrs[feature] is None:
                            feats_flt[i] = 0.0
                        else:
                            feats_flt[i] = 1.0
                    else:
                        try:
                            feats_flt[i] = FEATURE_MAPS[feature][attrs[feature]]
                        except KeyError as e:
                            # print(e)
                            # print(FEATURE_MAPS[feature])
                            FEATURE_MAPS[feature][attrs[feature]] = len(FEATURE_MAPS[feature])
                            feats_flt[i] = FEATURE_MAPS[feature][attrs[feature]]
                            unmapped_value[feature].add(attrs[feature])
                            # raise Exception(f'RNAGlib ERROR: Cannot convert node feature {feature} with value {attrs[feature]} to float')
            if 'cluster' == feature:
                feats_flt[i] = feats_flt[i] * attrs['suiteness']

            # if len(unmapped_value.values()) != 0:
                # for key, value in unmapped_value.items():
                    # print(f'key:{key}', f'value: {value}\n')
            feats[node] = feats_flt

        return feats, unmapped_value

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


class Loader:
    def __init__(self,
                 annotated_path='../data/annotated/samples/',
                 batch_size=5,
                 num_workers=20,
                 edge_map=EDGE_MAP,
                 node_features=None,
                 node_target=None,
                 label='LW',
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
        :param node_features: (str list) features to be included in feature tensor
        :param node_target: (str) target attribute for node classification
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = GraphDataset(annotated_path=annotated_path,
                                    node_simfunc=node_simfunc,
                                    node_features=node_features,
                                    node_target=node_target,
                                    label=label,
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
    annotated_path = os.path.join(script_dir, '../../data/annotated/undirected')
    simfunc_r1 = SimFunctionNode('R_1', 2)
    loader = Loader(annotated_path=annotated_path,
                    num_workers=0,
                    split=False,
                    directed=False,
                    node_simfunc=simfunc_r1,
                    node_features='all',
                    node_target='binding_protein')
    train_loader = loader.get_data()
    # for graph, K, lengths in train_loader:
        # print('graph :', graph)
        # print('K :', K)
        # print('length :', lengths)
