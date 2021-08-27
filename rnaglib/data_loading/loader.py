import os
import sys

import pickle
import networkx as nx
import numpy as np
import random
import requests
import warnings
import tarfile
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import dgl
from dgl.dataloading.pytorch import EdgeDataLoader

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.kernels.node_sim import SimFunctionNode, k_block_list, simfunc_from_hparams
from rnaglib.utils import graph_io
from rnaglib.data_loading.feature_maps import build_node_feature_parser
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL, EDGE_MAP_RGLIB_REVERSE

# This consists in the keys of the feature map that we consider as not relevant for now.
JUNK_ATTRS = ['index_chain', 'chain_name', 'nt_resnum', 'nt_id', 'nt_type', 'summary', 'C5prime_xyz', 'P_xyz',
              'frame', 'is_modified']

# The annotation fields also should not be included as node features
ANNOTS_ATTRS = ['node_annots', 'edge_annots', 'graphlet_annots']


def dict_union(a, b):
    """
    performs union operation on two dictionaries of sets
    """
    c = {k: a[k].union(b[k]) for k in set(a.keys()).intersection(set(b.keys()))}
    for k in (set(b.keys()) - set(c.keys())):
        c[k] = b[k]
    for k in (set(a.keys()) - set(c.keys())):
        c[k] = a[k]

    for k, v in c.items():
        print(f'\nkey: {k}\tset:')
        print(v)

    print('\nNEXT\n')
    return c


def download(url, path=None, overwrite=True, retries=5, verify_ssl=True, log=True):
    """Download a given URL.

    Codes borrowed from mxnet/gluon/utils.py

    Parameters
    ----------
    url : str
        URL to download.
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with the same name as in url.
    overwrite : bool, optional
        Whether to overwrite the destination file if it already exists.
        By default always overwrites the downloaded file.
    retries : integer, default 5
        The number of times to attempt downloading in case of failure or non 200 return codes.
    verify_ssl : bool, default True
        Verify SSL certificates.
    log : bool, default True
        Whether to print the progress for download

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
                      'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                if log:
                    print('Downloading %s from %s...' % (fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s" % url)
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    if log:
                        print("download failed, retrying, {} attempt{} left"
                              .format(retries, 's' if retries > 1 else ''))

    return fname


def download_name_generator(dirname=None,
                            release='iguana',
                            redundancy='NR',
                            chop=False,
                            annotated=False):
    # Generic name
    chop_str = '_chops' if chop else ''
    annotated_str = '_annot' if annotated else ''
    tarball_name = f'{redundancy}{chop_str}{annotated_str}'

    # Find remote url and get download link
    url = f'http://rnaglib.cs.mcgill.ca/static/datasets/{release}/{tarball_name}.tar.gz'
    dl_path = os.path.join(script_dir, f'../data/downloads/{tarball_name}.tar.gz')

    # Complete dl path depending on annotation and optionally get hashing too
    if annotated_str == '':
        data_path = os.path.join(script_dir, '../data/graphs/')
        hashing_info = None
    else:
        data_path = os.path.join(script_dir, '../data/annotated/')
        hashing_url = f'http://rnaglib.cs.mcgill.ca/static/datasets/{release}/{tarball_name}_hash.p'
        hashing_path = os.path.join(script_dir, f'../data/hashing/{tarball_name}_hash.p')
        hashing_info = (hashing_url, hashing_path)
    dirname = tarball_name if dirname is None else dirname
    return url, dl_path, data_path, dirname, hashing_info


def download_name_factory_deprecated(download_option):
    # Get graphs
    if download_option == 'samples_graphs':
        url = 'toto'
        dl_path = os.path.join(script_dir, '../data/downloads/samples.zip')
        data_path = os.path.join(script_dir, '../data/graphs/')
        dirname = 'samples'

        return url, dl_path, data_path, dirname, None
    if download_option == 'nr_graphs':
        url = 'http://rnaglib.cs.mcgill.ca/static/datasets/glib_nr_graphs.tar.gz'
        dl_path = os.path.join(script_dir, '../data/downloads/glib_nr_graphs.tar.gz')
        data_path = os.path.join(script_dir, '../data/graphs/')
        dirname = 'nr_graphs'
        return url, dl_path, data_path, dirname, None
    if download_option == 'graphs':
        url = 'toto'
        dl_path = os.path.join(script_dir, '../data/downloads/graphs.zip')
        data_path = os.path.join(script_dir, '../data/graphs/')
        dirname = 'graphs'
        return url, dl_path, data_path, dirname, None

    # Get annotations
    if download_option == 'samples_annotated':
        url = 'toto'
        dl_path = os.path.join(script_dir, '../data/downloads/samples_annotated.zip')
        data_path = os.path.join(script_dir, '../data/annotated/')
        dirname = 'samples_annotated'
        hashing_url = 'toto_hash'
        hashing_path = os.path.join(script_dir, '../data/hashing/samples_annotated.p')
        return url, dl_path, data_path, dirname, (hashing_url, hashing_path)
    if download_option == 'nr_annotated':
        url = 'http://rnaglib.cs.mcgill.ca/static/datasets/glib_nr_annot.tar.gz'
        dl_path = os.path.join(script_dir, '../data/downloads/glib_nr_annot.tar.gz')
        data_path = os.path.join(script_dir, '../data/annotated/')
        dirname = 'nr_annotated'
        hashing_url = 'http://rnaglib.cs.mcgill.ca/static/datasets/glib_nr_hashtable.p'
        hashing_path = os.path.join(script_dir, '../data/hashing/nr_annotated.p')
        return url, dl_path, data_path, dirname, (hashing_url, hashing_path)
    if download_option == 'annotated':
        url = 'toto'
        dl_path = os.path.join(script_dir, '../data/downloads/annotated.zip')
        data_path = os.path.join(script_dir, '../data/annotated/')
        dirname = 'annotated'
        hashing_url = 'toto_hash'
        hashing_path = os.path.join(script_dir, '../data/hashing/annotated.p')
        return url, dl_path, data_path, dirname, (hashing_url, hashing_path)
    else:
        raise ValueError(f'The download string command "{download_option}" is not supported. '
                         f'Options should be among : '
                         f'"samples_graphs", "nr_graphs", "graphs", '
                         f'"samples_annotated", "nr_annotated", "annotated"')


class GraphDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 redundancy='NR',
                 chop=False,
                 annotated=False,
                 all_graphs=None,
                 hashing_path=None,
                 edge_map=GRAPH_KEYS['edge_map'][TOOL],
                 label='LW',
                 node_simfunc=None,
                 node_features='nt_code',
                 node_target=None,
                 verbose=False):
        """
        :param data_path: The path of the data. If node_sim is not None, this data should be annotated
        :param redundancy: To use all graphs or just the non redundant set.
        :param chop: if we want full graphs or chopped ones for learning on smaller chunks
        :param annotated: if we want annotated graphs
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use
        :param edge_map: Necessary to build the one hot mapping from edge labels to an id
        :param label: The label to use
        :param node_simfunc: The node comparison object as defined in kernels/node_sim to use for the embeddings.
         If None is selected, this will just return graphs
        :param node_features: node features to include, stored in one tensor in order given by user,
        for example : ('nt_code','is_modified')
        :param node_features: node targets to include, stored in one tensor in order given by user
        for example : ('binding_protein', 'binding_small-molecule')
        """
        # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched
        # By default, we set hashing to None and potential node sim should be specified when creating
        # the node_sim function.
        # Then if a download occurs and no hashing was provided to the loader, the hashing used is the one
        # fetched by the downloading process to ensure it matches the data we iterate over.
        self.hashing_path = hashing_path
        if data_path is None:
            data_path = self.download_graphs(redundancy=redundancy, chop=chop, annotated=annotated)

        self.data_path = data_path

        if all_graphs is not None:
            self.all_graphs = all_graphs
        else:
            self.all_graphs = sorted(os.listdir(data_path))

        # This is len() so we have to add the +1
        self.label = label
        self.edge_map = edge_map
        self.num_edge_types = max(self.edge_map.values()) + 1
        if verbose:
            print(f"Found {self.num_edge_types} relations")

        # If it is not None, add a node comparison tool
        self.node_simfunc, self.level = self.add_node_sim(node_simfunc=node_simfunc)

        # If queried, add node features and node targets
        self.node_features = [node_features] if isinstance(node_features, str) else node_features
        self.node_target = [node_target] if isinstance(node_target, str) else node_target

        self.node_features_parser = build_node_feature_parser(self.node_features)
        self.node_target_parser = build_node_feature_parser(self.node_target)

        self.input_dim = self.compute_dim(self.node_features_parser)
        self.output_dim = self.compute_dim(self.node_target_parser)

    def download_graphs(self, redundancy='NR', chop=False, annotated=False, overwrite=False):
        # Get the correct names for the download option and download the correct files
        url, dl_path, data_path, dirname, hashing = download_name_generator(redundancy=redundancy,
                                                                            chop=chop,
                                                                            annotated=annotated)
        full_data_path = os.path.join(data_path, dirname)
        if not os.path.exists(full_data_path) or overwrite:
            if not os.path.exists(dl_path) or overwrite:
                print('Required dataset not found, launching a download. This should take about a minute')
                download(path=dl_path,
                         url=url)
            # Expand the compressed files at the right location
            if dl_path.endswith('.zip'):
                with zipfile.ZipFile(dl_path, 'r') as zip_file:
                    zip_file.extractall(path=data_path)
            elif '.tar' in url:
                with tarfile.open(dl_path) as tar_file:
                    tar_file.extractall(path=data_path)
        if hashing is not None:
            hashing_url, hashing_path = hashing
            if not os.path.exists(hashing_path) or overwrite:
                download(path=hashing_path,
                         url=hashing_url)
            # Don't overwrite if the user has specifically asked for a given hashing
            if self.hashing_path is None:
                self.hashing_path = hashing_path
        return full_data_path

    def __len__(self):
        return len(self.all_graphs)

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
        :param node_simfunc:
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

    def get_node_encoding(self, g, encode_feature=True):
        """
        Get targets for graph g
        for every node get the attribute specified by self.node_target
        output a mapping of nodes to their targets
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
        self.all_graphs = np.random.shuffle(self.all_graphs)

    def __getitem__(self, idx):
        g_path = os.path.join(self.data_path, self.all_graphs[idx])
        graph = graph_io.load_graph(g_path)
        graph = self.fix_buggy_edges(graph=graph)

        # Get Edge Labels
        edge_type = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                     (nx.get_edge_attributes(graph, self.label)).items()}
        nx.set_edge_attributes(graph, name='edge_type', values=edge_type)

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
        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=graph,
                                  edge_attrs=['edge_type'],
                                  node_attrs=node_attrs_toadd)

        if self.node_simfunc is not None:
            ring = list(sorted(graph.nodes(data=self.level)))
            return g_dgl, ring
        else:
            return g_dgl, 0


class UnsupervisedDataset(GraphDataset):
    """
    Basically just change the default of the loader based on the usecase
    """

    def __init__(self,
                 node_simfunc=SimFunctionNode('R_1', 2),
                 annotated=True,
                 chop=True,
                 **kwargs):
        super().__init__(annotated=annotated, chop=chop, node_simfunc=node_simfunc, **kwargs)


class SupervisedDataset(GraphDataset):
    """
    Basically just change the default of the loader based on the usecase
    """

    def __init__(self,
                 node_target='binding_protein',
                 annotated=False,
                 **kwargs):
        super().__init__(annotated=annotated, node_target=node_target, **kwargs)


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
            for ring in rings:
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
                 dataset,
                 batch_size=5,
                 num_workers=20,
                 max_size_kernel=None,
                 split=True,
                 verbose=False):
        """
        Turns a dataset into a dataloader

        :param dataset: The dataset to iterate over
        :param batch_size:
        :param num_workers:
        :param max_size_kernel: If we use K comptutations, we need to subsamble some nodes for the big graphs
        or else the k computation takes too long
        :param split: To return subsets to split the data
        :param verbose: To print some info about the data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_size_kernel = max_size_kernel
        self.split = split
        self.verbose = verbose

    def get_data(self):
        collate_block = collate_wrapper(self.dataset.node_simfunc, max_size_kernel=self.max_size_kernel)
        if not self.split:
            loader = DataLoader(dataset=self.dataset, shuffle=True, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=collate_block)
            return loader

        else:
            n = len(self.dataset)
            indices = list(range(n))
            np.random.seed(0)
            split_train, split_valid = 0.7, 0.85
            train_index, valid_index = int(split_train * n), int(split_valid * n)

            train_indices = indices[:train_index]
            valid_indices = indices[train_index:valid_index]
            test_indices = indices[valid_index:]

            train_set = Subset(self.dataset, train_indices)
            valid_set = Subset(self.dataset, valid_indices)
            test_set = Subset(self.dataset, test_indices)

            if self.verbose:
                print(f"training items: ", len(train_set))
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                     num_workers=self.num_workers, collate_fn=collate_block)
            return train_loader, valid_loader, test_loader


class InferenceLoader:
    def __init__(self,
                 list_to_predict,
                 data_path,
                 dataset=None,
                 batch_size=5,
                 num_workers=20,
                 **kwargs):
        if dataset is None:
            dataset = GraphDataset(data_path=data_path, **kwargs)
        self.dataset = dataset
        self.dataset.all_graphs = list_to_predict
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_data(self):
        collate_block = collate_wrapper(None)
        train_loader = DataLoader(dataset=self.dataset,
                                  shuffle=False,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=collate_block)
        return train_loader


class EdgeLoaderGenerator:
    def __init__(self,
                 graph_loader,
                 inner_batch_size=50,
                 sampler_layers=2,
                 neg_samples=1,
                 **kwargs):
        """
        This turns a graph dataloader or dataset into an edge data loader generator.
        It needs to be reinitialized every epochs because of the double iteration pattern

        Iterates over batches of base pairs and generates negative samples for each.
        Negative sampling is just uniform for the moment (eventually we should change it to only sample
        edges at a certain backbone distance.

        timing :
        - num workers should be used to load the graphs not in the inner loop
        - The inner batch size yields huge speedups (probably generating all MFGs is tedious)

        :param graph_loader:
        :param inner_batch_size:
        :param sampler_layers:
        :param neg_samples:
        :param num_workers:
        :param kwargs:
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

    def get_edge_loader(self):
        edge_loader = (EdgeDataLoader(g_batched, get_base_pairs(g_batched), self.sampler, **self.eloader_args)
                       for g_batched, _ in self.graph_loader)
        return edge_loader


class DefaultBasePairLoader:
    """ Just a default edge base pair loader.
    It deals with the splits
    """

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
        :param dataset:
        :param data_path:
        :param batch_size: This is the number of graphs that
        :param inner_batch_size:
        :param sampler_layers:
        :param neg_samples:
        :param num_workers:
        :param kwargs:
        """
        # Create default loaders
        if dataset is None:
            dataset = GraphDataset(data_path=data_path, **kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.g_train, self.g_val, self.g_test = Loader(self.dataset,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.num_workers).get_data()

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


def get_base_pairs(g):
    """ Returns edge IDS of edges in a base pair (non-backbone or unpaired).
    """
    eids = []
    for ind, e in enumerate(g.edata['edge_type']):
        if EDGE_MAP_RGLIB_REVERSE[e.item()][0] != 'B':
            eids.append(e)
    return eids


if __name__ == '__main__':
    pass
    import time

    node_features = ['nt_code', "alpha", "C5prime_xyz", "is_modified"]
    node_target = ['binding_ion']

    # GET THE DATA GOING
    toy_dataset = GraphDataset(data_path='data/graphs/all_graphs',
                               node_features=node_features,
                               node_target=node_target)
    train_loader, validation_loader, test_loader = Loader(dataset=toy_dataset,
                                                          batch_size=1,
                                                          num_workers=6).get_data()

    for i, item in enumerate(train_loader):
        print(item)
        if i > 10:
            break
        # if not i % 20: print(i)
        pass
