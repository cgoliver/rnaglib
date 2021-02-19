import os
import configparser
from ast import literal_eval
import pickle

from tqdm import tqdm
import torch
import dgl
import numpy as np
import networkx as nx

import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..', 'vernal'))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from train_embeddings.loader import Loader, loader_from_hparams
from train_embeddings.model import Model, model_from_hparams
from train_embeddings.learn import send_graph_to_device
from tools.graph_utils import fetch_graph, get_nc_nodes_index


def remove(name):
    """
    delete an experiment results
    :param name:
    :return:
    """
    import shutil

    logdir = os.path.join(script_dir, f'../results/logs/{name}')
    weights_dir = os.path.join(script_dir, f'../results/trained_models/{name}')
    experiment = os.path.join(script_dir, f'../results/experiments/{name}.exp')
    shutil.rmtree(logdir)
    shutil.rmtree(weights_dir)
    os.remove(experiment)
    return True


def setup():
    """
    Create all relevant directories to setup the learning procedure
    :return:
    """
    script_dir = os.path.dirname(__file__)
    resdir = os.path.join(script_dir, f'../results/')
    logdir = os.path.join(script_dir, f'../results/logs/')
    weights_dir = os.path.join(script_dir, f'../results/trained_models/')
    os.mkdir(resdir)
    os.mkdir(logdir)
    os.mkdir(weights_dir)


def mkdirs_learning(name, permissive=True):
    """
    Try to make the logs folder for each experiment
    :param name:
    :param permissive: If True will overwrite existing files (good for debugging)
    :return:
    """
    from tools.utils import makedir
    makedir(os.path.join(script_dir, '../results'), permissive)
    makedir(os.path.join(script_dir, '../results/logs'), permissive)
    makedir(os.path.join(script_dir, '../results/trained_models'), permissive)

    log_path = os.path.join(script_dir, '../results/logs', name)
    save_path = os.path.join(script_dir, '../results/trained_models', name)
    makedir(log_path, permissive)
    makedir(save_path, permissive)
    save_name = os.path.join(save_path, name + '.pth')
    return log_path, save_name


def meta_load_model(run):
    """
        Load full trained model with id `run`

    """

    meta = pickle.load(open(f'../results/trained_models/{run}/meta.p', 'rb'))

    edge_map = meta['edge_map']
    num_edge_types = len(edge_map)
    num_edge_types = 13

    model_dict = torch.load(f'../results/trained_models/{run}/{run}.pth', map_location='cpu')
    model = Model(dims=meta['dims'], num_rels=num_edge_types,
                  num_bases=-1, hard_embed=meta['hard_embed'])
    model.load_state_dict(model_dict['model_state_dict'])
    return model, meta


def meta_load_data(annotated_path, meta, get_sim_mat=True, all_graphs=False):
    """

        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
    """
    from learning.loader import Loader

    loader = Loader(annotated_path=annotated_path,
                    batch_size=1, num_workers=1,
                    sim_function=meta['sim_function'],
                    depth=meta['depth'],
                    hard_embed=meta['hard_embed'],
                    get_sim_mat=get_sim_mat)

    return loader.get_data()


def run_to_hparams(run):
    exp_path = os.path.join(script_dir, f'../results/trained_models/{run}/{run}.exp')
    hparams = ConfParser(default_path=os.path.join(script_dir, '../train_embeddings/inis/default.ini'),
                         path_to_ini=exp_path)
    return hparams


def load_model(run, permissive=False, verbose=True):
    """
    Input the name of a run
    :param run:
    :return:
    """
    hparams = run_to_hparams(run)
    hparams.add_value('argparse', 'num_edge_types', 13)
    model = model_from_hparams(hparams, verbose=verbose)
    # print(hparams)
    try:
        model_dict = torch.load(os.path.join(script_dir, f'../results/trained_models/{run}/{run}.pth')
                                , map_location='cpu')
        state_dict = model_dict['model_state_dict']
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        if not permissive:
            raise FileNotFoundError('There are no weights for this experiment...')
    return model


default_edge_map = {'B53': 0, 'CHH': 1, 'CHS': 2, 'CHW': 3, 'CSH': 2, 'CSS': 4, 'CSW': 5, 'CWH': 3, 'CWS': 5, 'CWW': 6,
                    'THH': 7, 'THS': 8, 'THW': 9, 'TSH': 8, 'TSS': 10, 'TSW': 11, 'TWH': 9, 'TWS': 11, 'TWW': 12}


def inference_on_graph(model,
                       graph,
                       edge_map=default_edge_map,
                       device='cpu',
                       nc_only=False):
    """
        Do inference on one networkx graph.
    """
    graph = nx.to_undirected(graph)
    one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in
               (nx.get_edge_attributes(graph, 'label')).items()}
    nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

    g_dgl = dgl.DGLGraph()
    g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
    g_dgl = send_graph_to_device(g_dgl, device)
    model = model.to(device)
    with torch.no_grad():
        embs = model(g_dgl)
        embs.cpu().numpy()
    g_nodes = list(sorted(graph.nodes()))

    keep_indices = range(len(graph.nodes()))

    if nc_only:
        keep_indices = get_nc_nodes_index(graph)

    node_map = {g_nodes[node_ind]: i for i, node_ind in enumerate(keep_indices)}
    embs = embs[keep_indices]
    return embs, node_map


def inference_on_graph_run(run,
                           graph,
                           device='cpu',
                           verbose=True,
                           nc_only=False):
    """
        Do inference on one networkx graph.
    """
    hparams = run_to_hparams(run)
    model = load_model(run, verbose=verbose)
    edge_map = hparams.get('edges', 'edge_map')

    return inference_on_graph(model=model,
                              graph=graph,
                              edge_map=edge_map,
                              device=device,
                              nc_only=nc_only)


def inference_on_dir(run,
                     graph_dir,
                     ini=True,
                     max_graphs=10,
                     get_sim_mat=False,
                     split_mode='test',
                     nc_only=False,
                     device='cpu'):
    """
        Load model and get node embeddings.

        The results then need to be parsed as the order of the graphs is random and that the order of
        each node in the graph is the messed up one (sorted)

        Returns : embeddings and attributions, as well as 'g_inds':
        a dict (graph name, node_id in sorted g_nodes) : index in the embedding matrix

        :params
        :get_sim_mat: switches off computation of rings and K matrix for faster loading.
        :max_graphs max number of graphs to get embeddings for
    """

    loader_mode = {'train': 0, 'test': 1, 'all': 2}
    if ini:
        hparams = run_to_hparams(run)
        model = load_model(run)
        loader = loader_from_hparams(graph_dir, hparams)
        # No mat needed here
        if not get_sim_mat:
            loader.node_simfunc = None
        loader = loader.get_data()[loader_mode[split_mode]]

    else:
        model, meta = meta_load_model(run)

        loaders = meta_load_data(graph_dir, meta, get_sim_mat=get_sim_mat, all_graphs=False)
        loader = loaders[loader_mode[split_mode]]

    model_outputs = predict(model,
                            loader,
                            max_graphs=max_graphs,
                            get_sim_mat=get_sim_mat,
                            nc_only=nc_only,
                            device=device)
    return model_outputs


def inference_on_list(run,
                      graphs_path,
                      graph_list,
                      max_graphs=None,
                      get_sim_mat=False,
                      nc_only=False,
                      device='cuda' if torch.cuda.is_available() else 'cpu'
                      ):
    """
    Same as before but one needs to provide a list of graphs name files in the annot path (of the form id_chunk_annot.p)
    :param run:
    :param graph_dir:
    :param max_graphs:
    :param get_sim_mat:
    :param split_mode:
    :param device:
    :return:
    """

    hparams = run_to_hparams(run)
    model = load_model(run)
    inference_loader = loader_from_hparams(annotated_path=graphs_path,
                                           hparams=hparams,
                                           list_inference=graph_list
                                           )
    loader = inference_loader.get_data()
    model_outputs = predict(model,
                            loader,
                            max_graphs=max_graphs,
                            get_sim_mat=get_sim_mat,
                            nc_only=nc_only,
                            device=device
                            )
    return model_outputs


def inference_on_list_gen(run,
                          graphs_path,
                          graph_list,
                          max_graphs=None,
                          get_sim_mat=False,
                          device='cpu',
                          batch_size=0):
    """
    Same as before but one needs to provide a list of graphs name files in the annot path (of the form id_chunk_annot.p)
    :param run:
    :param graph_dir:
    :param max_graphs:
    :param get_sim_mat:
    :param split_mode:
    :param device:
    :return:
    """

    hparams = run_to_hparams(run)
    if batch_size:
        hparams.hparams.set('argparse', 'bathc_size', '1')
    model = load_model(run)
    inference_loader = loader_from_hparams(annotated_path=graphs_path, hparams=hparams, list_inference=graph_list)
    loader = inference_loader.get_data()
    gen = predict_gen(model,
                      loader,
                      max_graphs=max_graphs,
                      get_sim_mat=get_sim_mat,
                      device=device)
    for stuff in gen:
        yield stuff


def predict(model,
            loader,
            max_graphs=10,
            nc_only=False,
            get_sim_mat=False,
            device='cpu'):
    """

    :param model:
    :param loader:
    :param max_graphs:
    :param get_sim_mat:
    :param device:
    :return:
    """

    all_graphs = loader.dataset.all_graphs
    graph_dir = loader.dataset.path
    Z = []
    Ks = []
    g_inds = []
    node_ids = []
    tot = max_graphs if max_graphs is not None else len(loader)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # For each batch, we have the graph, its index in the list and its size
        # for i, (graph, K, graph_indices, graph_sizes) in enumerate(loader):
        for i, (graph, K, graph_indices, graph_sizes) in tqdm(enumerate(loader), total=tot):
            if get_sim_mat:
                Ks.append(K)
            graph_indices = list(graph_indices.numpy().flatten())
            keep_Z_indices = []
            offset = 0
            for graph_index, n_nodes in zip(graph_indices, graph_sizes):
                # For each graph, we build an id list in rep
                # that contains all the nodes

                keep_indices = list(range(n_nodes))

                # list of node ids from original graph
                g_path = os.path.join(graph_dir, all_graphs[graph_index])
                G = fetch_graph(g_path)

                assert n_nodes == len(G.nodes())
                if nc_only:
                    keep_indices = get_nc_nodes_index(G)
                keep_Z_indices.extend([ind + offset for ind in keep_indices])

                rep = [(all_graphs[graph_index], node_index)
                       for node_index in keep_indices]

                g_inds.extend(rep)

                g_nodes = sorted(G.nodes())
                node_ids.extend([g_nodes[i] for i in keep_indices])

                offset += n_nodes

            graph = send_graph_to_device(graph, device)
            z = model(graph)
            z = z.cpu().numpy()
            z = z[keep_Z_indices]
            Z.append(z)

            if max_graphs is not None and i > max_graphs - 1:
                break

    Z = np.concatenate(Z)

    # node to index in graphlist
    g_inds = {value: i for i, value in enumerate(g_inds)}
    # node to index in Z
    node_map = {value: i for i, value in enumerate(node_ids)}
    # index in Z to node
    node_map_r = {i: value for i, value in enumerate(node_ids)}

    return {'Z': Z,
            'node_to_gind': g_inds,
            'node_to_zind': node_map,
            'ind_to_node': node_map_r,
            'node_id_list': node_ids
            }


def predict_gen(model,
                loader,
                max_graphs=10,
                nc_only=False,
                get_sim_mat=False,
                device='cpu'):
    """
    Yield embeddings one batch at a time.

    :param model:
    :param loader:
    :param max_graphs:
    :param get_sim_mat:
    :param device:
    :return:
    """
    all_graphs = loader.dataset.all_graphs
    graph_dir = loader.dataset.path

    model = model.to(device)
    Ks = []
    with torch.no_grad():
        # For each batch, we have the graph, its index in the list and its size
        for i, (graph, K, graph_indices, graph_sizes) in tqdm(enumerate(loader), total=len(loader)):
            if get_sim_mat:
                Ks.append(K)

            Z = []
            Ks = []
            g_inds = []
            node_ids = []

            graph_indices = list(graph_indices.numpy().flatten())
            for graph_index, n_nodes in zip(graph_indices, graph_sizes):
                # For each graph, we build an id list in rep that contains all the nodes
                rep = [(all_graphs[graph_index], node_index) for node_index in range(n_nodes)]
                g_inds.extend(rep)

                # list of node ids from original graph
                g_path = os.path.join(graph_dir, all_graphs[graph_index])
                G = fetch_graph(g_path)
                g_nodes = sorted(G.nodes())
                node_ids.extend([g_nodes[i] for i in range(n_nodes)])
            if max_graphs is not None and i > max_graphs - 1:
                raise StopIteration

            graph = send_graph_to_device(graph, device)
            z = model(graph)

            Z.append(z.cpu().numpy())
            Z = np.concatenate(Z)
            g_inds = {value: i for i, value in enumerate(g_inds)}
            node_map = {value: i for i, value in enumerate(node_ids)}
            yield Z, g_inds, node_map


# def parse_predictions(preds, graph_dir, hparams, run, map, split_mode='test', ini=True):
#     from tools.graph_utils import dgl_to_nx
#     loader_mode = {'train': 0, 'test': 1, 'all': 2}
#     if ini:
#         hparams = run_to_hparams(run)
#         loader = loader_from_hparams(graph_dir, hparams)
#         loader = loader.get_data()[loader_mode[split_mode]]
#
#     else:
#         model, meta = meta_load_model(run)
#         loaders = meta_load_data(graph_dir, meta, get_sim_mat=False, all_graphs=False)
#         loader = loaders[loader_mode[split_mode]]
#
#     # maps full nodeset index to graph and node index inside graph
#     node_map = {}
#
#     graphs_path = loader.dataset.dataset.path
#     all_graphs = loader.dataset.dataset.all_graphs
#     all_graphs = [s.decode("utf-8") for s in all_graphs]
#
#     current_g_ind = -1
#     for j, g_ind in enumerate(g_inds):
#         # To avoid reloading the graph everytime
#         if g_ind != current_g_ind:
#             current_g_ind = g_ind
#             G, _, _ = pickle.load(open(os.path.join(graphs_path, all_graphs[g_ind]), 'rb'))
#             G_nodes = iter(sorted(G.nodes(data=True)))
#
#         node_info = next(G_nodes)
#
#         node_map[ind] = (i, j, node_info, all_graphs[g_ind])
#         ind += 1
#     # nx_graphs.append(nx_graph)
#     nx_g = dgl_to_nx(graph, edge_map)
#     # assign unique id to graph nodes
#     nx_g = nx.relabel.convert_node_labels_to_integers(nx_g, first_label=offset, label_attribute='id')
#     offset += len(nx_g.nodes())
#     # print(z)
#     # rna_draw(nx_g)
#     nx_graphs.append(nx_g)
#     pass
#
#
#     return nx_graphs, Z, Sigma, Ks, node_map, similarity


class ConfParser:
    def __init__(self,
                 default_path=os.path.join(os.path.dirname(__file__), 'results/default.ini'),
                 path_to_ini=None,
                 argparse=None,
                 dump_path=None):
        self.dump_path = dump_path
        self.default_path = default_path

        # Build the hparam object
        self.hparams = configparser.ConfigParser()

        # Add the default configurations, optionaly another .conf and an argparse object
        self.hparams.read(self.default_path)

        if path_to_ini is not None:
            # print('confing')
            self.add_ini(path_to_ini)
        if argparse is not None:
            self.add_argparse(argparse)

    @staticmethod
    def merge_ini_into_default(default, new):
        for section in new.sections():
            for keys in new[section]:
                try:
                    default[section][keys]
                except KeyError:
                    raise KeyError(f'The provided value {section, keys} in the .ini are not present in the default, '
                                   f'thus not acceptable values, for retro-compatibility issues')
                # print(section, keys)
                default[section][keys] = new[section][keys]

    def add_ini(self, path_to_new):
        """
        Merge another conf parsing into self.hparams
        :param path_to_new:
        :return:
        """
        conf = configparser.ConfigParser()
        assert os.path.exists(path=path_to_new), f"failed on {path_to_new}"
        conf.read(path_to_new)
        # print(f'confing using {path_to_new}')
        return self.merge_ini_into_default(self.hparams, conf)

    @staticmethod
    def merge_dict_into_default(default, new):
        """
        Same merge but for a dict of dicts
        :param default:
        :param new:
        :return:
        """
        for section in new.sections():
            for keys in new[section]:
                try:
                    default[section][keys]
                except KeyError:
                    raise KeyError(f'The provided value {section, keys} in the .conf are not present in the default, '
                                   f'thus not acceptable values')
                default[section][keys] = new[section][keys]
        return default

    def add_dict(self, section_name, dict_to_add):
        """
        Add a dictionnary as a section of the .conf. It needs to be turned into strings
        :param section_name: string to be the name of the section
        :param dict_to_add: any dictionnary
        :return:
        """

        new = {item: str(value) for item, value in dict_to_add.items()}

        try:
            self.hparams[section_name]
        # If it does not exist
        except KeyError:
            self.hparams[section_name] = new
            return

        for keys in new:
            self.hparams[section_name][keys] = new[keys]

    def add_value(self, section_name, key, value):
        """
        Add a dictionnary as a section of the .conf. It needs to be turned into strings
        :param section_name: string to be the name of the section
        :param key: the key inside the section name
        :param value: the value to insert
        :return:
        """

        value = str(value)

        try:
            self.hparams[section_name]
        # If it does not exist
        except KeyError:
            self.hparams[section_name] = {key: value}
            return

        self.hparams[section_name][key] = value

    def add_argparse(self, argparse_obj):
        """
        Add the argparse object as a section of the .conf
        :param argparse_obj:
        :return:
        """
        self.add_dict('argparse', argparse_obj.__dict__)

    def get(self, section, key):
        """
        A get function that also does the casting into what is useful for model results
        :param section:
        :param key:
        :return:
        """
        try:
            return literal_eval(self.hparams[section][key])
        except ValueError:
            return self.hparams[section][key]
        except SyntaxError:
            return self.hparams[section][key]

    def __str__(self):
        print(self.hparams.sections())
        for section in self.hparams.sections():
            print(section.upper())
            for keys in self.hparams[section]:
                print(keys)
            print('-' * 10)
        return ' '

    def dump(self, dump_path=None):
        """
        If no dump is given, use the default one if it exists. Otherwise, set the dumping path as the new default
        :param dump_path:
        :return:
        """
        if dump_path is None:
            if self.dump_path is None:
                raise ValueError('Please add a path')
            with open(self.dump_path, 'w') as save_path:
                self.hparams.write(save_path)
        else:
            self.dump_path = dump_path
            with open(dump_path, 'w') as save_path:
                self.hparams.write(save_path)
