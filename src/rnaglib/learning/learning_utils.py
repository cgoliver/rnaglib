import os
import sys
import time

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.utils import misc


def weighted_MSE(output, target, weight):
    """
    Small utility function to compute the weighted mean square error loss

    :param output: tensor
    :param target: tensor
    :param weight: optional weighting tensor
    :return: the MSE loss
    """
    if weight is None:
        return torch.nn.MSELoss()(output, target)
    return torch.mean(weight * (output - target) ** 2)


def matrix_cosine(a, b, eps=1e-8):
    """
    Similar to pdist for cosine similarity. This is not implemented in Pytorch.

    :param a: List of vectors in the form of a tensor
    :param b: List of vectors in the form of a tensor
    :param eps: For numerical stability
    :return: The similarity matrix.
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def matrix_dist(a, plus_one=False):
    """
    Pairwise dist of a set of a vector of size b
    returns a matrix of size (a,a). This is a tad less efficient but more convenient than pdist.

    :param a : a torch Tensor of size a,b
    :param plus_one: if we want to get positive values
    """
    if plus_one:
        return torch.norm(a[:, None] - a, dim=2, p=2) + 1
    return torch.norm(a[:, None] - a, dim=2, p=2)


def get_nc_weight(graph, hops=2):
    """
    We want to give to each edge a higher weight if it is in the neighborhood of a non canonical edge.

    To do so, we first create a smaller adjacency matrix with just the non canonical and then we expand
    following all edges by multiplying by the adjacency matrix.

    Finally we perform scaling operations to get a weight with a mean of 1

    :param graph: a DiGraph
    :param hops: int, the number of expansion steps we want
    :return: a matrix weight
    """
    nx_graph = graph.to_networkx(edge_attrs=['edge_type'])
    nx_graph = nx.to_undirected(nx_graph)
    ordered = sorted(nx_graph.nodes())
    adj_matrix_full = nx.to_scipy_sparse_array(nx_graph, nodelist=ordered)

    edge_map = GRAPH_KEYS['edge_map'][TOOL]
    canonical = GRAPH_KEYS['canonical'][TOOL]

    # copy the matrix with only the non canonical
    canonicals_ids = {edge_map[key] for key in canonical if key in edge_map.keys()}
    extracted_edges = [(u, v) for u, v, e in nx_graph.edges.data('edge_type', default='0')
                       if e not in canonicals_ids]
    extracted_graph = nx.Graph()
    extracted_graph.add_nodes_from(ordered)
    extracted_graph.add_edges_from(extracted_edges)
    extracted_graph = nx.to_undirected(extracted_graph)
    adj_matrix_small = nx.to_scipy_sparse_array(extracted_graph, nodelist=ordered)

    # This is a matrix with non zero entries for non canonical relationships
    # One must then expand it based on the number of hops
    adj_matrix_full = np.array(adj_matrix_full.todense())
    adj_matrix_small = np.array(adj_matrix_small.todense())

    expanded_connectivity = [np.eye(len(adj_matrix_full))]
    for _ in range(hops):
        expanded_connectivity.append(expanded_connectivity[-1] @ adj_matrix_full)
    expanded_connectivity = np.sum(expanded_connectivity, axis=0)

    # What we are after is a matrix for which you start with a walk of len < max_len
    # that starts with node i and that ends with a non canonical with j
    # ie : all neighborhoods that include a non canonical.
    # multiplying on the left yields walks that start with a non canonical on the rows
    # expanded_connectivity_left = np.array(adj_matrix_small @ expanded_connectivity)
    expanded_connectivity_right = np.array(expanded_connectivity @ adj_matrix_small)
    enhanced = np.sum(expanded_connectivity_right, axis=0)
    enhanced = np.clip(enhanced, a_min=0, a_max=1)
    fraction = np.sum(enhanced) / len(enhanced)
    enhanced = ((1 / (fraction + 0.005)) * enhanced) + 1
    weight = np.outer(enhanced, enhanced)
    weight /= np.mean(weight)
    weight = torch.from_numpy(weight)
    return weight


def rec_loss(embeddings, target_K, similarity=True, normalize=False,
             use_graph=False, node_ids=None, graph=None, hops=2, device='cpu'):
    """
    This is to compute a reconstruction loss for embeddings.

    :param embeddings: The node embeddings
    :param target_K: The target distance/similarity matrix
    :param similarity: To use a similarity computation instead of a distance
    :param use_graph: This is to put extra focus on the parts of the graph that contain non canonical edges.
     We can input the graph too and adapt the reconstruction loss
    :param node_ids: The node ids are given as a one hot vector that represents the presence or absence of
     a non-canonical around a given node
    :param graph: we need to give graph as input in case we want to use use_graph
    :param hops : In case we use a graph, how many hops will be used around the given edge
    :return:
    """
    # First shape the K tensors, if we produce dot products or distances
    if similarity:
        if normalize:
            K_predict = matrix_cosine(embeddings, embeddings)
        else:
            K_predict = torch.mm(embeddings, embeddings.t())
    else:
        K_predict = matrix_dist(embeddings)
        target_K = torch.ones(target_K.shape, device=target_K.device) - target_K

    # Then optionnally compute a weighting tensor based on nc connectivity
    if use_graph:
        assert graph is not None
        graph_weight = get_nc_weight(graph=graph, hops=hops)
    else:
        graph_weight = torch.ones(size=(len(embeddings), len(embeddings)))
    graph_weight.to(K_predict.device)

    # Finally, subsample the prediction if needed
    if node_ids is not None:
        node_indices = np.argwhere(np.array(node_ids) > 0).squeeze()
        K_predict_1 = K_predict[node_indices]
        K_predict = K_predict_1[:, node_indices]

        graph_weight_1 = graph_weight[node_indices]
        graph_weight = graph_weight_1[:, node_indices]
    return weighted_MSE(K_predict, target_K, graph_weight)


def send_graph_to_device(g, device):
    """
    Send dgl graph to device, this is kind of deprecated in new versions of DGL

    :param g: a dgl graph
    :param device: a torch device
    :return: the graph on the device
    """
    import dgl

    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g


class LearningRoutine:
    def __init__(self,
                 validation_loader=None,
                 early_stop_threshold=60,
                 save_path=None,
                 writer=None,
                 best_loss=sys.maxsize,
                 device='cpu',
                 print_each=20,
                 num_epochs=25,
                 ):
        """
        A utility class for all learning routines: log writing, checkpointing, early stopping...
        It is also useful to pass all relevant objects from one function to another.

        :param num_epochs: The number of epochs
        :param print_each: The frequency with which we print information on learning
        :param device: The device on which to conduct all experiments
        :param writer: A writer object to write logs
        :param validation_loader: The validation loader that is used for validation. It also serves as a condition for
        performing early stopping : if one set it, early stopping will be used.
        :param save_path: The path where to save an early stopped model
        :param early_stop_threshold: The number of epochs without improvement before stopping.
        :param best_loss: To keep track of the current best loss
        """
        self.writer = writer
        self.best_loss = best_loss
        self.validation_loader = validation_loader
        self.save_path = save_path
        self.early_stop_threshold = early_stop_threshold
        self.epochs_from_best = 0
        self.device = device
        self.print_each = print_each
        self.num_epochs = num_epochs

    def early_stopping_routine(self, validation_loss, epoch, model, optimizer=None):
        """
        Based on the validation loss, update relevant parameters and optionally early stop the model

        :param validation_loss: A loss
        :param epoch: The epoch we are at, for
        :param model: The model to early stop
        :param optimizer: If we want to store the optimizer state
        :return: whether we early stopped
        """
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.epochs_from_best = 0

            if self.save_path is not None:
                model.cpu()
                print(">> saving checkpoint")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, self.save_path)
                model.to(self.device)
            return False

        # Early stopping
        else:
            self.epochs_from_best += 1
            if self.epochs_from_best > self.early_stop_threshold:
                print('This model was early stopped')
                return True


def compute_embeddings(model, validation_loader):
    """
    Compute the embeddings on a bunch of graphs

    :param model: The model to use
    :param validation_loader: A graph loader
    :return: a numpy array with all the embeddings
    """
    model.eval()
    device = model.current_device
    predicted = list()
    for batch_idx, batch in enumerate(validation_loader):
        # Get data on the devices
        graph, graph_sizes = batch['graphs'], batch['num_nodes']
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            out = model(graph)
            predicted.append(misc.tonumpy(out))
    predicted = np.concatenate(predicted)
    return predicted


def compute_outputs(model, validation_loader):
    """
    Just do the inference on a bunch of graphs

    :param model: The model to use
    :param validation_loader: A graph loader
    :return: two numpy arrays with all the supervisions and all the predictions
    """
    model.eval()
    device = model.current_device
    true, predicted = list(), list()
    for batch_idx, batch in enumerate(validation_loader):
        # Get data on the devices
        graph, graph_sizes = batch['graphs'], batch['num_nodes']
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            labels = graph.ndata['target']
            out = model(graph)

            true.append(misc.tonumpy(labels))
            predicted.append(misc.tonumpy(out))
    true = np.concatenate(true)
    predicted = np.concatenate(predicted)
    return true, predicted


def evaluate_model_unsupervised(model, validation_loader,
                                rec_params={"similarity": True, "normalize": False, "use_graph": False, "hops": 2}):
    """
    Simply get the score output for unsupervised training.

    :param model: The model to use
    :param validation_loader: A graph loader
    :param rec_params: The parameters of the loss
    :return: The loss value on the validation set
    """
    model.eval()
    device = model.current_device
    test_size = len(validation_loader)
    recons_loss_tot = 0
    for batch_idx, batch in enumerate(validation_loader):
        graph, graph_sizes, (K, node_ids) = batch['graphs'], batch['num_nodes'], batch['node_similarities']
        # Get data on the devices
        K = K.to(device)
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            out = model(graph)
            reconstruction_loss = rec_loss(embeddings=out,
                                           target_K=K,
                                           graph=graph,
                                           node_ids=node_ids,
                                           **rec_params)
            recons_loss_tot += reconstruction_loss
    return recons_loss_tot / test_size


def evaluate_model_supervised(model, loader, evaluation_function=roc_auc_score):
    """
    Make the inference and apply an evaluation function on it

    :param model: The model to use
    :param loader: A graph loader
    :param evaluation_function: A function that takes two np arrays and returns a score
    :return: The validation score for this evaluation function
    """
    true, predicted = compute_outputs(model, loader)
    score = evaluation_function(true, predicted)
    return score
