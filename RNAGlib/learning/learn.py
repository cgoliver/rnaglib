import os
import sys

import time

import numpy as np
import torch
from torch.utils.data import Subset
import dgl
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from config.graph_keys import GRAPH_KEYS, TOOL
from utils import misc


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
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


def evaluate_model_unsupervised(model, validation_loader):
    model.eval()
    device = model.current_device
    test_size = len(validation_loader)
    recons_loss_tot = 0
    for batch_idx, (graph, K, inds, graph_sizes) in enumerate(validation_loader):
        # Get data on the devices
        K = K.to(device)
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            out = model(graph)
            reconstruction_loss = rec_loss(embeddings=out,
                                           target_K=K,
                                           graph=graph)
            recons_loss_tot += reconstruction_loss
    return recons_loss_tot / test_size


def compute_outputs(model, validation_loader):
    model.eval()
    device = model.current_device
    true, predicted = list(), list()
    for batch_idx, (graph, graph_sizes) in enumerate(validation_loader):
        # Get data on the devices
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


def evaluate_model_supervised(model, validation_loader, evaluation_function=roc_auc_score):
    """
    Make the inference and apply an evalutation function on it

    :param model:
    :param validation_loader:
    :param evaluation_function:
    :return:
    """
    true, predicted = compute_outputs(model, validation_loader)
    score = evaluation_function(true, predicted)
    return score


def evaluate_model_supervised_deprecated(model, validation_loader,
                                         evaluation_function=lambda x, y: torch.nn.MSELoss(x, y).item()):
    model.eval()
    device = model.current_device
    test_size = len(validation_loader)
    loss_tot = 0
    for batch_idx, (graph, graph_sizes) in enumerate(validation_loader):
        # Get data on the devices
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            out = model(graph)
            labels = graph.ndata['target']
            loss = evaluation_function(out, labels)
            loss_tot += loss
    return loss_tot / test_size


class EarlyStopper:
    def __init__(self,
                 test_loader,
                 save_path,
                 early_stop_threshold=60
                 ):
        self.test_loader = test_loader
        self.save_path = save_path
        self.early_stop_threshold = early_stop_threshold


def weighted_MSE(output, target, weight):
    if weight is None:
        return torch.nn.MSELoss()(output, target)
    return torch.mean(weight * (output - target) ** 2)


def matrix_cosine(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def matrix_dist(a, plus_one=False):
    """
    Pairwise dist of a set of a vector of size b
    returns a matrix of size (a,a)
    :param a : a torch Tensor of size a,b
    :param plus_one: if we want to get positive values
    """
    if plus_one:
        return torch.norm(a[:, None] - a, dim=2, p=2) + 1
    return torch.norm(a[:, None] - a, dim=2, p=2)


def rec_loss(embeddings, target_K, similarity=True, normalize=False, use_graph=False, node_ids=None, graph=None,
             hops=2):
    """
    :param embeddings: The node embeddings
    :param target_K: The similarity matrix
    :param similarity: To use a similarity computation instead of a distance
    :param target_K: If we use a similarity, we can choose to normalize (cosine) or to just take a dot product
    :param use_graph: This is to put extra focus on the parts of the graph that contain non canonical edges.
     We can input the graph too and adapt the reconstruction loss
    :param node_ids: The node ids are given as a one hot vector that represents the presence or absence of
     a non-canonical around a given node
    :param graph: we need to give graph as input in case we want to use use_graph
    :param hops : In case we use a graph, how many hops will be used around the given edge
    :return:
    """
    if similarity:
        if normalize:
            K_predict = matrix_cosine(embeddings, embeddings)
        else:
            K_predict = torch.mm(embeddings, embeddings.t())

    else:
        K_predict = matrix_dist(embeddings)
        target_K = torch.ones(target_K.shape, device=target_K.device) - target_K

    # Cannot have both for now
    # assert node_ids is None or not use_graph
    if node_ids is not None:
        node_indices = np.argwhere(np.array(node_ids) > 0).squeeze()
        K_predict_1 = K_predict[node_indices]
        K_predict = K_predict_1[:, node_indices]

    if not use_graph:
        reconstruction_loss = torch.nn.MSELoss()(K_predict, target_K)
        return reconstruction_loss

    else:
        assert graph is not None
        import networkx as nx
        nx_graph = graph.to_networkx(edge_attrs=['edge_type'])
        nx_graph = nx.to_undirected(nx_graph)
        ordered = sorted(nx_graph.nodes())
        adj_matrix_full = nx.to_scipy_sparse_matrix(nx_graph, nodelist=ordered)

        edge_map = GRAPH_KEYS['edge_map'][TOOL]
        canonicals = {edge_map[key] for key in ['B53', 'B35', 'cWW', 'CWW']
                      if key in edge_map.keys()}
        # copy the matrix with only the non canonical
        extracted_edges = [(u, v) for u, v, e in nx_graph.edges.data('edge_type', default='0')
                           if e not in canonicals]
        extracted_graph = nx.Graph()
        extracted_graph.add_nodes_from(ordered)
        extracted_graph.add_edges_from(extracted_edges)
        extracted_graph = nx.to_undirected(extracted_graph)
        adj_matrix_small = nx.to_scipy_sparse_matrix(extracted_graph, nodelist=ordered)

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

        if node_ids is not None:
            node_indices = np.argwhere(np.array(node_ids) > 0).squeeze()
            weight_1 = weight[node_indices]
            weight = weight_1[:, node_indices]
        return weighted_MSE(K_predict, target_K, weight)


def pretrain_unsupervised(model,
                          node_sim,
                          train_loader,
                          optimizer,
                          num_epochs=25,
                          early_stopper=None,
                          writer=None,
                          rec_params={"similarity": True, "normalize": False, "use_graph": False, "hops": 2},
                          print_each=20
                          ):
    device = model.current_device
    # We need to access the GraphDataset object, which is accessed differently when using subsets.
    # This is the recommended fix in https://discuss.pytorch.org/t/how-to-get-a-part-of-datasets/82161/7
    if isinstance(train_loader, Subset):
        train_loader.dataset.dataset.add_node_sim(node_simfunc=node_sim)
    else:
        train_loader.dataset.add_node_sim(node_simfunc=node_sim)
    if early_stopper is not None:
        early_stopper.dataset.dataset.test_loader.add_node_sim(node_simfunc=node_sim)
    epochs_from_best = 0
    start_time = time.time()
    best_loss = sys.maxsize
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, (graph, K, graph_sizes, node_ids) in enumerate(train_loader):
            # Get data on the devices
            K = K.to(device)
            graph = send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            out = model(graph)
            loss = rec_loss(embeddings=out,
                            target_K=K,
                            graph=graph,
                            node_ids=node_ids,
                            **rec_params)
            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            loss = loss.item()
            running_loss += loss

            if batch_idx % print_each == 0:
                time_elapsed = time.time() - start_time

                print(
                    f'Train Epoch: {epoch + 1} [{(batch_idx + 1)}/{num_batches} '
                    f'({100. * (batch_idx + 1) / num_batches:.0f}%)]\t'
                    f'Loss: {loss:.6f}  Time: {time_elapsed:.2f}')

                # tensorboard logging
                step = epoch * num_batches + batch_idx
                if writer is not None:
                    writer.add_scalar("Training loss", loss, step)

        # # Log training metrics
        train_loss = running_loss / num_batches
        if writer is not None:
            writer.add_scalar("Training epoch loss", train_loss, epoch)

        # Test phase
        if early_stopper is None:
            if running_loss < best_loss:
                best_loss = running_loss
            continue
        else:
            test_loss = evaluate_model_unsupervised(model, validation_loader=early_stopper.test_loader)

            if writer is not None:
                writer.add_scalar("Test loss during training", test_loss, epoch)

            # Checkpointing
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_from_best = 0

                model.cpu()
                print(">> saving checkpoint")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, early_stopper.save_path)
                model.to(device)

            # Early stopping
            else:
                epochs_from_best += 1
                if epochs_from_best > early_stopper.early_stop_threshold:
                    print('This model was early stopped')
                    break
    return best_loss


def train_supervised(model,
                     optimizer,
                     train_loader,
                     num_epochs=25,
                     early_stopper=None,
                     writer=None,
                     print_each=20):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param train_loader: The loader to use for training
    :param writer: a pytorch writer object
    :param num_epochs: int number of epochs
    :param early_stopper: An early stopper object, if we want to also use a validation phase and early stopping
    :return:
    """
    device = model.current_device
    epochs_from_best = 0
    early_stop_threshold = 60

    start_time = time.time()
    best_loss = sys.maxsize
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (graph, graph_sizes) in enumerate(train_loader):
            # Get data on the devices
            graph = send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            out = model(graph)
            labels = graph.ndata['target']
            loss = torch.nn.MSELoss()(out, labels)
            # preds = (logits > threshold).float()
            # acc = (preds*labels).float().mean()

            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            loss = loss.item()
            running_loss += loss

            if batch_idx % print_each == 0:
                time_elapsed = time.time() - start_time
                print(
                    f'Train Epoch: {epoch + 1} [{(batch_idx + 1)}/{num_batches} '
                    f'({100. * (batch_idx + 1) / num_batches:.0f}%)]\t'
                    f'Loss: {loss:.6f}  Time: {time_elapsed:.2f}')

                # tensorboard logging
                if writer is not None:
                    step = epoch * num_batches + batch_idx
                    writer.add_scalar("Training loss", loss, step)

        if writer is not None:
            # # Log training metrics
            train_loss = running_loss / num_batches
            writer.add_scalar("Training epoch loss", train_loss, epoch)

        # Test phase
        # Test phase
        if early_stopper is None:
            if running_loss < best_loss:
                best_loss = running_loss
            continue
        else:
            test_loss = evaluate_model_supervised(model, validation_loader=early_stopper.test_loader)

            if writer is not None:
                writer.add_scalar("Test loss during training", test_loss, epoch)

            # Checkpointing
            if test_loss < best_loss:
                best_loss = test_loss
                epochs_from_best = 0

                model.cpu()
                print(">> saving checkpoint")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, early_stopper.save_path)
                model.to(device)

            # Early stopping
            else:
                epochs_from_best += 1
                if epochs_from_best > early_stopper.early_stop_threshold:
                    print('This model was early stopped')
                    break
    return best_loss


if __name__ == '__main__':
    pass
    from learning import models
    from data_loading import loader

    test_unsupervised = False
    test_supervised = True
    if test_unsupervised:
        from kernels import node_sim

        embedder_model = models.Embedder([10, 10])
        optimizer = torch.optim.Adam(embedder_model.parameters())
        node_sim_func = node_sim.SimFunctionNode(method='R_1', depth=2)

        data_path = os.path.join(script_dir, '../data/annotated/samples/')
        data_loader = loader.Loader(data_path=data_path,
                                    num_workers=0,
                                    batch_size=4,
                                    max_size_kernel=100,
                                    node_simfunc=node_sim_func)
        train_loader, validation_loader, test_loader = data_loader.get_data()

        pretrain_unsupervised(model=embedder_model,
                              optimizer=optimizer,
                              node_sim=node_sim_func,
                              train_loader=train_loader
                              )

    if test_supervised:
        data_path = os.path.join(script_dir, '../data/annotated/samples/')

        annotated_path = "../data/graphs"
        node_features = ['nt_code']
        node_target = ['binding_protein']

        # Define model
        # GET THE DATA GOING
        loader = loader.SupervisedLoader(data_path=annotated_path,
                                         node_features=node_features,
                                         node_target=node_target,
                                         num_workers=2)
        train_loader, validation_loader, test_loader = loader.get_data()

        embedder_model = models.Embedder([10, 10], infeatures_dim=1)
        classifier_model = models.Classifier(embedder=embedder_model, last_dim_embedder=10, classif_dims=[1])
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
        data_path = os.path.join(script_dir, '../data/graphs/')

        train_supervised(model=classifier_model,
                         optimizer=optimizer,
                         train_loader=train_loader)
