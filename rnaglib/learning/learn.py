import os
import sys
import time

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn.functional as F
import dgl

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.utils import misc


def send_graph_to_device(g, device):
    """
    Send dgl graph to device, this is kind of deprecated in new versions of DGL
    :param g: a dgl graph
    :param device: a torch device
    :return: the graph on the device
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


class LearningRoutine:
    """
    A utility class for all learning routines:
     log writing, checkpointing, early stopping...
    """

    def __init__(self,
                 validation_loader=None,
                 save_path=None,
                 early_stop_threshold=60,
                 writer=None,
                 best_loss=sys.maxsize,
                 device='cpu',
                 print_each=20,
                 num_epochs=25,
                 ):
        self.writer = writer
        self.best_loss = best_loss
        self.validation_loader = validation_loader
        self.save_path = save_path
        self.early_stop_threshold = early_stop_threshold
        self.epochs_from_best = 0
        self.device = device
        self.print_each = print_each
        self.num_epochs = num_epochs

    def early_stopping_routine(self, validation_loss, epoch, model, optimizer, validation=True):
        """

        :param validation_loss:
        :param epoch:
        :param model:
        :param validation: whether the validation loss is actually a validation loss.
        If no loader is given, we simply use the training value
        :return:
        """
        if self.writer is not None and validation:
            self.writer.add_scalar("Validation loss during training", validation_loss, epoch)

        # Checkpointing
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


def get_nc_weight(graph, hops=2):
    """
    We want to give to each node a higher weight if it is in the neighborhood of a non canonical edge.

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
    adj_matrix_full = nx.to_scipy_sparse_matrix(nx_graph, nodelist=ordered)

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
    return weight


def rec_loss(embeddings, target_K, similarity=True, normalize=False,
             use_graph=False, node_ids=None, graph=None, hops=2):
    """
    This is to compute a reconstruction loss for embeddings.

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


def evaluate_model_unsupervised(model, validation_loader,
                                rec_params={"similarity": True, "normalize": False, "use_graph": False, "hops": 2}):
    """
    Simply get the score output for unsupervised training.
    :param model:
    :param validation_loader:
    :param rec_params:
    :return:
    """
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
                                           graph=graph,
                                           **rec_params)
            recons_loss_tot += reconstruction_loss
    return recons_loss_tot / test_size


def compute_outputs(model, validation_loader):
    """
    Just do the inference on a bunch of graphs
    :param model:
    :param validation_loader:
    :return:
    """
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


def pretrain_unsupervised(model,
                          train_loader,
                          optimizer,
                          node_sim=None,
                          learning_routine=LearningRoutine(),
                          rec_params={"similarity": True, "normalize": False, "use_graph": False, "hops": 2}
                          ):
    """

    :param model:
    :param train_loader:
    :param optimizer:
    :param node_sim: If None, we just rely on the node_sim in the data loader.
    :param learning_routine:
    :param rec_params: These are parameters useful for the loss computation
    :return:
    """
    device = model.current_device
    learning_routine.device = device

    misc.get_dataset(train_loader).update_node_sim(node_simfunc=node_sim)
    if learning_routine.validation_loader is not None:
        misc.get_dataset(learning_routine.validation_loader).update_node_sim(node_simfunc=node_sim)

    start_time = time.time()
    for epoch in range(learning_routine.num_epochs):
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

            if batch_idx % learning_routine.print_each == 0:
                time_elapsed = time.time() - start_time

                print(
                    f'Train Epoch: {epoch + 1} [{(batch_idx + 1)}/{num_batches} '
                    f'({100. * (batch_idx + 1) / num_batches:.0f}%)]\t'
                    f'Loss: {loss:.6f}  Time: {time_elapsed:.2f}')

                # tensorboard logging
                step = epoch * num_batches + batch_idx
                if learning_routine.writer is not None:
                    learning_routine.writer.add_scalar("Training loss", loss, step)

        train_loss = running_loss / num_batches
        if learning_routine.writer is not None:
            learning_routine.writer.add_scalar("Training epoch loss", train_loss, epoch)

        # Test phase, if we do not have a validation, just iterate.
        # Otherwise call the routines.
        if learning_routine.validation_loader is None:
            early_stop = learning_routine.early_stopping_routine(validation_loss=train_loss, epoch=epoch,
                                                                 model=model, optimizer=optimizer, validation=False)
        else:
            validation_loss = evaluate_model_unsupervised(model,
                                                          validation_loader=learning_routine.validation_loader,
                                                          rec_params=rec_params)
            early_stop = learning_routine.early_stopping_routine(validation_loss=validation_loss, epoch=epoch,
                                                                 model=model, optimizer=optimizer)
        if early_stop:
            break
    return learning_routine.best_loss


def train_supervised(model,
                     optimizer,
                     train_loader,
                     learning_routine=LearningRoutine()):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param train_loader: The loader to use for training
    :param learning_routine: A LearningRoutine object, if we want to also use a validation phase and early stopping
    :return:
    """
    device = model.current_device

    start_time = time.time()
    for epoch in range(learning_routine.num_epochs):
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

            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            loss = loss.item()
            running_loss += loss

            if batch_idx % learning_routine.print_each == 0:
                time_elapsed = time.time() - start_time
                print(
                    f'Train Epoch: {epoch + 1} [{(batch_idx + 1)}/{num_batches} '
                    f'({100. * (batch_idx + 1) / num_batches:.0f}%)]\t'
                    f'Loss: {loss:.6f}  Time: {time_elapsed:.2f}')

                # tensorboard logging
                if learning_routine.writer is not None:
                    step = epoch * num_batches + batch_idx
                    learning_routine.writer.add_scalar("Training loss", loss, step)

        train_loss = running_loss / num_batches
        if learning_routine.writer is not None:
            learning_routine.writer.add_scalar("Training epoch loss", train_loss, epoch)

        # Test phase, if we do not have a validation, just iterate.
        # Otherwise call the routines.
        if learning_routine.validation_loader is None:
            early_stop = learning_routine.early_stopping_routine(validation_loss=train_loss, epoch=epoch,
                                                                 model=model, optimizer=optimizer, validation=False)
        else:
            validation_loss = evaluate_model_supervised(model, validation_loader=learning_routine.validation_loader)
            early_stop = learning_routine.early_stopping_routine(validation_loss=validation_loss, epoch=epoch,
                                                                 model=model, optimizer=optimizer)
        if early_stop:
            break
    return learning_routine.best_loss


def train_linkpred(model,
                   optimizer,
                   train_loader_generator,
                   validation_loader_generator
                   ):
    """

    :param model:
    :param optimizer:
    :param train_loader_generator:
    :param validation_loader_generator:
    :return:
    """
    for epoch in range(3):
        count = 0
        time_start = time.time()
        train_loader = train_loader_generator.get_edge_loader()
        for g in train_loader:
            for step, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(g):
                pos_score = model(positive_graph)
                neg_score = model(positive_graph, negative_graph=negative_graph)

                score = torch.cat([pos_score, neg_score])
                label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                loss = F.binary_cross_entropy_with_logits(score, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                count += len(input_nodes)
                # if True or not count % 50:
                #     print(count, loss.item(), time.time() - time_start)
        print(f"EPOCH  {epoch}, time for the epoch :  {time.time() - time_start:2f}, last loss {loss.item():2f}")

    aucs = []
    count = 0
    model.eval()
    validation_loader = validation_loader_generator.get_edge_loader()
    for i, g in enumerate(validation_loader):
        print("val graph ", i)
        for input_nodes, positive_graph, negative_graph, blocks in g:
            with torch.no_grad():
                pos_score = model(positive_graph)
                neg_score = model(positive_graph, negative_graph=negative_graph)

                score = torch.cat([pos_score, neg_score]).detach().numpy()
                label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                label = label.detach().numpy()
                # print(score, label)
                aucs.append(roc_auc_score(label, score))
                count += 1
    print('Time used : ', time.time() - time_start)
    print("AUC", np.mean(aucs))
    pass


if __name__ == '__main__':
    pass
    from rnaglib.learning import models
    from rnaglib.data_loading import loader

    test_unsupervised = False
    test_supervised = True
    if test_unsupervised:
        from rnaglib.kernels import node_sim

        embedder_model = models.Embedder([10, 10])
        optimizer = torch.optim.Adam(embedder_model.parameters())

        node_sim_func = node_sim.SimFunctionNode(method='R_1', depth=2)
        data_path = os.path.join(script_dir, '..', 'data/annotated/NR_annot/')
        node_features = ['nt_code']
        unsupervised_dataset = loader.UnsupervisedDataset(node_simfunc=node_sim_func,
                                                          node_features=node_features,
                                                          data_path=data_path)
        train_loader = loader.Loader(dataset=unsupervised_dataset, split=False,
                                     num_workers=0, max_size_kernel=100).get_data()

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
        classifier_model = models.Classifier(embedder=embedder_model, classif_dims=[1])
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
        train_supervised(model=classifier_model,
                         optimizer=optimizer,
                         train_loader=train_loader)
