import os
import sys

import time

import numpy as np
import torch
import dgl

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))


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


def test(model, test_loader):
    model.eval()
    device = model.current_device
    test_size = len(test_loader)
    recons_loss_tot = 0
    for batch_idx, (graph, K, inds, graph_sizes) in enumerate(test_loader):
        # Get data on the devices
        K = K.to(device)
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            out = model(graph)
            reconstruction_loss = model.rec_loss(embeddings=out,
                                                 target_K=K,
                                                 graph=graph)
            recons_loss_tot += reconstruction_loss
    return recons_loss_tot / test_size


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
    assert node_ids is None or not use_graph
    if node_ids is not None:
        node_ids = np.argwhere(np.array(node_ids) > 0).squeeze()
        K_predict_1 = K_predict[node_ids]
        K_predict = K_predict_1[:, node_ids]

    if use_graph:
        assert graph is not None
        import networkx as nx
        nx_graph = graph.to_networkx(edge_attrs=['one_hot'])
        nx_graph = nx.to_undirected(nx_graph)
        ordered = sorted(nx_graph.nodes())
        adj_matrix_full = nx.to_scipy_sparse_matrix(nx_graph, nodelist=ordered)

        # copy the matrix with only the non canonical
        extracted_edges = [(u, v) for u, v, e in nx_graph.edges.data('one_hot', default='0')
                           if e not in [0, 6]]
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
        return weighted_MSE(K_predict, target_K, weight)

    reconstruction_loss = torch.nn.MSELoss()(K_predict, target_K)
    return reconstruction_loss


def pretrain_unsupervised(model,
                          node_sim,
                          train_loader,
                          optimizer,
                          num_epochs=25,
                          early_stopper=None,
                          writer=None,
                          rec_params={"similarity": True, "normalize": False, "use_graph": False, "hops": 2}
                          ):
    device = model.current_device
    train_loader.dataset.dataset.add_node_sim(node_simfunc=node_sim)
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
            batch_size = len(K)

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

            if batch_idx % 20 == 0:
                time_elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Time: {:.2f}'.format(
                    epoch + 1,
                    (batch_idx + 1),
                    num_batches,
                    100. * (batch_idx + 1) / num_batches,
                    loss,
                    time_elapsed))

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
            test_loss = test(model, test_loader=early_stopper.test_loader)

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


def train_model(model,
                optimizer,
                train_loader,
                test_loader,
                save_path,
                writer=None, num_epochs=25):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param train_loader: data loader for training
    :param test_loader: data loader for validation
    :param save_path: where to save the model
    :param writer: a pytorch writer object
    :param num_epochs: int number of epochs
    :param wall_time: The number of hours you want the model to run
    :param embed_only: number of epochs before starting attributor training.
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

        for batch_idx, (graph, K, graph_sizes) in enumerate(train_loader):
            # Get data on the devices
            K = K.to(device)
            graph = send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            out = model(graph)
            loss = model.rec_loss(embeddings=out,
                                  target_K=K,
                                  graph=graph)
            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            loss = loss.item()
            running_loss += loss

            if batch_idx % 20 == 0:
                time_elapsed = time.time() - start_time
                print(
                    f'Train Epoch: {epoch + 1} [{(batch_idx + 1)}/{num_batches} '
                    f'({100. * (batch_idx + 1) / num_batches:.0f}%)]\t'
                    f'Loss: {loss:.6f}  Time: {time_elapsed:.2f}')

                # tensorboard logging
                step = epoch * num_batches + batch_idx
                writer.add_scalar("Training loss", loss, step)

        # # Log training metrics
        train_loss = running_loss / num_batches
        writer.add_scalar("Training epoch loss", train_loss, epoch)

        # Test phase
        test_loss = test(model, test_loader)

        writer.add_scalar("Test loss during training", test_loss, epoch)
        #
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
            }, save_path)
            model.to(device)

        # Early stopping
        else:
            epochs_from_best += 1
            if epochs_from_best > early_stop_threshold:
                print('This model was early stopped')
                break
    return best_loss


if __name__ == '__main__':
    from learning import models
    from kernels import node_sim
    from data_loading import loader

    embedder_model = models.Embedder([10, 10])
    optimizer = torch.optim.Adam(embedder_model.parameters())
    node_sim_func = node_sim.SimFunctionNode(method='R_1', depth=2)

    data_path = os.path.join(script_dir, '../data/annotated/samples/')
    data_loader = loader.Loader(annotated_path=data_path,
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
