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
from rnaglib.learning import learning_utils


def pretrain_unsupervised(model,
                          train_loader,
                          optimizer,
                          node_sim=None,
                          learning_routine=learning_utils.LearningRoutine(),
                          rec_params={"similarity": True, "normalize": False, "use_graph": False, "hops": 2}
                          ):
    """
    Perform the pretraining routine to get embeddings from graph nodes, that correlate with a node kernel.

    :param model: The model to train
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param train_loader: The loader to use for training, as defined in GraphLoader
    :param node_sim: If None, we just rely on the node_sim in the data loader.
    :param learning_routine: A LearningRoutine object, if we want to also use a validation phase and early stopping
    :param rec_params: These are parameters useful for the loss computation and
    further explained in learning_utils.rec_loss

    :return: The best loss obtained
    """
    device = model.current_device
    learning_routine.device = device

    start_time = time.time()
    for epoch in range(learning_routine.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            graph, (K, node_ids) = batch['graph'], batch['ring']
            # Get data on the devices
            K = K.to(device)
            graph = learning_utils.send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            graph, out = model(graph)
            loss = learning_utils.rec_loss(embeddings=out,
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

        # Validation phase, we always use early stopping, optionnaly on the training values
        if learning_routine.validation_loader is None:
            early_stop = learning_routine.early_stopping_routine(validation_loss=train_loss, epoch=epoch,
                                                                 model=model, optimizer=optimizer)

        else:
            validation_loss = learning_utils.evaluate_model_unsupervised(model,
                                                                         validation_loader=learning_routine.validation_loader,
                                                                         rec_params=rec_params)
            if learning_routine.writer is not None:
                learning_routine.writer.add_scalar("Validation loss during training", validation_loss, epoch)
            early_stop = learning_routine.early_stopping_routine(validation_loss=validation_loss, epoch=epoch,
                                                                 model=model, optimizer=optimizer)
        if early_stop:
            break
    return learning_routine.best_loss


def train_supervised(model,
                     optimizer,
                     train_loader,
                     learning_routine=learning_utils.LearningRoutine()):
    """
    Performs the entire training routine for a supervised task

    :param model: The model to train
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param train_loader: The loader to use for training, as defined in data_loading/GraphLoader
    :param learning_routine: A LearningRoutine object, if we want to also use a validation phase and early stopping

    :return: The best loss obtained
    """
    device = model.current_device

    start_time = time.time()
    for epoch in range(learning_routine.num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # Get data on the devices
            graph = batch['graph']
            graph = learning_utils.send_graph_to_device(graph, device)

            # Do the computations for the forward pass
            out = model(graph)
            labels = graph.ndata['nt_targets']
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
                                                                 model=model, optimizer=optimizer)
        else:
            validation_loss = learning_utils.evaluate_model_supervised(model,
                                                                       loader=learning_routine.validation_loader)
            if learning_routine.writer is not None:
                learning_routine.writer.add_scalar("Validation loss during training", validation_loss, epoch)
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
    Train a link prediction model : given RNA graphs, predict whether nodes are bound

    :param model: The model to train
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param train_loader_generator: The edge loader to use for training, as defined in data_loading/GraphLoader
    :param validation_loader_generator: The edge loader to use for training, as defined in data_loading/GraphLoader

    :return: The best loss obtained
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
    from rnaglib.data_loading import rna_loader

    test_unsupervised = False
    test_supervised = True
    if test_unsupervised:
        from rnaglib.kernels import node_sim

        embedder_model = models.Embedder([10, 10])
        optimizer = torch.optim.Adam(embedder_model.parameters())

        node_sim_func = node_sim.SimFunctionNode(method='R_1', depth=2)
        data_path = os.path.join(script_dir, '..', 'data/annotated/NR_annot/')
        node_features = ['nt_code']
        unsupervised_dataset = graphloader.GraphDataset(node_simfunc=node_sim_func,
                                                        node_features=node_features,
                                                        data_path=data_path,
                                                        chop=True)
        train_loader = graphloader.get_loader(dataset=unsupervised_dataset, split=False,
                                              num_workers=0, max_size_kernel=100)

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
        supervised_dataset = graphloader.GraphDataset(data_path=annotated_path,
                                                      node_features=node_features,
                                                      node_target=node_target)
        train_loader, validation_loader, test_loader = graphloader.get_loader(dataset=supervised_dataset, split=True,
                                                                              num_workers=0)

        embedder_model = models.Embedder([10, 10], infeatures_dim=1)
        classifier_model = models.Classifier(embedder=embedder_model, classif_dims=[1])
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
        train_supervised(model=classifier_model,
                         optimizer=optimizer,
                         train_loader=train_loader)
