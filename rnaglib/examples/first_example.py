#!/usr/bin/env python3

import os
import sys
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.learning import models, learn
from rnaglib.data_loading import graphloader

"""
This script just shows a first very basic example : learn binding protein preferences 
from the nucleotide types and the graph structure

To do so, we choose our data, create a data loader around it, build a RGCN model and train it.
"""

if __name__ == "__main__":
    # Choose the data, features and targets to use and GET THE DATA GOING
    node_features = ['nt_code']
    node_target = ['binding_protein']
    supervised_dataset = graphloader.SupervisedDataset(node_features=node_features,
                                                       node_target=node_target)
    train_loader, validation_loader, test_loader = graphloader.GraphLoader(dataset=supervised_dataset).get_data()

    # Define a model, we first embed our data in 10 dimensions, and then add one classification
    input_dim, target_dim = supervised_dataset.input_dim, supervised_dataset.output_dim
    embedder_model = models.Embedder(dims=[10, 10], infeatures_dim=input_dim)
    classifier_model = models.Classifier(embedder=embedder_model, classif_dims=[target_dim])

    # Finally get the training going
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
    learn.train_supervised(model=classifier_model,
                           optimizer=optimizer,
                           train_loader=train_loader)
