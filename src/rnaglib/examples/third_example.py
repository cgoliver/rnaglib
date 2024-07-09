#!/usr/bin/env python3

import os
import sys
import torch
import random

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.learning import models, learn
from rnaglib.data_loading import rna_loader
from rnaglib.benchmark import evaluate

"""
This is a very basic example of link prediction applied to RNA base pairs.
We use our Embedder object along with the nucleotide ID as features.
This is passed to an edge loader and a base pair predictor model.
"""

def main():
    # Get loader for link prediction,
    # use nucleotide identity as input features and base our fixed train/test split
    # on the binding protein one for reproducibility
    node_features = ['nt_code']
    node_target = ['binding_protein']
    train_split, test_split = evaluate.get_task_split(node_target=node_target)

    train_dataset = graphloader.GraphDataset(node_features=['nt_code'], all_graphs=train_split)
    test_dataset = graphloader.GraphDataset(node_features=['nt_code'], all_graphs=test_split)
    train_loader = graphloader.EdgeLoaderGenerator(graphloader.get_loader(train_dataset, split=False))
    test_loader = graphloader.EdgeLoaderGenerator(graphloader.get_loader(test_dataset, split=False))

    # Choose the data, features and targets to use and GET THE DATA GOING
    embedder_model = models.Embedder(dims=[10, 10], infeatures_dim=train_dataset.input_dim)
    linkpred_model = models.BasePairPredictor(embedder_model)

    # Finally get the training going
    optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=0.001)
    learn.train_linkpred(linkpred_model, optimizer, train_loader, test_loader)
