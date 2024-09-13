#!/usr/bin/env python3

import torch

from rnaglib.learning import models, learn
from rnaglib.data_loading import rna_dataset, rna_loader
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import GraphRepresentation

"""
This script just shows a first very basic example : learn binding protein preferences 
from the nucleotide types and the graph structure

To do so, we choose our data, create a data loader around it, build a RGCN model and train it.
"""


def main():
    # Choose the Features and Representation to use, and get the data going !
    features_computer = FeaturesComputer(
        nt_features="nt_code", nt_targets="binding_protein"
    )
    graph_rep = GraphRepresentation(framework="dgl")
    all_rnas = [
        "1a9n.json",
        "1b23.json",
        "1b7f.json",
        "1csl.json",
        "1d4r.json",
        "1dfu.json",
        "1duq.json",
        "1e8o.json",
        "1ec6.json",
        "1et4.json",
    ]
    supervised_dataset = rna_dataset.RNADataset(
        all_rnas=all_rnas,
        features_computer=features_computer,
        representations=[graph_rep],
    )
    train_loader, validation_loader, test_loader = rna_loader.get_loader(
        dataset=supervised_dataset
    )

    # Define a model, we first embed our data in 10 dimensions, and then add one classification
    input_dim, target_dim = features_computer.input_dim, features_computer.output_dim
    embedder_model = models.Embedder(dims=[10, 10], infeatures_dim=input_dim)
    classifier_model = models.Classifier(
        embedder=embedder_model, classif_dims=[target_dim]
    )

    # Finally get the training going
    optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
    learn.train_supervised(
        model=classifier_model, optimizer=optimizer, train_loader=train_loader
    )


main()
