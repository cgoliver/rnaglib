import os
import sys

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from learning import models, learn
from data_loading import loader

"""
This script just shows a first very basic example : learn binding protein preferences 
from the nucleotide types and the graph structure

To do so, we choose our data, create a data loader around it, build a RGCN model and train it.
"""


# Choose the data, features and targets to use
data_path = os.path.join(script_dir, "../data/graphs")
node_features = ['nt_code']
node_target = ['binding_protein']
infeatures_dim = len(node_features)
target_dim = len(node_target)

# GET THE DATA GOING
loader = loader.SupervisedLoader(data_path=data_path,
                                 node_features=node_features,
                                 node_target=node_target)
train_loader, validation_loader, test_loader = loader.get_data()

# Define a model and train it, we first embed our data in 10 dimensions, and then add one classification
embedder_model = models.Embedder(dims=[10, 10], infeatures_dim=infeatures_dim)
classifier_model = models.Classifier(embedder=embedder_model, last_dim_embedder=10, classif_dims=[target_dim])

# Finally get the training going
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
learn.train_supervised(model=classifier_model,
                       optimizer=optimizer,
                       train_loader=train_loader)
