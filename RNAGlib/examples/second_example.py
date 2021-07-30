import os
import sys

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from learning import models, learn
from data_loading import loader
from kernels import node_sim

"""
This script shows a second more complicated example : learn binding protein preferences as well as
small molecules binding from the nucleotide types and the graph structure
We also add a pretraining phase based on the R_1 kernel
"""

# Choose the data, features and targets to use
node_features = ['nt_code']
node_target = ['binding_protein']

###### Unsupervised phase : ######
# Choose the data and kernel to use for pretraining
print('Starting to pretrain the network')
node_sim_func = node_sim.SimFunctionNode(method='R_1', depth=2)
unsupervised_dataset = loader.UnsupervisedDataset(node_simfunc=node_sim_func, node_features=node_features)
# print(dataset[0][0].to_networkx().edges(data=True))
data_loader = loader.Loader(dataset=unsupervised_dataset, max_size_kernel=100)
train_loader, _, _ = data_loader.get_data()

# Then choose the embedder model and pre_train it
embedder_model = models.Embedder(infeatures_dim=unsupervised_dataset.input_dim,
                                 dims=[10, 10])
optimizer = torch.optim.Adam(embedder_model.parameters())
learn.pretrain_unsupervised(model=embedder_model,
                            optimizer=optimizer,
                            node_sim=node_sim_func,
                            train_loader=train_loader,
                            rec_params={"similarity": True, "normalize": False, "use_graph": True, "hops": 2},
                            num_epochs=2)
print()
print('We have finished pretraining the network, let us fine tune it')

###### Now the supervised phase : ######
# GET THE DATA GOING
supervised_dataset = loader.SupervisedDataset(node_features=node_features,
                                              node_target=node_target)
train_loader, validation_loader, test_loader = loader.Loader(dataset=supervised_dataset).get_data()

# Define a model and train it :
# We first embed our data in 10 dimensions, using the pretrained embedder and then add one classification
classifier_model = models.Classifier(embedder=embedder_model, last_dim_embedder=10,
                                     classif_dims=[supervised_dataset.output_dim])

# Finally get the training going
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
learn.train_supervised(model=classifier_model,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       num_epochs=2)
