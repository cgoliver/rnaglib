import os
import sys

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))
    
from rnaglib.learning import models, learn
from rnaglib.data_loading import loader
from rnaglib.benchmark import evaluate
from rnaglib.kernels import node_sim

"""
This script shows a second more complicated example : learn binding protein preferences as well as
small molecules binding from the nucleotide types and the graph structure
We also add a pretraining phase based on the R_graphlets kernel
"""

# Choose the data, features and targets to use
node_features = ['nt_code']
node_target = ['binding_protein']

###### Unsupervised phase : ######
# Choose the data and kernel to use for pretraining
print('Starting to pretrain the network')
node_sim_func = node_sim.SimFunctionNode(method='R_graphlets', depth=2)
unsupervised_dataset = loader.UnsupervisedDataset(node_simfunc=node_sim_func,
                                                  node_features=node_features)
train_loader = loader.Loader(dataset=unsupervised_dataset, split=False,
                             num_workers=0, max_size_kernel=100).get_data()

# Then choose the embedder model and pre_train it, we dump a version of this pretrained model
embedder_model = models.Embedder(infeatures_dim=unsupervised_dataset.input_dim,
                                 dims=[64, 64])
optimizer = torch.optim.Adam(embedder_model.parameters())
learn.pretrain_unsupervised(model=embedder_model,
                            optimizer=optimizer,
                            train_loader=train_loader,
                            learning_routine=learn.LearningRoutine(num_epochs=10),
                            rec_params={"similarity": True, "normalize": False, "use_graph": True, "hops": 2})
# torch.save(embedder_model.state_dict(), 'pretrained_model.pth')
print()

###### Now the supervised phase : ######
print('We have finished pretraining the network, let us fine tune it')
# GET THE DATA GOING, we want to use precise data splits to be able to use the benchmark.
train_split, test_split = evaluate.get_task_split(node_target=node_target)
supervised_train_dataset = loader.SupervisedDataset(node_features=node_features,
                                                    redundancy='all_graphs',
                                                    node_target=node_target,
                                                    all_graphs=train_split)
train_loader = loader.Loader(dataset=supervised_train_dataset, split=False).get_data()

# Define a model and train it :
# We first embed our data in 64 dimensions, using the pretrained embedder and then add one classification
# Then get the training going
classifier_model = models.Classifier(embedder=embedder_model, classif_dims=[supervised_train_dataset.output_dim])
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=0.001)
learn.train_supervised(model=classifier_model,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       learning_routine=learn.LearningRoutine(num_epochs=10))

# torch.save(classifier_model.state_dict(), 'final_model.pth')
# embedder_model = models.Embedder(infeatures_dim=4, dims=[64, 64])
# classifier_model = models.Classifier(embedder=embedder_model, classif_dims=[1])
# classifier_model.load_state_dict(torch.load('final_model.pth'))

# Get a benchmark performance on the official uncontaminated test set :
metric = evaluate.get_performance(node_target=node_target, node_features=node_features, model=classifier_model)
print('We get a performance of :', metric)
print()
