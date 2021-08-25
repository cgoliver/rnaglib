import os
import sys
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.learning import models, learn
from rnaglib.data_loading import loader

"""
This is a very basic example of link prediction applied to RNA base pairs.
We use our Embedder object along with the nucleotide ID as features.
This is passed to an edge loader and a base pair predictor model.
"""

# Get loader for link prediction

# use nucleotide identity as input features
dataset = loader.GraphDataset(node_features=['nt_code'])
train_loader, val_loader, test_loader = loader.BasePairLoader().get_data()


# Choose the data, features and targets to use and GET THE DATA GOING
embedder_model = models.Embedder(dims=[10, 10], infeatures_dim=dataset.input_dim)
linkpred_model = models.BasePairPredictor(embedder_model)

# Finally get the training going
optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=0.001)
learn.train_linkpred(linkpred_model, optimizer, train_loader, val_loader)

# final
