import os
import sys
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

script_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from data_loading.loader import *

# 2 Layer model
class GCN(nn.Module):
    def __init__(self,
                in_dim,
                h_dim,
                out_dim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_dim, h_dim)
        self.conv2 = GraphConv(h_dim, out_dim)

    def forward(self, g, in_dim):
        h = self.conv1(g, in_dim)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(train_loader, model, optim,
            num_epochs=100):


    for e in range(num_epochs):
        for g in train_loader:

            # TODO: combine node features into a single feat tensor
            features = g.ndata['feat']
            # TODO: decide on labels for ion/small molecule task
            labels = g.ndata['label']
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            loss = F.cross_entropy(logits, labels)

            # Compute accuracy
            acc = (pred == labels).float().mean()

            # Backward
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Print
        print('epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(e,
            loss,
            val_acc,
            best_val_acc,
            test_acc,
            best_test_acc))

def main():

    # Get data
    graph_dir = os.path.join(script_dir, '../data/graphs')
    loader = Loader(annotated_path=graph_dir)
    train_loader, valid_loader, test_loader = loader.get_data()

    # Define model
    model = GCN(32, 32, 1)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train(train_loader, model, optim=optim, num_epochs=100)

    # Test model
    test(test_loader, model, optim=optim)

if __name__ == '__main__':
    main()
