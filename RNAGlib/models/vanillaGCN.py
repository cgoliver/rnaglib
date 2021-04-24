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
from kernels.node_sim import SimFunctionNode

# 2 Layer model
class GCN(nn.Module):
    def __init__(self,
                in_dim,
                h_dim,
                out_dim,
                self_loop=True):
        super(GCN, self).__init__()
        self.self_loop = self_loop
        self.conv1 = GraphConv(in_dim, h_dim,
                allow_zero_in_degree=(not self.self_loop))
        self.conv2 = GraphConv(h_dim, out_dim,
                allow_zero_in_degree=(not self.self_loop) )

    def forward(self, g, in_dim):
        if self.self_loop: g = dgl.add_self_loop(g)
        h = self.conv1(g, in_dim)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = torch.sigmoid(h)
        return h

def train(loader, model, optim,
            num_epochs=100,
            threshold=0.5):
    best_val_acc = 0.0
    best_test_acc = 0.0
    train_loader, valid_loader, test_loader = loader.get_data()
    # train_loader = loader.get_data()

    for e in range(num_epochs):
        epoch_loss = 0.0
        for g, K, g_len in train_loader:

            features = g.ndata['features']
            labels = g.ndata['target']
            logits = model(g, features).squeeze()

            # Compute loss
            loss = F.binary_cross_entropy(logits, labels)

            # Compute accuracy
            preds = (logits > threshold).float()
            acc = (preds*labels).float().mean()

            # Backward
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Record loss
            epoch_loss += loss

        # Test
        val_acc = test(valid_loader, model, threshold=threshold)
        test_acc = test(test_loader, model, threshold=threshold)

        # Print
        print('epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(e,
            epoch_loss,
            val_acc,
            best_val_acc,
            test_acc,
            best_test_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

def test(loader, model, threshold=0.5):

    for g, K, graph_len in loader:
        features = g.ndata['features']
        logits = model(g, features)

        # Compute accuracy
        preds = (logits > threshold).float()
        labels = g.ndata['target']
        acc = (preds*labels).float().mean()

        return acc

def main():

    # Get data
    annotated_path = os.path.join(script_dir, '../../data/annotated/undirected')
    simfunc_r1 = SimFunctionNode('R_1', 2)
    loader = Loader(annotated_path=annotated_path,
                    num_workers=0,
                    split=True,
                    directed=False,
                    node_simfunc=simfunc_r1,
                    node_features='all',
                    node_target='binding_ion')

    # Define model
    model = GCN(46, 32, 1)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    train(loader, model, optim=optim, num_epochs=100)

if __name__ == '__main__':
    main()
