import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import RelGraphConv


class Embedder(nn.Module):

    def __init__(self,
                 dims,
                 initial_node_dims=None,
                 num_rels=19,
                 num_bases=-1,
                 conv_output=False,
                 self_loop=False,
                 verbose=False):
        """

        :param dims: an iterable that lists the successive dimensions of the node embeddings
        :param num_rels:
        :param num_bases:
        :param conv_output:
        :param self_loop:
        :param verbose:
        """
        super(Embedder, self).__init__()
        self.initial_node_dims = initial_node_dims
        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.verbose = verbose

        self.layers = self.build_model()
        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = self.dims[-2:]

        # input feature is just node degree or constant ones
        if self.initial_node_dims is None:
            self.initial_node_dims = 1
        i2h = RelGraphConv(self.initial_node_dims, self.dims[0],
                           num_rels=self.num_rels,
                           num_bases=self.num_bases,
                           activation=F.relu,
                           self_loop=self.self_loop)
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = RelGraphConv(dim_in, dim_out,
                               num_rels=self.num_rels,
                               num_bases=self.num_bases,
                               activation=F.relu,
                               self_loop=self.self_loop)
            layers.append(h2h)
        # hidden to output
        h2o = RelGraphConv(last_hidden, last,
                           num_rels=self.num_rels,
                           num_bases=self.num_bases,
                           activation=None,
                           self_loop=self.self_loop)
        layers.append(h2o)
        return layers

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g, node_features=None):
        if node_features is None:
            # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        else:
            h = node_features
        for i, layer in enumerate(self.layers):
            h = layer(g, h, g.edata['one_hot'])
        g.ndata['h'] = h
        return g.ndata['h']
