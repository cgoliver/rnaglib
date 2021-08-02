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
                 infeatures_dim=-1,
                 num_rels=19,
                 num_bases=-1,
                 conv_output=True,
                 self_loop=True,
                 verbose=False):
        super(Embedder, self).__init__()
        self.dims = dims
        self.use_node_features = (infeatures_dim != -1)
        self.in_dim = 1 if infeatures_dim == -1 else infeatures_dim
        self.conv_output = conv_output
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
        if self.verbose:
            print("short, ", short)
            print("last_hidden, last ", last_hidden, last)

        # input feature is just node degree
        i2h = RelGraphConv(in_feat=self.in_dim,
                           out_feat=self.dims[0],
                           num_rels=self.num_rels,
                           num_bases=self.num_bases,
                           activation=F.relu,
                           self_loop=self.self_loop)
        layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            h2h = RelGraphConv(in_feat=dim_in,
                               out_feat=dim_out,
                               num_rels=self.num_rels,
                               num_bases=self.num_bases,
                               activation=F.relu,
                               self_loop=self.self_loop)
            layers.append(h2h)

        # hidden to output
        if self.conv_output:
            h2o = RelGraphConv(in_feat=last_hidden,
                               out_feat=last,
                               num_rels=self.num_rels,
                               num_bases=self.num_bases,
                               self_loop=self.self_loop,
                               activation=None)
        else:
            h2o = nn.Linear(last_hidden, last)
        layers.append(h2o)
        return layers

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g):
        if self.use_node_features:
            h = g.ndata['features']
        else:
            # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            # layer(g)
            if not self.conv_output and (i == len(self.layers) - 1):
                h = layer(h)
            else:
                h = layer(g, h, g.edata['edge_type'])
        g.ndata['h'] = h
        return g.ndata['h']


class Classifier(nn.Module):

    def __init__(self,
                 embedder,
                 last_dim_embedder,
                 classif_dims=None,
                 num_rels=19,
                 num_bases=-1,
                 conv_output=True,
                 self_loop=True,
                 verbose=False):
        super(Classifier, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.conv_output = conv_output

        self.embedder = embedder
        self.last_dim_embedder = last_dim_embedder
        self.classif_dims = classif_dims

        self.classif_layers = self.build_model()

        self.verbose = verbose
        if self.verbose:
            print(self.layers)
            print("Num rels: ", self.num_rels)

    def build_model(self):
        if self.classif_dims is None:
            return []

        classif_layers = nn.ModuleList()
        # Just one convolution
        if len(self.classif_dims) == 1:
            if self.conv_output:
                h2o = RelGraphConv(in_feat=self.last_dim_embedder,
                                   out_feat=self.classif_dims[0],
                                   num_rels=self.num_rels,
                                   num_bases=self.num_bases,
                                   # TODO : a PR was sent to DGL, self loops crash
                                   #  with the current implementations if outfeat==1
                                   # self_loop=self.self_loop,
                                   self_loop=self.self_loop and self.classif_dims[0] > 1,
                                   activation=None)
            else:
                h2o = nn.Linear(self.last_dim_embedder, self.classif_dims[0])
            classif_layers.append(h2o)
            return classif_layers

        # The supervised is more than one layer
        else:
            i2h = RelGraphConv(in_feat=self.last_dim_embedder,
                               out_feat=self.classif_dims[0],
                               num_rels=self.num_rels,
                               num_bases=self.num_bases,
                               activation=F.relu,
                               self_loop=self.self_loop)
            classif_layers.append(i2h)
            last_hidden, last = self.classif_dims[-2:]
            short = self.classif_dims[:-1]
            for dim_in, dim_out in zip(short, short[1:]):
                h2h = RelGraphConv(in_feat=dim_in,
                                   out_feat=dim_out,
                                   num_rels=self.num_rels,
                                   num_bases=self.num_bases,
                                   activation=F.relu,
                                   self_loop=self.self_loop)
                classif_layers.append(h2h)

            # hidden to output
            if self.conv_output:
                h2o = RelGraphConv(in_feat=last_hidden,
                                   out_feat=last,
                                   num_rels=self.num_rels,
                                   num_bases=self.num_bases,
                                   # TODO : a PR was sent to DGL, self loops crash
                                   #  with the current implementations if outfeat==1
                                   # self_loop=self.self_loop,
                                   self_loop=self.self_loop and last > 1,
                                   activation=None)
            else:
                h2o = nn.Linear(last_hidden, last)
            classif_layers.append(h2o)
            return classif_layers

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def forward(self, g):
        h = self.embedder(g)
        for i, layer in enumerate(self.classif_layers):
            # layer(g)
            if not self.conv_output and (i == len(self.classif_layers) - 1):
                h = layer(h)
            # Convolution layer
            else:
                h = layer(g, h, g.edata['edge_type'])

        # This is necessary due to the dgl bug
        if len(h.shape) == 1:
            h = h[..., None]
        g.ndata['h'] = h
        return g.ndata['h']
