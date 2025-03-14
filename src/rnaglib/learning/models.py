import torch
import torch.nn as nn
import torch.nn.functional as F



class Embedder(nn.Module):
    def __init__(self,
                 dims,
                 infeatures_dim=0,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 verbose=False):
        """
        This is an exemple RGCN for unsupervised learning, going from one element of "dims" to the other

        It maps the "features" of an input graph to an "h" node attribute and returns the corresponding tensor.

        :param dims: The succesive dimensions of the embeddings, should be an iterable or an int
        :param infeatures_dim: The dimension of the input features
        :param num_rels: The number of relations that are to be found in the graphs. Defaults to the 20 base pair types
        :param num_bases: This is to use the basis sharing trick used in RGCN in general
        :param conv_output: Whether to use a convolution at the end of the embedding or simply a linear layer
        :param self_loop: Whether each node is also connected to itself
        :param verbose: blah
        """
        super(Embedder, self).__init__()
        self.dims = dims
        self.use_node_features = (infeatures_dim != 0)
        self.in_dim = 1 if infeatures_dim == 0 else infeatures_dim
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
            h = g.ndata['nt_features']
        else:
            # h = g.in_degrees().view(-1, 1).float().to(self.current_device)
            h = torch.ones(len(g.nodes())).view(-1, 1).to(self.current_device)
        for i, layer in enumerate(self.layers):
            if not self.conv_output and (i == len(self.layers) - 1):
                h = layer(h)
            else:
                h = layer(g=g, feat=h, etypes=g.edata['edge_type'])
        g.ndata['h'] = h
        return g.ndata['h']


class Classifier(nn.Module):
    def __init__(self,
                 embedder,
                 classif_dims=None,
                 num_rels=20,
                 num_bases=None,
                 conv_output=True,
                 self_loop=True,
                 verbose=False):
        """
            This is an exemple RGCN for supervised learning, that uses the previous Embedder network

        :param embedder: An embedder network as defined above
        :param classif_dims: An iterable of the successive embedding dimensions, similarly to the dims of the Embedder
        :param num_rels: The number of relations that are to be found in the graphs. Defaults to the 20 base pair types
        :param num_bases: This is to use the basis sharing trick used in RGCN in general
        :param conv_output: Whether to use a convolution at the end of the embedding or simply a linear layer
        :param self_loop: Whether each node is also connected to itself
        :param verbose: blah
        """
        super(Classifier, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.self_loop = self_loop
        self.conv_output = conv_output

        self.embedder = embedder
        self.last_dim_embedder = embedder.dims[-1]
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
                                   self_loop=self.self_loop,
                                   # Old fix for a bug in dgl<0.6
                                   # self_loop=self.self_loop and self.classif_dims[0] > 1,
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
                                   self_loop=self.self_loop,
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
            # if this is the last layer and we want to use a linear layer, the call is different
            if (i == len(self.classif_layers) - 1) and not self.conv_output:
                h = layer(h)
            # Convolution layer
            else:
                h = layer(g, h, g.edata['edge_type'])
        g.ndata['h'] = h
        return g.ndata['h']


class DotPredictor(nn.Module):
    def __init__(self):
        """
        Given node embeddings and a connectivity, predict a dot product score for each edge
        """
        super(DotPredictor, self).__init__()
        self.norm = torch.nn.Sigmoid()

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            g.edata['score'] = self.norm(g.edata['score'])
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class BasePairPredictor(nn.Module):

    def __init__(self, encoder, decoder=DotPredictor()):
        """This is an exemple RGCN for link prediction, that uses the previous Embedder network
        Predict the probability that two nucleotides are base paired, based on the dot product of the node embeddings

        :param encoder: An Embedder network as defined above
        :param decoder: A tool to compute the dot products of a given connectivity.
        """
        super(BasePairPredictor, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, g, negative_graph=None):
        """
        Predicts the probability that each edge exists.
            If negative graph is not None, we embed the real graph and then predict the negative graph connectivity

        :param g: The real graph to compute node embeddings and edge likelihood over
        :param negative_graph: A decoy connectivity to compute edge likelihood over
        :return: The score for the edge likelihood
        """
        with g.local_scope():
            h = self.encoder(g)
            if negative_graph is not None:
                return self.decoder(negative_graph, h)
            return self.decoder(g, h)
