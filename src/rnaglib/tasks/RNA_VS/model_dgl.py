"""
Script for RGCN model.
This code is adapted from https://github.com/cgoliver/rnamigos2/blob/master/rnamigos_dock/learning/models.py
"""

import os

import dgl
from dgl.nn.pytorch.glob import SumPooling, GlobalAttentionPooling
from dgl.nn.pytorch.conv import RelGraphConv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCN(nn.Module):
    """ RGCN encoder with num_hidden_layers + 2 RGCN layers, and sum pooling. """

    def __init__(self, features_dim, h_dim, num_rels, num_layers, num_bases=-1, gcn_dropout=0, batch_norm=False,
                 self_loop=False):
        super(RGCN, self).__init__()

        self.features_dim, self.h_dim = features_dim, h_dim
        self.num_layers = num_layers
        self.p = gcn_dropout

        self.self_loop = self_loop
        self.num_rels = num_rels
        self.num_bases = num_bases
        # create rgcn layers
        self.build_model()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(h_dim) for _ in range(num_layers)])
        self.pool = SumPooling()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = RelGraphConv(self.features_dim, self.h_dim, self.num_rels, self_loop=self.self_loop,
                           activation=nn.ReLU(), dropout=self.p)
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_layers - 2):
            h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels,
                               self_loop=self.self_loop, activation=nn.ReLU(), dropout=self.p)
            self.layers.append(h2h)
        # hidden to output
        h2o = RelGraphConv(self.h_dim, self.h_dim, self.num_rels,
                           self_loop=self.self_loop, activation=nn.ReLU(), dropout=self.p)
        self.layers.append(h2o)

    def forward(self, g):
        sequence = []
        for i, layer in enumerate(self.layers):
            # Node update
            g.ndata['h'] = layer(g, g.ndata['h'], g.edata['edge_type'])
            # Jumping knowledge connexion
            sequence.append(g.ndata['h'])
            if self.batch_norm:
                g.ndata['h'] = self.batch_norm_layers[i](g.ndata['h'])
        # Concatenation :
        g.ndata['h'] = torch.cat(sequence, dim=1)  # Num_nodes * (h_dim*num_layers)
        out = self.pool(g, g.ndata['h'])
        return out


class Decoder(nn.Module):
    """
        NN which makes a prediction (fp or binding/non binding) from a pooled
        graph embedding.

        Linear/ReLu layers with Sigmoid in output since fingerprints between 0 and 1.
    """

    def __init__(self, in_dim=96, out_dim=1, hidden_dim=32, num_layers=3, dropout=0.2, batch_norm=True):
        super(Decoder, self).__init__()
        # self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = nn.Sigmoid()

        # create layers
        self.layers, self.batch_norms = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()
        batch_norms = nn.ModuleList()
        layers.append(nn.Linear(self.in_dim, self.hidden_dim))
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # hidden to output
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        batch_norms.append(nn.BatchNorm1d(self.out_dim))

        return layers, batch_norms

    def forward(self, x):
        output = x
        for layer in range(self.num_layers):
            output = self.layers[layer](output)
            if self.batch_norm:
                output = self.batch_norms[layer](output)
            if layer < self.num_layers - 1:
                output = F.dropout(F.relu(output), self.dropout, training=self.training)
        if self.activation is not None:
            output = self.activation(output)
        return output


class LigandGraphEncoder(nn.Module):
    def __init__(self,
                 l_size=32,
                 gcn_hdim=32,
                 gcn_layers=3,
                 features_dim=22,
                 num_rels=4,
                 batch_norm=False,
                 cut_embeddings=False
                 ):
        super(LigandGraphEncoder, self).__init__()
        self.features_dim = features_dim
        self.gcn_hdim = gcn_hdim
        self.gcn_layers = gcn_layers
        self.num_rels = num_rels
        # To use on optimol's embeddings
        self.cut_embeddings = cut_embeddings
        # Bottleneck
        self.l_size = l_size
        # layers:
        self.encoder = RGCN(self.features_dim, self.gcn_hdim, self.num_rels, self.gcn_layers, num_bases=-1,
                            batch_norm=batch_norm)
        self.encoder_mean = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)
        self.encoder_logv = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)

    @classmethod
    def from_pretrained(cls, trained_dir):
        # Loads trained model weights, with or without the affinity predictor
        params = json.load(open(os.path.join(trained_dir, 'params.json'), 'r'))
        weight_path = os.path.join(trained_dir, 'weights.pth')
        ligraph_encoder = cls(**params, cut_embeddings=True)
        whole_state_dict = torch.load(weight_path)
        filtered_state_dict = {}
        for (k, v) in whole_state_dict.items():
            if 'encoder' in k:
                if k.startswith('encoder.layers'):
                    filtered_state_dict[k.replace('weight', 'linear_r.W')] = v
                else:
                    filtered_state_dict[k] = v
        ligraph_encoder.load_state_dict(filtered_state_dict)
        return ligraph_encoder

    def forward(self, g):
        g.ndata['h'] = g.ndata['node_features']
        # Weird optimol pretrained_model
        if self.cut_embeddings:
            g.ndata['h'] = g.ndata['h'][:, :-6]
        e_out = self.encoder(g)
        mu = self.encoder_mean(e_out)
        return mu


class RNAEncoder(nn.Module):
    """
        Model for producing node embeddings.
    """

    def __init__(self, in_dim=4, hidden_dim=64, num_hidden_layers=3, subset_pocket_nodes=True,
                 batch_norm=True, num_rels=20, dropout=0.2, num_bases=-1):
        super(RNAEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_rels = num_rels
        num_bases = num_rels if num_bases == -1 else num_bases
        self.num_bases = num_bases
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.subset_pocket_nodes = subset_pocket_nodes

        self.layers, self.batch_norms = self.build_model()

    def build_model(self):
        layers = nn.ModuleList()
        batch_norms = nn.ModuleList()

        # input feature is just node degree
        i2h = self.build_hidden_layer(self.in_dim, self.hidden_dim)
        layers.append(i2h)
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        for i in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(self.hidden_dim, self.hidden_dim)
            layers.append(h2h)
            batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        # hidden to output
        h2o = self.build_output_layer(self.hidden_dim, self.hidden_dim)
        batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(h2o)
        return layers, batch_norms

    @property
    def current_device(self):
        """
        :return: current device this model is on
        """
        return next(self.parameters()).device

    def build_hidden_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels,
                            regularizer='basis' if self.num_rels > 0 else None,
                            num_bases=self.num_bases,
                            activation=None)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels, num_bases=self.num_bases,
                            regularizer='basis' if self.num_rels > 0 else None,
                            activation=None)

    def forward(self, g):
        h = g.ndata['nt_features']
        for i, layer in enumerate(self.layers):
            h = layer(g, h, g.edata['edge_type'])
            if self.batch_norm:
                h = self.batch_norms[i](h)

            if i < self.num_hidden_layers:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

        g.ndata['h'] = h

        graphs, embeddings = g, h
        if self.subset_pocket_nodes:
            # This tedious step is necessary, otherwise subgraphing looses track of the batch
            graphs = dgl.unbatch(g)
            all_subgraphs = []
            all_embs = []
            for graph in graphs:
                subgraph = dgl.node_subgraph(graph, graph.ndata['in_pocket'])
                embeddings = subgraph.ndata.pop('h')
                all_subgraphs.append(subgraph)
                all_embs.append(embeddings)
            graphs = dgl.batch(all_subgraphs)
            embeddings = torch.cat(all_embs, dim=0)
        return graphs, embeddings


class VSModel(nn.Module):
    def __init__(self,
                 encoder,
                 lig_encoder,
                 decoder,
                 pool_dim=64
                 ):
        """

        :param dims: the embeddings dimensions
        :param attributor_dims: the number of motifs to look for
        :param num_rels: the number of possible edge types
        :param num_bases: technical rGCN option
        :param rec: the constant in front of reconstruction loss
        :param mot: the constant in front of motif detection loss
        :param orth: the constant in front of dictionnary orthogonality loss
        :param attribute: Wether we want the network to use the attribution module
        """
        super(VSModel, self).__init__()
        pooling_gate_nn = nn.Linear(pool_dim, 1)
        self.pool = GlobalAttentionPooling(pooling_gate_nn)
        self.encoder = encoder
        self.decoder = decoder
        self.lig_encoder = lig_encoder

    def predict_ligands(self, pocket, ligands):
        g = pocket['graph']
        with torch.no_grad():
            g, embeddings = self.encoder(g)
            graph_emb = self.pool(g, embeddings)
            lig_embs = self.lig_encoder(ligands)
            graph_emb = graph_emb.expand(len(lig_embs), -1)
            pred = torch.cat((graph_emb, lig_embs), dim=1)
            pred = self.decoder(pred)
            return pred

    def forward(self, pocket, ligand):
        g = pocket['graph']
        g, embeddings = self.encoder(g)
        graph_emb = self.pool(g, embeddings)
        lig_emb = self.lig_encoder(ligand)
        pred = torch.cat((graph_emb, lig_emb), dim=1)
        pred = self.decoder(pred)
        return pred


if __name__ == '__main__':
    import pickle

    rna_encoder = RNAEncoder()
    lig_encoder = LigandGraphEncoder()
    decoder = Decoder()
    model = VSModel(encoder=rna_encoder, lig_encoder=lig_encoder, decoder=decoder)
    data = pickle.load(open('toto_data.p', 'rb'))
    pocket_name = data['group_rep']
    actives = data['active_ligands'][0]
    inactives = data['inactive_ligands'][0]
    pocket = data['pocket']
    model.eval()
    actives_scores = model.predict_ligands(pocket, actives)[:, 0].numpy()
    inactives_scores = model.predict_ligands(pocket, inactives)[:, 0].numpy()
    a = 1
