import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool, GlobalAttention
from torch_geometric.data import Batch, Data
import torch_geometric.utils as pyg_utils


class RGCN(nn.Module):
    """ RGCN encoder with num_hidden_layers + 2 RGCN layers, and sum pooling. """

    def __init__(self, features_dim, h_dim, num_rels, num_layers, num_bases=None, gcn_dropout=0, batch_norm=False,
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
        self.pool = global_add_pool  # Equivalent to SumPooling in DGL

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = RGCNConv(self.features_dim, self.h_dim, self.num_rels, num_bases=self.num_bases)
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_layers - 2):
            h2h = RGCNConv(self.h_dim, self.h_dim, self.num_rels, num_bases=self.num_bases)
            self.layers.append(h2h)
        # hidden to output
        h2o = RGCNConv(self.h_dim, self.h_dim, self.num_rels, num_bases=self.num_bases)
        self.layers.append(h2o)

    def forward(self, data):
        sequence = []
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for i, layer in enumerate(self.layers):
            # Node update
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.p, training=self.training)

            # Jumping knowledge connection
            sequence.append(x)

            if self.batch_norm:
                x = self.batch_norm_layers[i](x)

        # Concatenation
        x = torch.cat(sequence, dim=1)  # Num_nodes * (h_dim*num_layers)
        out = self.pool(x, batch)
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
        self.encoder = RGCN(self.features_dim, self.gcn_hdim, self.num_rels, self.gcn_layers, batch_norm=batch_norm)
        self.encoder_mean = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)
        self.encoder_logv = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)

    def forward(self, data):
        # Handle feature cutting if needed
        if self.cut_embeddings:
            data.x = data.x[:, :-6]

        e_out = self.encoder(data)
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
        return RGCNConv(in_dim, out_dim, self.num_rels, num_bases=self.num_bases)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim):
        return RGCNConv(in_dim, out_dim, self.num_rels, num_bases=self.num_bases)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)

            if self.batch_norm:
                x = self.batch_norms[i](x)

            if i < self.num_hidden_layers:
                x = F.dropout(x, self.dropout, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.dropout, training=self.training)

        # Store node embeddings in data
        data.x = x

        if self.subset_pocket_nodes:
            # This part requires mapping from the DGL batch subgraphing to PyG equivalent
            # We need to collect nodes that are in_pocket and create subgraphs

            # Split the batch into individual graphs
            graphs = Batch.to_data_list(data)
            all_subgraphs = []
            all_embs = []

            for graph in graphs:
                # Get nodes that are in the pocket. Handle case with single node
                pocket_mask = graph.in_pocket
                if pocket_mask.dim() == 0:
                    pocket_mask = pocket_mask.unsqueeze(0)

                sub_x = graph.x[pocket_mask]
                sub_edge_index, sub_edge_attr = pyg_utils.subgraph(pocket_mask, graph.edge_index, graph.edge_attr,
                                                                   relabel_nodes=True)
                subgraph = Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr)

                all_subgraphs.append(subgraph)
                all_embs.append(sub_x)

            # Combine all subgraphs into a new batch
            batch_data = Batch.from_data_list(all_subgraphs)
            embeddings = torch.cat(all_embs, dim=0)

            return batch_data, embeddings

        return data, x


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
        :param attribute: Whether we want the network to use the attribution module
        """
        super(VSModel, self).__init__()
        pooling_gate_nn = nn.Linear(pool_dim, 1)
        self.pool = GlobalAttention(pooling_gate_nn)
        self.encoder = encoder
        self.decoder = decoder
        self.lig_encoder = lig_encoder

    def predict_ligands(self, pocket, ligands):
        data = pocket['graph']
        with torch.no_grad():
            data, embeddings = self.encoder(data)
            graph_emb = self.pool(embeddings, data.batch)
            lig_embs = self.lig_encoder(ligands)
            graph_emb = graph_emb.expand(len(lig_embs), -1)
            pred = torch.cat((graph_emb, lig_embs), dim=1)
            pred = self.decoder(pred)
            return pred

    def forward(self, pocket, ligand):
        data = pocket['graph']
        data, embeddings = self.encoder(data)
        graph_emb = self.pool(embeddings, data.batch)
        lig_emb = self.lig_encoder(ligand)
        # # handle empty ligands as zero embs
        # existing_ids = ligand.batch.unique()
        # if len(existing_ids) != ligand.batch_size:
        #     result = torch.zeros((ligand.batch_size, lig_emb.shape[-1]))
        #     result[existing_ids] = lig_emb
        joint = torch.cat((graph_emb, lig_emb), dim=1)
        pred = self.decoder(joint)
        return pred
