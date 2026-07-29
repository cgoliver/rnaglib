import os
import re

import pickle
import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.algorithms import fix_buggy_edges, get_sequences

from .graph import GraphRepresentation

_POS_RE = re.compile(r"(-?\d+)(.*)")


def _node_sort_key(node_id):
    """Sort key for a '<pdbid>.<chain>.<pos>' node id.

    `pos` may carry a PDB insertion code (e.g. "17A"), which is not a
    plain integer, so it is split into a numeric part and a code suffix
    to keep residues ordered as 17 < 17A < 18.
    """
    chain, pos = node_id.split(".")[1], node_id.split(".")[2]
    num, ins_code = _POS_RE.match(pos).groups()
    return (chain, int(num), ins_code)


class DistographRepresentation(GraphRepresentation):

    def __init__(
        self,
        distograms_path,
        distogram_edges=True,
        distogram_edge_features=False,
        B=10,
        tau=1e-4,
        distogram_files_prefix="distogram_",
        distogram_files_suffix="_model_0.pkl",
        framework="pyg",
        clean_edges=True,
        edge_map=GRAPH_KEYS["edge_map"][TOOL],
        etype_key="LW",
        distogram_only=False,
        one_hot=True,
        **kwargs,
    ):
        self.distograms_path = distograms_path
        self.distogram_edges = distogram_edges
        self.distogram_edge_features = distogram_edge_features
        self.B = B
        self.tau = tau
        self.distogram_files_prefix = distogram_files_prefix
        self.distogram_files_suffix = distogram_files_suffix
        self.distogram_only = distogram_only
        self.one_hot = one_hot
        super().__init__(framework, clean_edges, edge_map, etype_key, **kwargs)
        pass

    def __call__(self, rna_graph, features_dict):

        if self.clean_edges:

            base_graph = fix_buggy_edges(graph=rna_graph, label=self.etype_key, edge_map=self.edge_map)

        else:

            base_graph = rna_graph

        if self.framework == "nx":

            raise ValueError("Distographs only supported with pyg framework, not networkx")
        
        if self.framework == "dgl":

            raise ValueError("Distographs only supported with pyg framework, not dgl")
        
        if self.framework == "pyg":

            pyg_graph = super().to_pyg(base_graph, features_dict)

            with open(os.path.join(self.distograms_path,f"{self.distogram_files_prefix}{rna_graph.name}{self.distogram_files_suffix}"),"rb") as f:

                distogram_dict = pickle.load(f)
                distogram = distogram_dict["distogram"]["softmax"]
            
            nb_bins = distogram.shape[2]
            chain_dict = get_sequences(base_graph, verbose=False)
            sorted_distogram_residues = [item for chain in sorted(chain_dict.keys()) for item in chain_dict[chain][1]]

            if self.distogram_edges:
                dist_tensor = torch.from_numpy(distogram)
                proba_matrix = dist_tensor[:, :, :self.B].sum(dim=2)
                proba_matrix.fill_diagonal_(float(0))
                new_edge_indices = torch.nonzero(proba_matrix > self.tau, as_tuple=False)
                node_map = {n: i for i, n in enumerate(sorted(base_graph.nodes(), key=_node_sort_key))}
                new_edges = [[node_map[sorted_distogram_residues[u]],node_map[sorted_distogram_residues[v]]] for u, v in new_edge_indices]
                new_edges = torch.tensor(new_edges, dtype=torch.long).T

                max_occupied_index = max(self.edge_map.values()) if self.graph_construction=="base_pair" else 0

                new_edge_attr = torch.full((new_edges.size(1),), max_occupied_index+1, dtype=torch.long)

                if self.distogram_only:

                    pyg_graph.edge_index = new_edges
                    pyg_graph.edge_attr = new_edge_attr

                else:

                    pyg_graph.edge_index = torch.cat([pyg_graph.edge_index, new_edges], dim=1)
                    pyg_graph.edge_attr = torch.cat([pyg_graph.edge_attr, new_edge_attr], dim=0)

            if self.distogram_edge_features:
                
                distogram_map = {n: i for i, n in enumerate(sorted_distogram_residues)}
                sorted_graph_residues = sorted(base_graph.nodes(), key=_node_sort_key)
                edge_distances = []
                for i, j in pyg_graph.edge_index.t():
                    try:
                        distance = distogram[distogram_map[sorted_graph_residues[i]],distogram_map[sorted_graph_residues[j]]]
                    except:
                        distance = np.zeros(nb_bins)
                    edge_distances.append(distance)

                edge_distances = torch.tensor(edge_distances)

                if self.one_hot:

                    num_classes = len(self.edge_map) + int(self.distogram_edges)
                    edge_attr_one_hot = F.one_hot(pyg_graph.edge_attr.long(), num_classes=num_classes)
                    pyg_graph.edge_feats = torch.cat([edge_attr_one_hot, edge_distances], dim=1).float()

                else:

                    pyg_graph.edge_feats = edge_distances

            return pyg_graph

    def batch(self, samples):
        """
        Batch a list of graph samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        if self.framework == "nx":
            raise ValueError("Distographs only supported with pyg framework, not networkx")
        if self.framework == "dgl":
            raise ValueError("Distographs only supported with pyg framework, not dgl")
        if self.framework == "pyg":
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(samples)
            # sometimes batching changes dtype from int to float32?
            batch.edge_index = batch.edge_index.to(torch.int64)
            batch.edge_attr = batch.edge_attr.to(torch.int64)
            if self.distogram_edge_features:
                batch.edge_feats = batch.edge_feats.to(torch.float32)
            return batch


        