import os
import pickle

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem


def mol_to_nx(mol):
    mol_graph = nx.Graph()

    for atom in mol.GetAtoms():
        mol_graph.add_node(atom.GetIdx(),
                           atomic_num=atom.GetAtomicNum(),
                           formal_charge=atom.GetFormalCharge(),
                           chiral_tag=atom.GetChiralTag(),
                           num_explicit_hs=atom.GetNumExplicitHs(),
                           is_aromatic=atom.GetIsAromatic())

    for bond in mol.GetBonds():
        mol_graph.add_edge(bond.GetBeginAtomIdx(),
                           bond.GetEndAtomIdx(),
                           bond_type=bond.GetBondType())
    return mol_graph


def smiles_to_nx(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        nx_graph = mol_to_nx(mol)
        return nx_graph
    except:
        return None


def oh_tensor(category, n):
    # One-hot float tensor construction
    t = torch.zeros(n, dtype=torch.float)
    t[category] = 1.0
    return t


class MolGraphEncoder:
    """
    Stateful encoder for using cashed computations
    """

    def __init__(self, cache_path=None, framework="dgl"):
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir, 'data/edges_and_nodes_map.p'), "rb") as f:
            self.edge_map = pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map = pickle.load(f)
            self.charges_map = pickle.load(f)

        if cache_path is not None:
            self.cached_graphs = pickle.load(open(cache_path, 'rb'))
        else:
            self.cached_graphs = list()
        if framework not in {"dgl", "pyg"}:
            raise NotImplementedError("Ligand framework must be dgl or pyg")
        self.framework = framework

    @staticmethod
    def set_as_one_hot_feat(graph_nx, edge_map, node_label, default_value=None):
        one_hot = {a: oh_tensor(edge_map.get(label, default_value), len(edge_map)) for a, label in
                   (nx.get_node_attributes(graph_nx, node_label)).items()}
        nx.set_node_attributes(graph_nx, name=node_label, values=one_hot)

    def as_one_hot(self, graph_nx):
        graph_nx_oh = graph_nx.copy()
        self.set_as_one_hot_feat(graph_nx_oh, edge_map=self.at_map, node_label='atomic_num', default_value=6)
        self.set_as_one_hot_feat(graph_nx_oh, edge_map=self.charges_map, node_label='formal_charge', default_value=0)
        self.set_as_one_hot_feat(graph_nx_oh, edge_map=self.chi_map, node_label='num_explicit_hs', default_value=0)
        self.set_as_one_hot_feat(graph_nx_oh, edge_map=self.chi_map, node_label='is_aromatic', default_value=0)
        self.set_as_one_hot_feat(graph_nx_oh, edge_map=self.chi_map, node_label='chiral_tag', default_value=0)
        return graph_nx_oh

    def nx_mol_to_dgl(self, graph_nx):
        import dgl

        try:
            assert graph_nx is not None
            # Get edges as one hot
            edge_type = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                         (nx.get_edge_attributes(graph_nx, 'bond_type')).items()}
            nx.set_edge_attributes(graph_nx, name='edge_type', values=edge_type)

            # Set node features as one_hot
            graph_nx = self.as_one_hot(graph_nx)

            # to dgl
            node_features = ['atomic_num', 'formal_charge', 'num_explicit_hs', 'is_aromatic', 'chiral_tag']
            graph_nx = graph_nx.to_directed()
            graph_dgl = dgl.from_networkx(nx_graph=graph_nx,
                                          node_attrs=node_features,
                                          edge_attrs=['edge_type'])

            N = graph_dgl.number_of_nodes()
            graph_dgl.ndata['node_features'] = torch.cat([graph_dgl.ndata[f].view(N, -1) for f in node_features], dim=1)
            return graph_dgl
        except Exception as e:
            print(f"Failed with exception {e}")
            return dgl.graph(([], []))

    def nx_mol_to_pyg(self, graph_nx):
        from torch_geometric.data import Data

        try:
            assert graph_nx is not None

            # Get edges as one hot
            edge_type = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                         (nx.get_edge_attributes(graph_nx, 'bond_type')).items()}
            nx.set_edge_attributes(graph_nx, name='edge_type', values=edge_type)

            # Set node features as one_hot
            graph_nx = self.as_one_hot(graph_nx)

            # Extract edge indices and attr
            graph_nx = graph_nx.to_directed()
            edge_index = torch.tensor([[u, v] for u, v in graph_nx.edges()], dtype=torch.long).t().contiguous()
            edge_attr = torch.stack([graph_nx[u][v]['edge_type'] for u, v in graph_nx.edges()])

            # Extract node features and concatenate them
            node_features = ['atomic_num', 'formal_charge', 'num_explicit_hs', 'is_aromatic', 'chiral_tag']
            x = torch.cat([torch.stack([graph_nx.nodes[i][feat] for i in range(len(graph_nx.nodes))])
                           for feat in node_features], dim=1)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            return data

        except Exception as e:
            print(f"Failed with exception {e}")
            # Return empty graph
            return Data(x=torch.empty((0, 22)),
                        edge_index=torch.empty((2, 0), dtype=torch.long),
                        edge_attr=torch.empty((0), dtype=torch.long))

    def smiles_to_graph_one(self, smiles):
        if smiles in self.cached_graphs:
            return self.cached_graphs[smiles]
        graph_nx = smiles_to_nx(smiles)

        if self.framework == 'dgl':
            return self.nx_mol_to_dgl(graph_nx)
        return self.nx_mol_to_pyg(graph_nx)

    def smiles_to_graph_list(self, smiles_list):
        graphs = []
        for i, sm in enumerate(smiles_list):
            graph = self.smiles_to_graph_one(sm)
            graphs.append(graph)
        return self.collate_fn(graphs)

    def collate_fn(self, graphs):
        if self.framework == 'dgl':
            import dgl
            batch = dgl.batch(graphs)
        else:
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(graphs)
        return batch
