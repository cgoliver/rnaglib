import os
import pickle

import dgl
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

    def __init__(self, cache_path=None):
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir,  '../data/rna_vs/edges_and_nodes_map.p'), "rb") as f:
            self.edge_map = pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map = pickle.load(f)
            self.charges_map = pickle.load(f)

        if cache_path is not None:
            self.cached_graphs = pickle.load(open(cache_path, 'rb'))
        else:
            self.cached_graphs = list()

    @staticmethod
    def set_as_one_hot_feat(graph_nx, edge_map, node_label, default_value=None):
        one_hot = {a: oh_tensor(edge_map.get(label, default_value), len(edge_map)) for a, label in
                   (nx.get_node_attributes(graph_nx, node_label)).items()}
        nx.set_node_attributes(graph_nx, name=node_label, values=one_hot)

    def as_one_hot(self, graph_nx):
        self.set_as_one_hot_feat(graph_nx, edge_map=self.at_map, node_label='atomic_num', default_value=6)
        self.set_as_one_hot_feat(graph_nx, edge_map=self.charges_map, node_label='formal_charge', default_value=0)
        self.set_as_one_hot_feat(graph_nx, edge_map=self.chi_map, node_label='num_explicit_hs', default_value=0)
        self.set_as_one_hot_feat(graph_nx, edge_map=self.chi_map, node_label='is_aromatic', default_value=0)
        self.set_as_one_hot_feat(graph_nx, edge_map=self.chi_map, node_label='chiral_tag', default_value=0)

    def mol2_to_graph_one(self, mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, sanitize=False)

        atom_names = []
        in_section = False
        with open(mol2_path) as mol2:
            for row in mol2:
                if row.startswith("@<TRIPOS>ATOM"):
                    in_section = True
                    continue
                elif row.startswith("@<TRIPOS>BOND"):
                    break
                if in_section:
                    atom_names.append(row.split()[1])
                else:
                    continue
        graph_nx = mol_to_nx(mol)
        for node, i in enumerate(graph_nx.nodes()):
            graph_nx.nodes[node]['atom_name'] = atom_names[i]
        graph_dgl = self.nx_mol_to_dgl(graph_nx)
        return graph_dgl, graph_nx

    def nx_mol_to_dgl(self, graph_nx):
        try:
            assert graph_nx is not None
            # Get edges as one hot
            edge_type = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                         (nx.get_edge_attributes(graph_nx, 'bond_type')).items()}
            nx.set_edge_attributes(graph_nx, name='edge_type', values=edge_type)

            # Set node features as one_hot
            self.as_one_hot(graph_nx)

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

    def smiles_to_graph_one(self, smiles):
        if smiles in self.cached_graphs:
            return self.cached_graphs[smiles]
        graph_nx = smiles_to_nx(smiles)
        return self.nx_mol_to_dgl(graph_nx)

    def smiles_to_graph_list(self, smiles_list):
        graphs = []
        for i, sm in enumerate(smiles_list):
            graph = self.smiles_to_graph_one(sm)
            graphs.append(graph)
        batch = dgl.batch(graphs)
        return batch

    def collate_fn(self, samples):
        # TODO unDGL it
        batched_graph = dgl.batch([sample for sample in samples])
        return batched_graph
