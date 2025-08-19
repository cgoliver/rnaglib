import torch
import networkx as nx
from torch_geometric.utils import to_undirected
from torch_geometric.nn.pool import knn_graph
from typing import List

from rnaglib.config.feature_encoders import NODE_FEATURE_MAP, EDGE_FEATURE_MAP
from rnaglib.algorithms import fix_buggy_edges, internal_coords, internal_vecs, rbf_expansion, positional_encoding, normed_vec, get_backbone_coords, forward_reverse_vecs, get_sequences

from .representation import Representation

# Small epsilon value added to distances to avoid division by zero
DISTANCE_EPS = 0.001

class GVPGraphRepresentation(Representation):
    """
    Converts RNA into a graph akin to be processed by a GVP (geometric vector perceptron)-
    like architecture where nodes are residues and edges are either base pairs or backbones.
    Node and edge features include both scalar and vector features.

    References: 
    Jing, B., Eismann, S., Suriana, P., Townshend, R. J., & Dror, R. (2020). Learning
    from protein structure with geometric vector perceptrons. arXiv preprint arXiv:2009.01411.

    Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, Alex Morehead, 
    and Pietro Liò. gRNAde: Geometric Deep Learning for 3D RNA inverse design. International 
    Conference on Learning Representations 2025.
    """

    def __init__(
        self,
        clean_edges = True,
        graph_construction: str = "knn",
        top_k: int = 16,
        node_scalar_features: List[str] = ["nt_code"],
        node_vector_features: List[str] = ["forward_reverse_unit_vectors"],
        edge_scalar_features: List[str] = ["RBF"],
        edge_vector_features: List[str] = ["unit_vector"],
        num_rbf: int = 32,
        num_posenc: int = 32,
        center_atom: str = "P",
        pyrimidine_bead_atoms: List[str] = ["P","C4'","N1"],
        purine_bead_atoms: List[str] = ["P","C4'","N9"],
        distance_eps: float = DISTANCE_EPS,
        **kwargs,
    ):
        self.graph_construction = graph_construction
        self.top_k = top_k
        self.node_scalar_features = node_scalar_features
        self.node_vector_features = node_vector_features
        self.edge_scalar_features = edge_scalar_features
        self.edge_vector_features = edge_vector_features
        self.num_rbf = num_rbf
        self.num_posenc = num_posenc
        self.center_atom = center_atom
        self.clean_edges = clean_edges
        self.pyrimidine_bead_atoms = pyrimidine_bead_atoms
        self.purine_bead_atoms = purine_bead_atoms
        self.distance_eps = distance_eps

        super().__init__(**kwargs)
        pass

    def __call__(self, rna_graph, features_dict):

        if self.clean_edges:
            base_graph = fix_buggy_edges(graph=rna_graph)
        else:
            base_graph = rna_graph

        return self.to_pyg(base_graph, features_dict)

    def to_pyg(self, graph, features_dict):
        from torch_geometric.data import Data

        # Get ordered RNA sequences
        chain_seqs = get_sequences(graph, gap_tolerance=1, longest_only=False, min_size_return=1)
        global_index = 0
        node_map = {}
        # Indices at which the forward vector can't be computed (before discontinuities or at chain ends)
        no_forward = []
        # Indices at which the backward vector can't be computed (rightr after discontinuities or at chain beginnings)
        no_backward = []

        for chain in chain_seqs:
            no_backward.append(global_index)
            resname = chain_seqs[chain][1][0]
            ind = int(resname.split('.')[2])
            # Sort nodes
            node_map[resname] = global_index
            global_index += 1
            for resname in chain_seqs[chain][1][1:]:
                ind_new = int(resname.split('.')[2])
                if ind_new != ind + 1:
                    no_forward.append(global_index-1)
                    no_backward.append(global_index)
                ind = ind_new
                node_map[resname] = global_index
                global_index += 1
            no_forward.append(global_index-1)
        
        # Retrieve features natively available in the base graph
        base_graph_node_features = list(graph.nodes[list(graph.nodes())[0]].keys())
        base_graph_edge_features = list(graph.edges[list(graph.edges())[0]].keys())

        # Encode target variable(s)
        y = None
        if "nt_targets" in features_dict:
            list_y = [features_dict["nt_targets"][n] for n in node_map.keys()]
            # In the case of single target, pytorch CE loss expects shape (n,) and not (n,1)
            # For multi-target cases, we stack to get (n,d)
            if len(list_y[0]) == 1:
                y = torch.cat(list_y)
            else:
                y = torch.stack(list_y)
        if "rna_targets" in features_dict:
            y = features_dict["rna_targets"].clone().detach()

        # Get backbone coordinates
        nucleotide_coords, mask_coords = get_backbone_coords(graph, node_map, [self.center_atom], [self.center_atom])
        nucleotide_coords = nucleotide_coords[:, 0, :]
        compute_frame_coords = False
        if any(feat in self.node_scalar_features for feat in ["dihedrals","angles","lengths"]) or "internal_vecs" in self.node_vector_features:
            compute_frame_coords = True
            coords, mask_coords = get_backbone_coords(graph, node_map, self.pyrimidine_bead_atoms, self.purine_bead_atoms)
        
        # Mask for missing coordinates for any backbone atom: num_res
        mask_coords = torch.BoolTensor(mask_coords)

        # Encode node scalar features
        node_s_list = []
        for feature in self.node_scalar_features:
            if feature in base_graph_node_features:
                node_feature_tensor = torch.stack([NODE_FEATURE_MAP[feature].encode(graph.nodes()[n][feature]) for n in node_map.keys()])
                node_s_list.append(node_feature_tensor)
        if compute_frame_coords:
            dihedrals, angles, lengths = internal_coords(coords, mask_coords)
        if "dihedrals" in self.node_scalar_features:
            node_s_list += [torch.cos(dihedrals), torch.sin(dihedrals)]
        if "angles" in self.node_scalar_features:
            node_s_list += [torch.cos(angles), torch.sin(angles)]
        if "lengths" in self.node_scalar_features:
            lengths = torch.log(lengths + self.distance_eps)
            node_s_list.append(lengths)
        node_s = torch.cat(node_s_list, dim = -1)[mask_coords]

        # Encode node vector features
        node_v_list = []
        if "forward_reverse_unit_vectors" in self.node_vector_features:
            forward_reverse_vecs_feat = forward_reverse_vecs(nucleotide_coords, no_forward, no_backward)
            node_v_list.append(forward_reverse_vecs_feat)
        if "internal_vecs" in self.node_vector_features:
            internal_vecs_feat = internal_vecs(coords)
            node_v_list.append(internal_vecs_feat)
        node_v = torch.cat(node_v_list, dim = -2)[mask_coords]

        # Remove residues with missing coordinates or non-standard nucleotides
        if "nt_targets" in features_dict:
            y = y[mask_coords]
        if compute_frame_coords:
            coords = coords[mask_coords]

        if self.graph_construction == "knn":
            # K-nearest neighbour graph using centroids of each neucleotide
            edge_index = knn_graph(nucleotide_coords, self.top_k)
            edge_index = to_undirected(edge_index)
        elif self.graph_construction == "base_pair":
            # Base-pairing graph
            edge_index = [[node_map[u], node_map[v]] for u, v in sorted(graph.edges())]
            edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # Encode edge vector features
        edge_v_list = []
        if "unit_vector" in self.edge_vector_features:
            # Edge displacement vectors: num_edges x num_res x 3
            edge_vector = nucleotide_coords[edge_index[0]] - nucleotide_coords[edge_index[1]]
            edge_vector = normed_vec(edge_vector).unsqueeze_(-2)
            edge_v_list.append(edge_vector)
        edge_v = torch.cat(edge_v_list, dim = -2)

        # Encode edge scalar features
        edge_s_list = []
        if self.graph_construction == "base_pair":
            for feature in self.edge_scalar_features:
                if feature in base_graph_edge_features:
                    edge_feature_tensor = torch.stack([EDGE_FEATURE_MAP[feature].encode(graph.edges()[(u,v)][feature]) for u, v in sorted(graph.edges())])
                    edge_s_list.append(edge_feature_tensor)
        if "lengths" in self.edge_scalar_features:
            edge_vector = nucleotide_coords[edge_index[0]] - nucleotide_coords[edge_index[1]]
            edge_lengths = torch.sqrt((edge_vector ** 2).sum(dim=-1) + self.distance_eps)
            log_edge_lengths = torch.log(edge_lengths)
            edge_s_list.append(log_edge_lengths)
        if "RBF" in self.edge_scalar_features:
            edge_vector = nucleotide_coords[edge_index[0]] - nucleotide_coords[edge_index[1]]
            edge_lengths = torch.sqrt((edge_vector ** 2).sum(dim=-1) + self.distance_eps)
            # Edge RBF features
            edge_rbf = rbf_expansion(edge_lengths, num_rbf=self.num_rbf)
            edge_s_list.append(edge_rbf)
        if "posenc" in self.edge_scalar_features:
            # Edge positional encodings: num_edges x num_conf x num_posenc
            edge_posenc = positional_encoding(
                (edge_index[0] - edge_index[1])[..., None], self.num_posenc
            )
            edge_s_list.append(edge_posenc)
        edge_s = torch.cat(edge_s_list, dim=-1)

        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num,
            (node_s, node_v, edge_s, edge_v)
        )
        num_nodes = node_s.shape[0]

        return Data(h_V=(node_s, node_v), y=y, edge_index=edge_index, h_E=(edge_s, edge_v), num_nodes=num_nodes)

    @property
    def name(self):
        return "gvp_graph"

    def batch(self, samples):
        """
        Batch a list of graph samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(samples)
        # sometimes batching changes dtype from int to float32?
        batch.edge_index = batch.edge_index.to(torch.int64)
        return batch
