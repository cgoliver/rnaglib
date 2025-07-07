import torch
import networkx as nx
from torch_geometric.utils import coalesce, to_undirected
import torch_cluster

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.config.feature_encoders import NODE_FEATURE_MAP
from rnaglib.algorithms import fix_buggy_edges, internal_coords, internal_vecs, rbf_expansion, positional_encoding, normed_vec, get_backbone_coords

from .representation import Representation


class GVPGraphRepresentation(Representation):
    """
    Converts RNA into a graph akin to be processed by a GVP (geometric vector perceptron)-
    like architecture where nodes are residues and edges are either base pairs or backbones.
    Node and edge features include both scalar and vector features.

    Reference: Jing, B., Eismann, S., Suriana, P., Townshend, R. J., & Dror, R. (2020). Learning
    from protein structure with geometric vector perceptrons. arXiv preprint arXiv:2009.01411.
    """

    def __init__(
        self,
        clean_edges=True,
        num_rbf = 32,
        num_posenc = 32,
        pyrimidine_bb_indices = ["P","C4'","N1"],
        purine_bb_indices = ["P","C4'","N9"],
        **kwargs,
    ):
        self.num_rbf = num_rbf
        self.num_posenc = num_posenc
        self.clean_edges = clean_edges
        self.pyrimidine_bb_indices = pyrimidine_bb_indices
        self.purine_bb_indices = purine_bb_indices

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

        # for some reason from_networkx is not working so doing by hand
        # not super efficient at the moment
        node_map = {n: i for i, n in enumerate(sorted(graph.nodes()))}
        seq, y = None, None
        if "nt_features" in features_dict:
            seq = (
                torch.stack([features_dict["nt_features"][n] for n in node_map.keys()])
                if "nt_features" in features_dict
                else None
            )
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
        coords = get_backbone_coords(graph, node_map, self.pyrimidine_bb_indices, self.purine_bb_indices)

        # Mask for missing coordinates for any backbone atom: num_res
        mask_coords = torch.BoolTensor(mask_coords)
        # Also mask non-standard nucleotides
        mask_coords = (mask_coords) & (seq != self.letter_to_num["_"])

        # Node internal coordinates (scalars) and normalised vectors
        dihedrals, angles, lengths = internal_coords(coords)
        angle_stack = torch.cat([dihedrals, angles], dim=-1)
        lengths = torch.log(lengths + self.distance_eps)
        internal_coords_feat = torch.cat([torch.cos(angle_stack), torch.sin(angle_stack), lengths], dim=-1)
        internal_vecs_feat = internal_vecs(coords)

        # Remove residues with missing coordinates or non-standard nucleotides
        seq = seq[mask_coords]
        coords_list = coords_list[:, mask_coords] # [:, :, 1]  # only retain C4'
        internal_coords_feat = internal_coords_feat[:, mask_coords]
        internal_vecs_feat = internal_vecs_feat[:, mask_coords]
        
        # Construct merged edge index
        edge_index = []
        for coord in coords:
            # K-nearest neighbour graph using centroids of each neucleotide
            edge_index.append(torch_cluster.knn_graph(coord.mean(1), self.top_k))
        edge_index = to_undirected(coalesce(
            torch.concat(edge_index, dim=1)
        ))

        # Edge displacement vectors: num_edges x num_conf x num_res x 3
        edge_vectors = coords[edge_index[0]] - coords[edge_index[1]]
        edge_lengths = torch.sqrt((edge_vectors ** 2).sum(dim=-1) + self.distance_eps) #.unsqueeze(-1)

        # Edge RBF features: num_edges x num_conf x num_rbf
        edge_rbf = rbf_expansion(edge_lengths, num_rbf=self.num_rbf)

        # Edge positional encodings: num_edges x num_conf x num_posenc
        edge_posenc = positional_encoding(
            (edge_index[0] - edge_index[1])[..., None], self.num_posenc
        ).unsqueeze_(1).repeat(1, self.max_num_conformers, 1)

        node_s = internal_coords_feat
        node_v = internal_vecs_feat
        edge_s = torch.cat([edge_rbf, edge_posenc, torch.log(edge_lengths)], dim=-1)
        edge_v = normed_vec(edge_vectors) # .unsqueeze(-2)
        node_s, node_v, edge_s, edge_v = map(
            torch.nan_to_num,
            (node_s, node_v, edge_s, edge_v)
        )

        edge_index = [[node_map[u], node_map[v]] for u, v in sorted(graph.edges())]
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        edge_attrs = [self.edge_map[data[self.etype_key]] for u, v, data in sorted(graph.edges(data=True))]
        edge_attrs = torch.tensor(edge_attrs)
        return Data(h_V=(node_s, node_v), y=y, edge_index=edge_index, h_E=(edge_s, edge_v), seq=seq)

    @property
    def name(self):
        return "graph"

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
        batch.edge_attr = batch.edge_attr.to(torch.int64)
        return batch
