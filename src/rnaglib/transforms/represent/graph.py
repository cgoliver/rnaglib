import torch
import networkx as nx
import copy
from torch_geometric.utils import to_undirected
from torch_geometric.nn.pool import knn_graph

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.algorithms import fix_buggy_edges
from rnaglib.utils import rbf_expand

from .representation import Representation


class GraphRepresentation(Representation):
    """
    Converts RNA into a Leontis-Westhof graph (2.5D) where nodes are residues
    and edges are either base pairs or backbones. Base pairs are annotated with the
    Leontis-Westhof classification for canonical and non-canonical base pairs.
    """

    def __init__(
        self,
        framework="nx",
        clean_edges=True,
        edge_map=GRAPH_KEYS["edge_map"][TOOL],
        etype_key="LW",
        graph_construction="base_pair",
        top_k=16,
        threshold=10,
        representative="P",
        purine_representative="P",
        pyrimidine_representative="P",
        distance_edge_features=False,
        **kwargs,
    ):

        authorized_frameworks = {"nx", "dgl", "pyg"}
        assert framework in authorized_frameworks, (
            f"Framework {framework} not supported for this representation. " f"Choose one of {authorized_frameworks}."
        )
        self.framework = framework

        self.clean_edges = clean_edges
        self.etype_key = etype_key
        self.edge_map = edge_map

        self.graph_construction = graph_construction
        self.top_k = top_k
        self.threshold = threshold
        self.representative = representative
        self.purine_representative = purine_representative
        self.pyrimidine_representative = pyrimidine_representative
        self.distance_edge_features = distance_edge_features

        super().__init__(**kwargs)
        pass

    def __call__(self, rna_graph, features_dict):
        if self.clean_edges:
            base_graph = fix_buggy_edges(graph=rna_graph, label=self.etype_key, edge_map=self.edge_map)
        else:
            base_graph = rna_graph

        if self.framework == "nx":
            return self.to_nx(base_graph, features_dict)
        if self.framework == "dgl":
            return self.to_dgl(base_graph, features_dict)
        if self.framework == "pyg":
            return self.to_pyg(base_graph, features_dict)

    def to_nx(self, graph, features_dict):
        # Get Edge Labels
        edge_type = {(u, v): self.edge_map[data[self.etype_key]] for u, v, data in graph.edges(data=True)}
        nx.set_edge_attributes(graph, name="edge_type", values=edge_type)

        # Add features and targets
        for name, encoding in features_dict.items():
            nx.set_node_attributes(graph, name=name, values=encoding)
        return graph

    def to_dgl(self, graph, features_dict):
        import dgl

        nx_graph = self.to_nx(graph, features_dict)

        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=nx_graph, edge_attrs=["edge_type"], node_attrs=features_dict.keys())
        return g_dgl

    def to_pyg(self, graph, features_dict):
        from torch_geometric.data import Data

        # for some reason from_networkx is not working so doing by hand
        # not super efficient at the moment
        node_map = {n: i for i, n in enumerate(sorted(graph.nodes(), key=lambda x: (graph.nodes[x]['chain_id'], graph.nodes[x]['label_seq_id'])))}
        x, y = None, None

        if "nt_features" in features_dict:

            x = (
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

        if self.graph_construction in ["knn","threshold"] or self.distance_edge_features:

            if self.purine_representative != self.pyrimidine_representative:

                all_heavy_atoms = nx.get_node_attributes(graph, f"heavy_atoms")
                all_attrs_pyrimidine_rep = {n: atoms.get(self.pyrimidine_representative) if atoms is not None else None for n, atoms in all_heavy_atoms.items()}
                all_attrs_purine_rep = {n: atoms.get(self.purine_representative) if atoms is not None else None for n, atoms in all_heavy_atoms.items()}
                all_attrs_base_identity = nx.get_node_attributes(graph, 'nt')
                pyrimidine_rep_coords_list = [all_attrs_pyrimidine_rep[n] if all_attrs_pyrimidine_rep[n] is not None else 3*[float('nan')] for n in node_map.keys()]
                purine_rep_coords_list = [all_attrs_purine_rep[n] if all_attrs_purine_rep[n] is not None else 3*[float('nan')] for n in node_map.keys()]
                purine_mask_list = [1 if all_attrs_base_identity[n] in ["A","G"] else 0 for n in node_map.keys()]
                pyrimidine_rep_coords = torch.tensor(pyrimidine_rep_coords_list)
                purine_rep_coords = torch.tensor(purine_rep_coords_list)
                purine_mask = torch.tensor(purine_mask_list)
                nucleotide_coords = purine_rep_coords*purine_mask.view(-1,1)+pyrimidine_rep_coords*(1-purine_mask).view(-1,1)

            else:

                all_attrs = nx.get_node_attributes(graph, f'xyz_{self.representative}')
                nucleotide_coords_list = [all_attrs[n] if all_attrs[n] is not None else float('nan') for n in node_map.keys()]
                nucleotide_coords = torch.tensor(nucleotide_coords_list)
                
            # Compute full distance matrix [N, N]
            dist_matrix = torch.cdist(nucleotide_coords, nucleotide_coords)
            

        if self.graph_construction == "knn":
            # Find k+1 smallest elements
            graph_dist_matrix = copy.deepcopy(dist_matrix)
            graph_dist_matrix.fill_diagonal_(float('inf'))
            _, indices = graph_dist_matrix.topk(self.top_k + 1, largest=False)
            # Remove the first column (self-loops)
            neighbor_indices = indices[:, 1:]
            neighbor_indices, _ = torch.sort(neighbor_indices, dim=1)
            # Construct edge_index
            source = torch.arange(nucleotide_coords.size(0), device=nucleotide_coords.device).repeat_interleave(self.top_k)
            # targets: flatten the nearest neighbor indices
            target = neighbor_indices.flatten()
            edge_index= torch.stack([source, target], dim=0)
            edge_attrs = torch.zeros(edge_index.shape[1],dtype=int)

        elif self.graph_construction == "threshold":
            graph_dist_matrix = copy.deepcopy(dist_matrix)
            graph_dist_matrix.fill_diagonal_(float('inf'))
            edges = torch.nonzero(graph_dist_matrix < self.threshold, as_tuple=False)
            edge_index = edges.t()
            edge_attrs = torch.zeros(edge_index.shape[1],dtype=int)

        else:
            edge_index = [[node_map[u], node_map[v]] for u, v in sorted(graph.edges(), key=lambda x: (graph.nodes[x[0]]['chain_id'], graph.nodes[x[0]]['label_seq_id'], graph.nodes[x[1]]['chain_id'], graph.nodes[x[1]]['label_seq_id']))]
            edge_index = torch.tensor(edge_index, dtype=torch.long).T
            edge_attrs = [self.edge_map[data[self.etype_key]] for u, v, data in sorted(graph.edges(data=True), key=lambda x: (graph.nodes[x[0]]['chain_id'], graph.nodes[x[0]]['label_seq_id'], graph.nodes[x[1]]['chain_id'], graph.nodes[x[1]]['label_seq_id']))]
            edge_attrs = torch.tensor(edge_attrs)

        if self.distance_edge_features:
            edge_distances = dist_matrix[edge_index[0, :], edge_index[1, :]]
            edge_distances.nan_to_num_(nan=0.0)
            edge_feats = rbf_expand(dists=edge_distances, num_bins=64, min_distance=2.0, max_distance=22.0)
            return Data(x=x, y=y, edge_attr=edge_attrs, edge_index=edge_index, edge_feats=edge_feats)
        
        return Data(x=x, y=y, edge_attr=edge_attrs, edge_index=edge_index)


    @property
    def name(self):
        return "graph"

    def batch(self, samples):
        """
        Batch a list of graph samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        if self.framework == "nx":
            return samples
        if self.framework == "dgl":
            import dgl

            batched_graph = dgl.batch([sample for sample in samples])
            return batched_graph
        if self.framework == "pyg":
            from torch_geometric.data import Batch
            batch = Batch.from_data_list(samples)
            # sometimes batching changes dtype from int to float32?
            batch.edge_index = batch.edge_index.to(torch.int64)
            batch.edge_attr = batch.edge_attr.to(torch.int64)
            try:
                batch.edge_feats = batch.edge_feats.to(torch.float32)
                batch.has_edge_feats_mask = torch.tensor([~torch.isnan(sample.edge_feats).any() for sample in samples])
            except:
                pass
            return batch
