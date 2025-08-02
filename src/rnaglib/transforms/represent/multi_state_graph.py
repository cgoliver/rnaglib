import torch
from torch_geometric.utils import to_undirected, coalesce
import networkx as nx

from rnaglib.algorithms.graph_algos import remove_noncanonicals
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.algorithms import fix_buggy_edges, remove_noncanonical_edges
from rnaglib.config.feature_encoders import EDGE_FEATURE_MAP


from .representation import Representation


class MultiStateGraphRepresentation(Representation):
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
            canonical=False,
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
        self.canonical = canonical

        super().__init__(**kwargs)
        pass

    def __call__(self, rna_graph, features_dict):
        if self.clean_edges:
            if not isinstance(rna_graph, list):
                base_graph = fix_buggy_edges(graph=rna_graph)
            else:
                base_graph = [fix_buggy_edges(graph=conf_graph) for conf_graph in rna_graph]
        else:
            base_graph = rna_graph

        if self.canonical:
            if not isinstance(rna_graph, list):
                base_graph = remove_noncanonical_edges(base_graph)
            else:
                base_graph = [remove_noncanonical_edges(conf_graph) for conf_graph in base_graph]

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

        from_multi_state = isinstance(graph, list)
        if from_multi_state:
            assert self.framework=="pyg", (
                f"Framework {self.framework} not supported for multi-state graph representation. " f"Only pyg is supported for multi-state graphs."
            )
            graph_list = graph
            features_dict_list = features_dict
        else:
            graph_list = [graph]
            features_dict_list = [features_dict]

        x_list = []
        y_list = []
        edge_index_list = []

        for graph, features_dict in zip(graph_list, features_dict_list):
            # for some reason from_networkx is not working so doing by hand
            # not super efficient at the moment
            node_map = {n: i for i, n in enumerate(sorted(graph.nodes()))}
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

            edge_index = [[node_map[u], node_map[v]] for u, v in sorted(graph.edges())]
            edge_index = torch.tensor(edge_index, dtype=torch.long).T

            x_list.append(x)
            y_list.append(y)
            edge_index_list.append(edge_index)
        
        if from_multi_state:
            x = torch.stack(x_list).permute(1,0,2)
            y = torch.stack(y_list).permute(1,0)
            edge_index = to_undirected(
                coalesce(torch.concat(edge_index_list, dim=1))
            )
        else:
            x = x_list[0]
            y = y_list[0]
            edge_index = edge_index_list[0]

        edge_attrs_list = []

        for graph, features_dict in zip(graph_list, features_dict_list):
            inverted_dict = {v: k for k, v in node_map.items()}

            conf_edge_attrs_list = []
            for pair in edge_index.T.tolist():
                names_pair = (inverted_dict[pair[0]],inverted_dict[pair[1]])
                if names_pair in graph.edges():
                    conf_edge_attrs_list.append(EDGE_FEATURE_MAP[self.etype_key].encode(graph.edges()[names_pair][self.etype_key]))
                else:
                    conf_edge_attrs_list.append(EDGE_FEATURE_MAP[self.etype_key].encode_default())
            edge_attrs_tensor = torch.stack(conf_edge_attrs_list)
            edge_attrs_list.append(edge_attrs_tensor)
        
        if from_multi_state:
            edge_attrs = torch.stack(edge_attrs_list).permute(1,0,2)
        else:
            edge_attrs = edge_attrs_list[0]

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
            return batch
