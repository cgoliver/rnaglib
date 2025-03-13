import torch
import networkx as nx

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.algorithms import fix_buggy_edges

from .representation import Representation


class GraphRepresentation(Representation):
    """
    Converts RNA into a Leontis-Westhof graph (2.5D) where nodes are residues
    and edges are either base pairs or backbones. Base pairs are annotated with the
    Leontis-Westhof classification for canonical and non-canonical base pairs.
    """

    def __init__(
            self,
            clean_edges=True,
            framework="nx",
            edge_map=GRAPH_KEYS["edge_map"][TOOL],
            etype_key="LW",
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

        super().__init__(**kwargs)
        pass

    def __call__(self, rna_graph, features_dict):
        if self.clean_edges:
            base_graph = fix_buggy_edges(graph=rna_graph)
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
        edge_attrs = [self.edge_map[data[self.etype_key]] for u, v, data in sorted(graph.edges(data=True))]
        edge_attrs = torch.tensor(edge_attrs)
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
