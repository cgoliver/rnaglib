import torch
import networkx as nx

from rnaglib.algorithms import get_sequences

from .representation import Representation


class SequenceRepresentation(Representation):
    """
    Represents RNA as a linear sequence following the 5'to 3' order of backbone edges. Note that this only works on single-chain. If you have a multi-chain RNA make sure to first apply the ``ChainSplitTransform``.
    RNAs. When using a graph-based framework (e.g. pyg or dgl) the RNA is stored as a linear graph with edges going in 5' to 3' as well as 3' to 3'. This can be controlled using the `backbone` argument.

    :param framework: which learning framework to store representation.
    :param backbone: if 'both' graph will have 5' -> 3' edges and 3' -> 5', if '5p3p' will only have the former and if '3p5p' only the latter.
    """

    def __init__(
        self,
        framework: str = "pyg",
        backbone: str = "both",
        **kwargs,
    ):

        authorized_frameworks = {"pyg", "torch"}
        assert framework in authorized_frameworks, (
            f"Framework {framework} not supported for this representation. " f"Choose one of {authorized_frameworks}."
        )
        self.framework = framework
        self.backbone = backbone

        super().__init__(**kwargs)
        pass

    def __call__(self, rna_graph, features_dict):
        sequence = get_sequences(rna_graph)
        assert (
            len(sequence) == 1
        ), "Sequence representation only works on single-chain RNAs. Use the ChainSplitTransform() to subdivide the whole RNA into individual chains first."

        sequence, node_ids = list(sequence.values())[0]
        self.sequence = sequence
        self.node_ids = node_ids

        seq_graph = nx.DiGraph()
        seq_graph.add_nodes_from(node_ids)
        if self.backbone in ["both", "5p3p"]:
            seq_graph.add_edges_from([(node_ids[i], node_ids[i + 1], {"LW": "B53"}) for i in range(len(node_ids) - 1)])
        if self.backbone in ["both", "3p5p"]:
            seq_graph.add_edges_from([(node_ids[i - 1], node_ids[i], {"LW": "B35"}) for i in range(1, len(node_ids))])

        if self.framework == "torch":
            return self.to_torch(seq_graph, features_dict)
        if self.framework == "dgl":
            return self.to_dgl(seq_graph, features_dict)
        if self.framework == "pyg":
            return self.to_pyg(seq_graph, features_dict)

    def to_torch(self, graph, features_dict):
        x, y = None, None
        if "nt_features" in features_dict:
            x = (
                torch.stack([features_dict["nt_features"][n] for n in self.node_ids])
                if "nt_features" in features_dict
                else None
            )
        if "nt_targets" in features_dict:
            list_y = [features_dict["nt_targets"][n] for n in self.node_ids]
            # In the case of single target, pytorch CE loss expects shape (n,) and not (n,1)
            # For multi-target cases, we stack to get (n,d)
            if len(list_y[0]) == 1:
                y = torch.cat(list_y)
            else:
                y = torch.stack(list_y)
        if "rna_targets" in features_dict:
            y = torch.tensor(features_dict["rna_targets"])

        return x

    def to_pyg(self, graph, features_dict):
        from torch_geometric.data import Data

        # for some reason from_networkx is not working so doing by hand
        # not super efficient at the moment
        x, y = None, None
        print(self.sequence)
        if "nt_features" in features_dict:
            x = (
                torch.stack([features_dict["nt_features"][n] for n in self.node_ids])
                if "nt_features" in features_dict
                else None
            )
        if "nt_targets" in features_dict:
            list_y = [features_dict["nt_targets"][n] for n in self.node_ids]
            # In the case of single target, pytorch CE loss expects shape (n,) and not (n,1)
            # For multi-target cases, we stack to get (n,d)
            if len(list_y[0]) == 1:
                y = torch.cat(list_y)
            else:
                y = torch.stack(list_y)
        if "rna_targets" in features_dict:
            y = torch.tensor(features_dict["rna_targets"])

        node_map = {nid: idx for idx, nid in enumerate(sorted(graph.nodes()))}
        edge_index = [[node_map[u], node_map[v]] for u, v in sorted(graph.edges())]
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        return Data(x=x, y=y, edge_index=edge_index)

    @property
    def name(self):
        return "sequence"

    def batch(self, samples):
        """
        Batch a list of graph samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        if self.framework == "pyg":
            from torch_geometric.data import Batch

            batch = Batch.from_data_list(samples)
            # sometimes batching changes dtype from int to float32?
            batch.edge_index = batch.edge_index.to(torch.int64)
            batch.edge_attr = batch.edge_attr.to(torch.int64)
            return batch
