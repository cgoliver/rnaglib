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

        super().__init__(**kwargs)
        pass

    def __call__(self, rna_graph, features_dict):
        sequence_dict = get_sequences(rna_graph)

        full_seq = ""
        full_nid = []
        chain_index = []

        for i, (chain_id, (seq, nids)) in enumerate(sequence_dict.items()):
            full_seq += seq
            full_nid.extend(nids)
            chain_index.extend([i] * len(seq))

        if self.framework == "torch":
            data = self.to_torch(full_nid, features_dict, chain_index)
            return data
        if self.framework == "pyg":
            data = self.to_pyg(full_nid, features_dict, chain_index)
            return data

    def to_torch(self, node_ids, features_dict, chain_index):
        x, y = None, None
        if "nt_features" in features_dict:
            x = (
                torch.stack([features_dict["nt_features"][n] for n in node_ids])
                if "nt_features" in features_dict
                else None
            )
        if "nt_targets" in features_dict:
            list_y = [features_dict["nt_targets"][n] for n in node_ids]
            # In the case of single target, pytorch CE loss expects shape (n,) and not (n,1)
            # For multi-target cases, we stack to get (n,d)
            if len(list_y[0]) == 1:
                y = torch.cat(list_y)
            else:
                y = torch.stack(list_y)
        if "rna_targets" in features_dict:
            y = torch.tensor(features_dict["rna_targets"])

        chain_index = torch.tensor(chain_index, dtype=torch.long)
        return x, chain_index

    def to_pyg(self, node_ids, features_dict, chain_index):
        from torch_geometric.data import Data

        # for some reason from_networkx is not working so doing by hand
        # not super efficient at the moment
        x, y = None, None
        if "nt_features" in features_dict:
            x = (
                torch.stack([features_dict["nt_features"][n] for n in node_ids])
                if "nt_features" in features_dict
                else None
            )
        if "nt_targets" in features_dict:
            list_y = [features_dict["nt_targets"][n] for n in node_ids]
            # In the case of single target, pytorch CE loss expects shape (n,) and not (n,1)
            # For multi-target cases, we stack to get (n,d)
            if len(list_y[0]) == 1:
                y = torch.cat(list_y)
            else:
                y = torch.stack(list_y)
        if "rna_targets" in features_dict:
            y = torch.tensor(features_dict["rna_targets"])

        chain_index = torch.tensor(chain_index, dtype=torch.long)
        return Data(x=x, y=y, chain_index=chain_index)

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
            return batch
