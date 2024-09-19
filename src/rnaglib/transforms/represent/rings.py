import random
import torch

from rnaglib.algorithms import k_block_list
from rnaglib.transforms.represent import Representation


class RingRepresentation(Representation):
    """
    Converts RNA into a ring based representation
    """

    def __init__(self, node_simfunc=None, max_size_kernel=None, hash_path=None, **kwargs):
        super().__init__(**kwargs)
        if node_simfunc is None:
            raise ValueError("node_simfunc cannot be None to create a RingRepresentation")
        self.node_simfunc = node_simfunc
        self.max_size_kernel = max_size_kernel
        if node_simfunc.method in ['R_graphlets', 'graphlet', 'R_ged']:
            if hash_path is not None:
                node_simfunc.add_hashtable(hash_path)
            self.level = 'graphlet_annots'
        else:
            self.level = 'edge_annots'

    def __call__(self, rna_graph, features_dict):
        ring = list(sorted(rna_graph.nodes(data=self.level)))
        if ring[0][1] is None:
            raise ValueError(
                f"To use rings, one needs to use annotated data. The key {self.level} is missing from the graph.")
        return ring

    @property
    def name(self):
        return "ring"

    def batch(self, samples):
        """
        Batch a list of ring samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        # we need to flatten the list and then use the kernels :
        # The rings is now a list of lists of tuples
        # If we have a huge graph, we can sample max_size_kernel nodes to avoid huge computations,
        # We then return the sampled ids

        flat_rings = list()
        for ring in samples:
            flat_rings.extend(ring)
        if self.max_size_kernel is None or len(flat_rings) < self.max_size_kernel:
            # Just take them all
            node_ids = [1 for _ in flat_rings]
        else:
            # Take only 'max_size_kernel' elements
            node_ids = [1 for _ in range(self.max_size_kernel)] + \
                       [0 for _ in range(len(flat_rings) - self.max_size_kernel)]
            random.shuffle(node_ids)
            flat_rings = [node for i, node in enumerate(flat_rings) if node_ids[i] == 1]
        k_block = k_block_list(flat_rings, self.node_simfunc)
        return torch.from_numpy(k_block).detach().float(), node_ids
