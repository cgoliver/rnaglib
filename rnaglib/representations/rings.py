import torch
import numpy as np

from rnaglib.representations import Representation


class RingRepresentation(Representation):
    """ Converts and RNA into a voxel based representation """

    def __init__(self, level='graphlet_annots', **kwargs):
        super().__init__(**kwargs)
        self.level = level

    def __call__(self, rna_graph, features_dict):
        ring = list(sorted(rna_graph.nodes(data=self.level)))
        return ring

    @property
    def name(self):
        return "ring"

    def batch(self, samples):
        """
        Batch a list of voxel samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        raise NotImplementedError
