import numpy as np
from scipy.sparse.csgraph import connected_components

from rnaglib.data_loading import RNADataset

class DistanceComputer:
    def __init__(self, name):
        self.name = name
        pass

    def __call__(self, dataset):
        if self.name not in dataset.distances:
            similarity_matrix, keep_dataset_names = self.forward(dataset)
            if len(keep_dataset_names) != len(dataset):
                print(
                    f"""WARNING: Similarity matrix computer dropped some data points.
                        Original dataset had: {len(dataset)}
                        New dataset has: {len(keep_dataset_names)}\n"""
                )
            # saving the distance matrices to the object in case we want to use them later
            dataset = dataset.subset(list_of_names=keep_dataset_names)
            dataset.distances[self.name] = 1-similarity_matrix
        return dataset

    def forward(self, dataset):
        raise NotImplementedError