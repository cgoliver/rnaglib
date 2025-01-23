import numpy as np
from scipy.sparse.csgraph import connected_components

from rnaglib.data_loading import RNADataset

class RedundancyRemover:
    def __init__(
        self,
        distance_name: str = "rna_align",
        threshold: float = 0.95,
    ):
        self.distance_name = distance_name
        self.threshold = threshold
        pass

    def __call__(self, dataset):
        if not self.distance_name in dataset.distances:
            raise ValueError(f"The distance matrix using distances {self.distance_name} has not been computed")
        
        adjacency_matrix = (dataset.distances[self.distance_name] <= 1-self.threshold).astype(int)
        n_components, labels = connected_components(adjacency_matrix)

        neighbors = []
        for i in range(n_components):
            neighborhood = np.where(labels == i)[0].tolist()
            neighbors.append(neighborhood)

        final_list_ids = []
        for neighborhood in neighbors:
            highest_resolution = 100
            highest_resolution_idx = neighborhood[0]
            for rna_idx in neighborhood:
                rna_dict = dataset[rna_idx]
                try:
                    resolution = rna_dict['rna'].graph['resolution_high']
                    if resolution<highest_resolution:
                        highest_resolution = resolution
                        highest_resolution_idx = rna_idx
                except:
                    continue
            final_list_ids.append(highest_resolution_idx)
        dataset = dataset.subset(list_of_ids=final_list_ids)

        return dataset