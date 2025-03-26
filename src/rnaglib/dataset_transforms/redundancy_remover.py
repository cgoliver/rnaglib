import numpy as np
from scipy.sparse.csgraph import connected_components

from rnaglib.dataset_transforms import DSTransform


class RedundancyRemover(DSTransform):
    """Dataset transform removing redundancy in a dataset by performing clustering on the dataset then keeping only the RNA with the highest resolution within each cluster

    :param str distance_name: the name of the distance metric which has to be used to perform clustering. The distance must have been computed on the dataset (see DistanceComputer)
    :param float threshold: the similarity threshold (considering similarity as 1-distance) to use to perform clustering
    """
    def __init__(
        self,
        distance_name: str = "USalign",
        threshold: float = 0.95,
    ):
        self.distance_name = distance_name
        self.threshold = threshold

    def __call__(self, dataset):
        """"Removes redundancy to a specific dataset following the parameters specified in the RedundancyRemover object

        :return: the dataset with redundancy removed according to specified criteria
        :rtype: RNADataset
        """
        if dataset.distances is None or not self.distance_name in dataset.distances:
            raise ValueError(f"The distance matrix using distances {self.distance_name} has not been computed")

        adjacency_matrix = (dataset.distances[self.distance_name] <= 1 - self.threshold).astype(int)
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
                    if resolution < highest_resolution:
                        highest_resolution = resolution
                        highest_resolution_idx = rna_idx
                except:
                    continue
            final_list_ids.append(highest_resolution_idx)
        dataset = dataset.subset(list_of_ids=final_list_ids)
        return dataset
