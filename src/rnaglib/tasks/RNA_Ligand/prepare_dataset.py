import numpy as np
from scipy.sparse.csgraph import connected_components

from rnaglib.dataset_transforms import DSTransform


class PrepareDataset(DSTransform):
    """
    Dataset transform removing redundancy in a dataset by performing clustering on the dataset, then keeping the pocket
    with the highest resolution within each cluster. This is the RNA_Ligand counterpart of RedundancyRemover, from which
    it differs on a single point: a cluster whose pockets do not all bind the same ligand is dropped entirely.

    The members of such a cluster are superimposable sites binding different compounds in different depositions, so
    structure does not determine the ligand there and no representative can be picked without asserting an arbitrary
    answer: which ligand the benchmark would claim binds that site would come down to which deposition happened to be
    better resolved. Dropping the cluster asserts nothing instead. This restricts the task to the sites where the label
    is well defined, and the resulting score should be read as such, since the sites removed here are the hard ones.

    :param str distance_name: the name of the distance metric which has to be used to perform clustering. The distance
    must have been computed on the dataset (see DistanceComputer)
    :param float threshold: the similarity threshold (considering similarity as 1-distance) to use to perform clustering
    :param str target_name: the graph-level attribute holding the ligand a pocket is labelled with (default "ligand")
    """

    def __init__(
        self,
        distance_name: str = "USalign",
        threshold: float = 0.95,
        target_name: str = "ligand",
    ):
        self.distance_name = distance_name
        self.threshold = threshold
        self.target_name = target_name

    def __call__(self, dataset):
        """
        Applies the redundancy removal transformation to a dataset.

        :param RNADataset dataset: the initial dataset (without redundancy removal)
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
            # a cluster gathering pockets labelled with different ligands is ambiguous: keeping any of its members would
            # assert that this site binds that ligand, when a superimposable site is known to bind another one. Drop it
            bound_ligands = set()
            for rna_idx in neighborhood:
                try:
                    bound_ligands.add(dataset[rna_idx]["rna"].graph[self.target_name])
                except Exception:
                    continue
            if len(bound_ligands) > 1:
                continue

            highest_resolution = 100
            highest_resolution_idx = neighborhood[0]
            for rna_idx in neighborhood:
                rna_dict = dataset[rna_idx]
                try:
                    resolution = rna_dict["rna"].graph["resolution_high"]
                    if resolution < highest_resolution:
                        highest_resolution = resolution
                        highest_resolution_idx = rna_idx
                except Exception:
                    continue
            final_list_ids.append(highest_resolution_idx)
        dataset = dataset.subset(list_of_ids=final_list_ids)
        return dataset
