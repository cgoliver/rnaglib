from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import connected_components


class PrepareDataset:
    """
    Transformation to apply to an RNADataset to perform redundancy removal for RNA_Ligand task. The redundancy removal process is quite different for RNA-Ligand than for other
    tasks. Indeed, whereas for other tasks, redundancy removal consists in keeping the RNA/RNA chunk with the highest resolution in each similarity cluster, for RNA-Ligand, it
    consists in keeping the RNA/RNA chunk with the highest resolution and excludimg the clusters containing binding pockets binding different ligands.

    :param str distance_name: name of the distance to use to perform redundancy removal (either "cd_hit" or "us_align")
    :param float threshold: threshold of the clustering (maximal similarity (as defined by distance_name) between two biding pockets we would like to allow in the final dataset)
    """
    def __init__(
        self,
        distance_name: str = "cd_hit",
        threshold: float = 0.90,
    ):
        self.distance_name = distance_name
        self.threshold = threshold

    def __call__(self, dataset):
        """
        Applies the redundancy removal transformation to a dataset.

        :param RNADataset dataset: the initial dataset (without redundancy removal)
        :return: the dataset after redundancy removal
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

        cluster_ligand_map = defaultdict(set)
        for i, cluster in enumerate(neighbors):
            for rna_idx in cluster:
                cluster_ligand_map[i].add(dataset[rna_idx]['rna'].graph['ligand'])

        # dictionary indicating the ligands binding to pockets of multi-ligand clusters binding with at least one top ligand
        multiligand_clusters_mapping = {cluster_id:cluster_ligand_map[cluster_id] for cluster_id in cluster_ligand_map if len(cluster_ligand_map[cluster_id])>1}

        final_list_ids = []
        
        for i, neighborhood in enumerate(neighbors):
            # eliminate clusters containing binding pockets binding different ligands
            if not i in multiligand_clusters_mapping:
                highest_resolution = 100
                highest_resolution_idx = neighborhood[0]
                for rna_idx in neighborhood:
                    rna_dict = dataset[rna_idx]
                    # if resolution is well specified for the considered binding pocket
                    try:
                        resolution = rna_dict['rna'].graph['resolution_high']
                        if resolution < highest_resolution:
                            highest_resolution = resolution
                            highest_resolution_idx = rna_idx
                    except:
                        continue
                # within a given cluster, keep only the pocket with the highest resolution
                final_list_ids.append(highest_resolution_idx)
        dataset = dataset.subset(list_of_ids=final_list_ids)
        return dataset