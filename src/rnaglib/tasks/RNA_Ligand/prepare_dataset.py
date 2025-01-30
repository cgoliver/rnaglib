from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import connected_components


class PrepareDataset:
    def __init__(
        self,
        distance_name: str = "cd_hit",
        threshold: float = 0.90,
    ):
        self.distance_name = distance_name
        self.threshold = threshold

    def __call__(self, dataset):
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
            # eliminate clusters containing binding pockets binding to different ligands
            if not i in multiligand_clusters_mapping:
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