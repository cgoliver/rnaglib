"""Various splitters taking similarity into account for RNA tasks."""

from collections import Counter
from collections.abc import Iterable

import numpy as np
from scipy.sparse.csgraph import connected_components

from rnaglib.dataset_transforms import Splitter
from rnaglib.dataset_transforms.linear_optimisation import assign_clusters
from rnaglib.dataset_transforms.splitting_utils import label_counter


class ClusterSplitter(Splitter):
    """Abstract class for splitting by clustering with a similarity function."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        n_jobs: int = -1,
        seed: int = 0,
        balanced: bool = True,
        distance_name: str = "USalign",
        verbose=False,
        *args,
        **kwargs,
    ):
        self.similarity_threshold = similarity_threshold
        self.n_jobs = n_jobs
        self.seed = seed
        self.balanced = balanced
        self.distance_name = distance_name
        self.verbose = verbose
        super().__init__(**kwargs)

    def forward(self, dataset):
        print(f"pre cluster len: {len(dataset)}")
        clusters = self.cluster_split(dataset, frac=0, split=False)
        _, label_counts = label_counter(dataset)
        # print(f"dataset:{dataset}")
        # print(f"label_counts:{label_counts}")
        named_clusters = []
        for cluster in clusters:
            named_clusters.append([dataset[i]["rna"].name for i in cluster])
        # print(f"names:{named_clusters}")
        # print(f"clusters: {clusters}")
        train, val, test = self.balancer(
            named_clusters,
            label_counts,
            dataset,
            (self.split_train, self.split_valid, self.split_test),
        )
        return train, val, test

    def balancer(self, clusters, label_counts, dataset, fracs, n=0.2):
        """Splits clusters into train, val, test keeping into account label balance.

        Fracs is a tuple of fractions to get the right proportions.
        Dataset needs to be passed since the cluster indices apply to keep_dataset,
        not necessarily the original one.
        """
        print("Computing balanced clusters...")
        # Here we need to choose from clusters keeping labels in account.
        # Like Plinder, we should (potentially) make sure that singleton
        # clusters don't go into test in a second step.

        # First, we need to know what the label balance is
        labelcounts = []
        for cluster in clusters:
            # Summing all the label counts from each element of the cluster
            # print(f"cluster:{cluster}")

            labelcount = sum([label_counts[i] for i in cluster], Counter())
            # print(f"labelcount:{labelcount}")
            labelcounts.append(labelcount)

        # overall_counts = reduce(lambda x, y: x + y, labelcounts)
        # print(f"overall_counts:{overall_counts}")

        # print(f"balanced:{self.balanced}")
        weight = self.balanced if not self.debug else 0
        train, val, test, metrics = assign_clusters(
            clusters, labelcounts, split_ratios=fracs, label_weight=int(weight), verbose=self.verbose
        )
        print("Done.")

        # print(f"metrics:{metrics}")
        return (
            [x for x in range(len(dataset)) if dataset[x]["rna"].name in sum(train, [])],
            [x for x in range(len(dataset)) if dataset[x]["rna"].name in sum(val, [])],
            [x for x in range(len(dataset)) if dataset[x]["rna"].name in sum(test, [])],
        )

    def cluster_split(
        self,
        dataset: Iterable,
        frac: float,
        n: float = 0.05,
        split: bool = True,
    ):
        """Fast cluster-based splitting adapted from ProteinShake.

        (https://github.com/BorgwardtLab/proteinshake_release/blob/main/structure_split.py).
        Splits the dataset into two splits, with the guarantee
        that no two points above ``similarity_threshold`` of each other belong to the same split.
        Computes a similarity matrix used to identify redundant clusters based on the ``similarity_threshold``.
        To split the dataset, we iterate over a pool of data points until the desired size of the
        test set is reached. The pool initially consists of the whole dataset.
        At each step, we choose a random point from the pool and fetch all points
        from the pool with similarity above ``similarity_threshold``, we call this the current cluster.
        If the cluster contains more than ``test_size * n`` points, we sub-sample the cluster.
        If the cluster would make the test set larger than ``test_size`` we sub-sample it to the
        difference between the current test set and ``test_size``.
        We then remove the current cluster from the pool and add it to the test set.
        Points that remain in the pool are kept as the training set.

        :param dataset: dataset to split
        :param frac: fraction of dataset to use as the test set
        :param n: portion of the test set size to use as largest test set cluster size
        :param split: if split is False, we return all clusters instead of splitting them
        """
        if dataset.distances is not None:
            if not self.distance_name in dataset.distances:
                raise ValueError(f"The distance matrix using distances {self.distance_name} has not been computed")

        similarity_matrix = 1 - dataset.distances[self.distance_name]
        adjacency_matrix = (similarity_matrix >= self.similarity_threshold).astype(int)
        n_components, labels = connected_components(adjacency_matrix)

        neighbors = []
        for i in range(n_components):
            neighborhood = np.where(labels == i)[0].tolist()
            neighbors.append(neighborhood)

        print(f"We have {len(neighbors)} clusters.")
        return neighbors
