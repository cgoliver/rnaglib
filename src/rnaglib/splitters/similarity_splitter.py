"""Various splitters taking similarity into account for RNA tasks."""

import itertools
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import reduce
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from rnaglib.algorithms import get_sequences
from rnaglib.data_loading.rna_dataset import RNADataset
from rnaglib.splitters import Splitter
from rnaglib.splitters.linear_optimisation import assign_clusters
from rnaglib.splitters.splitting_utils import label_counter
from rnaglib.utils import (
    US_align_wrapper,
    cdhit_wrapper,
    cif_remove_residues,
    clean_mmcif,
    rna_align_wrapper,
    split_mmcif_by_chain,
)


class ClusterSplitter(Splitter):
    """Abstract class for splitting by clustering with a similarity function."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        n_jobs: int = -1,
        seed: int = 0,
        balanced: bool = True,
        *args,
        **kwargs,
    ):
        self.similarity_threshold = similarity_threshold
        self.n_jobs = n_jobs
        self.seed = seed
        self.balanced = balanced
        super().__init__(**kwargs)

    def forward(self, dataset):
        # overall_test_proportion = self.split_test + self.split_valid
        # relative_test_proportion = self.split_test / overall_test_proportion
        # if not self.balanced:
        #     train, test = self.cluster_split(dataset, overall_test_proportion, n=0.2)
        #     val, test = self.cluster_split(test, relative_test_proportion, n=0.2)
        # else:
        clusters, keep_dataset = self.cluster_split(dataset, frac=0, split=False)
        _, label_counts = label_counter(dataset)
        print(f"dataset:{dataset}")

        named_clusters = []
        for cluster in clusters:
            named_clusters.append([keep_dataset[i]["rna"].name for i in cluster])
        print(f"names:{named_clusters}")
        print(f"clusters: {clusters}")
        train, val, test = self.balancer(
            named_clusters,
            label_counts,
            keep_dataset,
            (self.split_train, self.split_valid, self.split_test),
        )
        return train, val, test

    def balancer(self, clusters, label_counts, dataset, fracs, n=0.2):
        """Splits clusters into train, val, test keeping into account label balance.

        Fracs is a tuple of fractions to get the right proportions.
        Dataset needs to be passed since the cluster indices apply to keep_dataset,
        not necessarily the original one.
        """
        # Here we need to choose from clusters keeping labels in account.
        # Like Plinder, we should (potentially) make sure that singleton
        # clusters don't go into test in a second step.

        # First, we need to know what the label balance is
        labelcounts = []
        for cluster in clusters:
            # Summing all the label counts from each element of the cluster
            print(f"cluster:{cluster}")

            labelcount = sum([label_counts[i] for i in cluster], Counter())
            print(f"labelcount:{labelcount}")
            labelcounts.append(labelcount)

        overall_counts = reduce(lambda x, y: x + y, labelcounts)
        print(f"overall_counts:{overall_counts}")

        print(f"balanced:{self.balanced}")
        train, val, test, metrics = assign_clusters(
            clusters,
            labelcounts,
            split_ratios=fracs,
            label_weight=int(self.balanced),
        )

        print(f"metrics:{metrics}")
        return (
            [dataset[x] for x in range(len(dataset)) if dataset[x]["rna"].name in sum(train, [])],
            [dataset[x] for x in range(len(dataset)) if dataset[x]["rna"].name in sum(val, [])],
            [dataset[x] for x in range(len(dataset)) if dataset[x]["rna"].name in sum(test, [])],
        )

    def compute_similarity_matrix(self, dataset: RNADataset) -> tuple[np.array, list]:
        """Must be implemented by splitter inherting from ClusterSplitter."""
        raise NotImplementedError

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
        print("Computing similarity matrix...")
        similarity_matrix, keep_dataset = self.compute_similarity_matrix(dataset)
        print("Clustering...")  # DEBUG: can confirm sim matrix is symmetric
        adjacency_matrix = (similarity_matrix >= self.similarity_threshold).astype(int)
        n_components, labels = connected_components(adjacency_matrix)

        neighbors = []
        for i in range(n_components):
            neighborhood = np.where(labels == i)[0].tolist()
            neighbors.append(neighborhood)

        print(f"We have {len(neighbors)} clusters.")
        return neighbors, keep_dataset


class CDHitSplitter(ClusterSplitter):
    """Splits based on sequence similarity using CDHit.
    NOTE: Make sure cd-hit is in your PATH.
    """

    def compute_similarity_matrix(self, dataset: RNADataset) -> tuple[np.array, list]:
        """Computes sequence similarity between all pairs
        of RNAs. To deal with multi-chain RNAs we cluster all chains independently
        using CD-Hit. For a given pair of multi-chained RNAs, their overall similarity score
        is given by the Tanimoto coefficient of the sets of clusters assigned to each of the RNA's chains.

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        # prepare input for CD-Hit. One entry per chain.
        ids, sequences = [], []
        for idx, rna in enumerate(dataset):
            seqs = get_sequences(rna["rna"])
            ids.extend(
                [f"{idx}-{seq_id.replace('.', '-')}" for seq_id, (seq, _) in seqs.items()],
            )
            sequences.extend([seq for _, (seq, _) in seqs.items()])

        ids_to_cluster, cluster_to_ids = cdhit_wrapper(
            ids,
            sequences,
            sim_thresh=self.similarity_threshold,
        )

        idx_to_clusters = defaultdict(set)
        idxs = set()
        for seq_id, cluster_id in ids_to_cluster.items():
            idx = seq_id.split("-")[0]
            idxs.add(int(idx))
            idx_to_clusters[int(idx)].add(cluster_id)
        idxs = sorted(idxs)

        if len(idxs) != len(dataset):
            keep_dataset = [rna for i, rna in enumerate(dataset) if i in idxs]
        else:
            keep_dataset = dataset

        def tanimoto(set_1, set_2):
            return len(set_1 & set_2) / len(set_1 | set_2)

        sims = [
            tanimoto(idx_to_clusters[rna_1], idx_to_clusters[rna_2])
            for rna_1, rna_2 in tqdm(itertools.combinations(idxs, 2), desc="CD-Hit")
        ]
        sim_mat = np.zeros((len(idxs), len(idxs)))
        sim_mat[np.triu_indices(len(idxs), 1)] = sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)
        return sim_mat, keep_dataset


class RNAalignSplitter(ClusterSplitter):
    """Splits based on structural similarity using RNAalign.

    NOTE: running this splitter requires that you have the
    RNAalign executable in your PATH. You can install it by
    following these instructions: https://zhanggroup.org/RNA-align/download.html.
    """

    def __init__(
        self,
        structures_dir: str | os.PathLike,
        seed: int = 0,
        use_substructures: bool = True,
        *args,
        **kwargs,
    ):
        """Use RNAalign to split structures.

        :param structures_dir: path to folder containing mmCIF files for all elements in dataset.
        :param seed: random seed for reproducibility.
        :param use_substructures: if True only uses residues in the dataset item's graph.nodes().
            Useful for pocket tasks. Otherwise uses the full mmCIF file from the PDB.

        """
        self.use_substructures = use_substructures
        self.structures_dir = structures_dir
        super().__init__(**kwargs)

    def compute_similarity_matrix(self, dataset: RNADataset):
        """Computes pairwise structural similarity between all pairs of RNAs with rna-align. Stalls with RNAs > 200 nts.

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        pdbids = [rna["rna"].graph["pdbid"] for rna in dataset]
        pdb_paths = (Path(self.structures_dir) / f"{pdbid.lower()}.cif" for pdbid in pdbids)

        with tempfile.TemporaryDirectory() as tmpdir:
            if self.use_substructures:
                reslists = [[(n.split(".")[1], n.split(".")[2]) for n in rna["rna"].nodes()] for rna in dataset]
                new_paths = []
                for idx, (cif_path, reslist) in enumerate(zip(pdb_paths, reslists, strict=False)):
                    new_cif = Path(tmpdir) / f"{idx}.cif"
                    cif_remove_residues(cif_path, reslist, new_cif)
                    new_paths.append(new_cif)
                pdb_paths = new_paths

            pdb_paths_clean = []
            for pdb_path in pdb_paths:
                clean_path = Path(tmpdir) / Path(pdb_path).name
                clean_mmcif(pdb_path, clean_path)
                pdb_paths_clean.append(clean_path)
            todo = list(itertools.combinations(pdb_paths_clean, 2))
            sims = Parallel(n_jobs=self.n_jobs)(
                delayed(US_align_wrapper)(pdbid1, pdbid2)
                for pdbid1, pdbid2 in tqdm(todo, total=len(todo), desc="RNAalign")
            )
        sim_mat = np.zeros((len(pdb_paths_clean), len(pdb_paths_clean)))
        sim_mat[np.triu_indices(len(pdb_paths_clean), 1)] = sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)

        row_nan_count = np.isnan(sim_mat).sum(axis=1)
        # find rnas that failed against all others
        keep_idx = np.where(row_nan_count != sim_mat.shape[0] - 1)[0]
        sim_mat = sim_mat[keep_idx][:, keep_idx]

        keep_dataset = [rna for i, rna in enumerate(dataset) if i in keep_idx]
        return sim_mat, keep_dataset

    def compute_similarity_matrix_(self, dataset: RNADataset):
        """Computes pairwise structural similarity between all pairs
        of RNAs using rna-align. Stalls with larger RNA (> 200 nts).

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        pdbids = [rna["rna"].graph["pdbid"] for rna in dataset]
        pdb_paths = (Path(self.structures_dir) / f"{pdbid.lower()}.cif" for pdbid in pdbids)

        with tempfile.TemporaryDirectory() as tmpdir:
            if self.use_substructures:
                reslists = [[(n.split(".")[1], n.split(".")[2]) for n in rna["rna"].nodes()] for rna in dataset]
                new_paths = []
                for idx, (cif_path, reslist) in enumerate(zip(pdb_paths, reslists, strict=False)):
                    new_cif = Path(tmpdir) / f"{idx}.cif"
                    cif_remove_residues(cif_path, reslist, new_cif)
                    new_paths.append(new_cif)
                pdb_paths = new_paths

        def rna_align_wrapper_(a, b):
            return 1

        # first sim mat is all chains vs all chains
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_paths = []
            pdbid_to_idxlist = {}
            pdbids_keep = []
            for i, pdb in enumerate(pdb_paths):
                chain_pdbs = split_mmcif_by_chain(
                    pdb,
                    output_dir=tmpdir,
                    min_length=5,
                    max_length=1000,
                )
                pdbid_to_idxlist[Path(pdb).stem.lower()] = list(
                    range(i, i + len(chain_pdbs)),
                )
                chain_paths.extend(chain_pdbs)
                if len(chain_pdbs) > 1:
                    pdbids_keep.append(Path(pdb).stem.lower())

            todo = list(itertools.combinations(chain_paths, 2))
            sims_chain = Parallel(n_jobs=self.n_jobs)(
                delayed(rna_align_wrapper)(pdbid1, pdbid2)
                for pdbid1, pdbid2 in tqdm(todo, total=len(todo), desc="RNAalign")
            )

            sim_mat_chain = np.zeros((len(chain_paths), len(chain_paths)))
            sim_mat_chain[np.triu_indices(len(chain_paths), 1)] = sims_chain
            sim_mat_chain += sim_mat_chain.T
            np.fill_diagonal(sim_mat_chain, 1)

        # coalesce chain-wise sim-mat by taking a hungarian between chains of two
        # PDBs being compared
        final_sims = []
        for pdbid1, pdbid2 in itertools.combinations(pdbids_keep, 2):
            rows = pdbid_to_idxlist[pdbid1.lower()]
            cols = pdbid_to_idxlist[pdbid2.lower()]
            cost_matrix = 1 - sim_mat_chain[np.ix_(rows, cols)]
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_indices, col_indices].sum()
            # put back to similarity
            final_sims.append(1 - total_cost)

        # final sim mat is coalesced chain-wise sim mat.
        sim_mat = np.zeros((len(pdbids_keep), len(pdbids_keep)))
        sim_mat[np.triu_indices(len(pdbids_keep), 1)] = final_sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)

        row_nan_count = np.isnan(sim_mat).sum(axis=1)
        # find rnas that failed against all others
        keep_idx = np.where(row_nan_count != sim_mat.shape[0] - 1)[0]
        final_pdbids = [pdbids_keep[i] for i in keep_idx]
        sim_mat = sim_mat[keep_idx][:, keep_idx]

        keep_dataset = [rna for i, rna in enumerate(dataset) if rna["rna"].graph["pdbid"].lower() in final_pdbids]
        return sim_mat, keep_dataset
