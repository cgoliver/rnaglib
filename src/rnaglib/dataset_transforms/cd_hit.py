import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from rnaglib.algorithms import get_sequences
from .distance_computer import DistanceComputer
from rnaglib.utils import cdhit_wrapper


class CDHitComputer(DistanceComputer):
    def __init__(
            self,
            similarity_threshold: float = 0.9,
            **kwargs,
    ):
        self.similarity_threshold = similarity_threshold
        super().__init__(name="cd_hit", **kwargs)

    def forward(self, dataset) -> tuple[np.array, list]:
        """Computes sequence similarity between all pairs of RNAs.

        To deal with multi-chain RNAs we cluster all chains independently
        using CD-Hit. For a given pair of multi-chained RNAs, their overall similarity score
        is given by the Tanimoto coefficient of the sets of clusters assigned to each of the RNA's chains.

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        # prepare input for CD-Hit. One entry per consecutive chunk in sequence.
        ids, sequences = [], []
        for idx, rna in enumerate(dataset):
            # Each chunk get a unique ID, starting with the "idx" of the corresponding RNA
            seqs = get_sequences(rna["rna"], longest_only=False, min_size_return=5, verbose=False)
            ids.extend([f"{idx}-{seq_id.replace('.', '-')}" for seq_id, (seq, _) in seqs.items()], )
            sequences.extend([seq for _, (seq, _) in seqs.items()])

        # For each chunk, get its cluster affectation
        ids_to_cluster, cluster_to_ids = cdhit_wrapper(
            ids,
            sequences,
            sim_thresh=self.similarity_threshold,
        )

        # Group together chunks coming from one RNA
        # TODO: this should be a Counter
        idx_to_clusters = defaultdict(set)
        idxs = set()
        for seq_id, cluster_id in ids_to_cluster.items():
            idx = seq_id.split("-")[0]
            idxs.add(int(idx))
            idx_to_clusters[int(idx)].add(cluster_id)
        idxs = sorted(idxs)

        # Compute an RNA-level pairwise distance by the clusters its chunks belong to
        def tanimoto(set_1, set_2):
            return len(set_1 & set_2) / len(set_1 | set_2)

        def custom_tanimoto(set_1, set_2):
            return len(set_1 & set_2) / min(len(set_1), len(set_2))

        todo = list(itertools.combinations(idxs, 2))
        sims = [
            tanimoto(idx_to_clusters[rna_1], idx_to_clusters[rna_2])
            for rna_1, rna_2 in tqdm(todo, desc="CD-Hit", total=len(todo))
        ]
        sim_mat = np.zeros((len(idxs), len(idxs)))
        sim_mat[np.triu_indices(len(idxs), 1)] = sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)

        keep_dataset_names = [dataset.all_rnas.inv[i] for i in idxs] if len(idxs) != len(dataset) \
            else list(dataset.all_rnas)
        return sim_mat, keep_dataset_names
