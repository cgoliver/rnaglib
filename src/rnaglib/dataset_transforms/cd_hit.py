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
            similarity_threshold: float = 0.5,
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

        def tanimoto(set_1, set_2):
            return len(set_1 & set_2) / len(set_1 | set_2)
        
        def custom_tanimoto(set_1, set_2):
            return len(set_1 & set_2) / min(len(set_1), len(set_2))

        sims = [
            tanimoto(idx_to_clusters[rna_1], idx_to_clusters[rna_2])
            for rna_1, rna_2 in tqdm(itertools.combinations(idxs, 2), desc="CD-Hit")
        ]
        sim_mat = np.zeros((len(idxs), len(idxs)))
        sim_mat[np.triu_indices(len(idxs), 1)] = sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)

        keep_dataset_names = [rna["rna"].name for i, rna in  enumerate(dataset) if i in idxs] if len(idxs) != len(dataset) else [rna["rna"].name for rna in dataset]

        return sim_mat, keep_dataset_names