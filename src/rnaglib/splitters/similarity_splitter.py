import os
import random
from typing import Optional, Union, Callable, Literal
from collections import defaultdict
from pathlib import Path
from joblib import Parallel, delayed
import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from rnaglib.splitters import Splitter
from rnaglib.data_loading import RNADataset
from rnaglib.utils import rna_align_wrapper, cdhit_wrapper, get_sequences

class ClusterSplitter(Splitter):
    """ Abstract class for splitting by clustering with a
    similarity function
    """
    def __init__(self, 
                 similarity_threshold: float = 0.3,
                 n_jobs : int = -1,
                 seed : int = 0,
                 *args, **kwargs):
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        super().__init__(**kwargs)
        pass

    def __call__(self, dataset):
        train, test = self.cluster_split(dataset, self.split_test, n=0.2)
        val, test = self.cluster_split([dataset[i] for i in test], self.split_test, ids=test, n=0.2)
        return train, val, test


    def cluster_split(self, dataset, frac, n=0.05, ids=None):
        """ Fast cluster-based splitting adapted from ProteinShake (https://github.com/BorgwardtLab/proteinshake_release/blob/main/structure_split.py).
            """
        similarity_matrix = self.compute_similarity_matrix(dataset)
        nei = NearestNeighbors(radius=1-self.similarity_threshold, metric='precomputed').fit(1 - similarity_matrix)
        neighbors = nei.radius_neighbors(return_distance=False)

        test_size = int(len(dataset) * frac)
        random.seed(self.seed)
        test = set()
        n = max(1, int(test_size*n))
        pool = list(range(len(dataset)))
        with tqdm(total=test_size, desc='Sampling split') as pbar:
            while len(test) < test_size:
                query = random.choice(pool)
                cluster = set(neighbors[query])
                pool = list(set(pool) - cluster)
                # if cluster is too big, subsample it
                if len(cluster) > n: cluster = random.sample(cluster, n)
                if len(cluster) > (test_size-len(test)): cluster = random.sample(cluster, (test_size-len(test)))
                test.update(cluster)
                pbar.update(len(cluster))
        pool = sorted(list(pool))
        test = sorted(list(test))
        if ids is None:
            return pool, test
        else:
            return [ids[i] for i in pool], [ids[i] for i in test]


    pass

class CDHitSplitter(ClusterSplitter):
    """ Splits based on sequence similarity using CDHit.
    NOTE: Make sure cd-hit is in your PATH.
    """
    
    def compute_similarity_matrix(self, dataset : RNADataset):
        """ Computes sequence similarity between all pairs
        of RNAs. To deal with multi-chain RNAs we clusterb all chains independently
        using CD-Hit. For a given pair of multi-chained RNAs, their overall similarity score
        is given by the Tanimoto coefficient of the sets of clusters assigned to each of the RNA's chains.

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        # prepare input for CD-Hit. One entry per chain.
        ids, sequences = [], []
        for idx, rna in enumerate(dataset):
            seqs = get_sequences(rna['rna'])
            ids.extend([f"{idx}-{s[0].replace('.', '-')}" for s in seqs.items()])
            sequences.extend([s[1] for s in seqs.items()])

        ids_to_cluster, cluster_to_ids = cdhit_wrapper(ids,
                                                       sequences,
                                                       sim_thresh=self.similarity_threshold
                                                       )

        idx_to_clusters = defaultdict(set)
        for seq_id, cluster_id in ids_to_cluster.items():
            idx = seq_id.split("-")[0]
            idx_to_clusters[int(idx)].add(cluster_id)

        def tanimoto(set_1, set_2):
            return len(set_1 & set_2) / len(set_1 | set_2)

        idxs = list(range(len(dataset)))
        sims = [tanimoto(idx_to_clusters[rna_1], idx_to_clusters[rna_2]) for \
                                                                        rna_1, rna_2 in\
                                                                        itertools.combinations(idxs, 2)
                ]
        sim_mat = np.zeros((len(dataset), len(dataset)))
        sim_mat[np.triu_indices(len(dataset), 1)] = sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)
        return sim_mat


class RNAalignSplitter(ClusterSplitter):
    """ Splits based on structural similarity using RNAalign.
    NOTE: running this splitter requires that you have the
    RNAalign executable in your PATH. You can install it by 
    following these instructions: https://zhanggroup.org/RNA-align/download.html."""
    def __init__(self,
                 structures_dir: Union[str, os.PathLike],
                 seed: int = 0,
                 *args,
                 **kwargs
                ):

        self.structures_dir = structures_dir
        super().__init__(**kwargs)

    def compute_similarity_matrix(self, dataset : RNADataset):
        """ Computes pairwise structural similarity between all pairs
        of RNAs using rna-align. Stalls with very large RNA (> 200 nts). Currently only supports whole RNA alignments. For subgraph tasks, will need to edit the corresponding cif file.

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        pdbids = [rna['rna'].graph['pdbid'][0] for rna in dataset]
        pdb_paths = (Path(self.structures_dir) / f"{pdbid.lower()}.cif" for pdbid in pdbids)
        todo = list(itertools.combinations(pdb_paths, 2))
        sims = Parallel(n_jobs=self.n_jobs)(delayed(rna_align_wrapper)(pdbid1, pdbid2)
                                                    for pdbid1, pdbid2 in tqdm(todo, total=len(todo)))
        sim_mat = np.zeros((len(pdbids), len(pdbids)))
        sim_mat[np.triu_indices(len(pdbids), 1)] = sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)
        return sim_mat
        pass
    pass
