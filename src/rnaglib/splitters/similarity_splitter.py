import os
import random
from typing import Optional, Union, Callable, Literal
from pathlib import Path
from joblib import Parallel, delayed
import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from rnaglib.splitters import Splitter
from rnaglib.data_loading import RNADataset
from rnaglib.utils import rna_align_wrapper, cdhit_wrapper

class CDHitSplitter(Splitter):
    """ Splits based on sequence similarity using CDHit.
    NOTE: Make sure cd-hit is in your PATH.
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, dataset):
        pass

    def cluster_split(self, dataset, frac, n=0.05, ids=None):
        """ Fast cluster-based splitting adapted from ProteinShake (https://github.com/BorgwardtLab/proteinshake_release/blob/main/structure_split.py).
        """
        test_size = int(len(dataset) * frac)
        random.seed(self.seed)
        test = set()
        print(test_size, n)
        n = max(1, int(test_size*n))
        print(f"Got test frac {frac}, test set of size {test_size} from {len(dataset)} samples. Max cluster size {n}")
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

    
class RNAalignSplitter(Splitter):
    """ Splits based on structural similarity using RNAalign.
    NOTE: running this splitter requires that you have the
    RNAalign executable in your PATH. You can install it by 
    following these instructions: https://zhanggroup.org/RNA-align/download.html."""
    def __init__(self,
                 structures_dir: Union[str, os.PathLike],
                 similarity_threshold: float = 0.3,
                 n_jobs: Optional[int] = 4,
                 seed: int = 0,
                 *args,
                 **kwargs
                ):

        self.seed = seed
        self.similarity_threshold = similarity_threshold
        self.structures_dir = structures_dir
        self.n_jobs = n_jobs

        super().__init__(**kwargs)

    def __call__(self, dataset):
        train, test = self.cluster_split(dataset, self.split_test, n=0.2)
        val, test = self.cluster_split([dataset[i] for i in test], self.split_test, ids=test, n=0.2)
        return train, val, test

    def compute_similarity_matrix(self, dataset : RNADataset):
        """ Computes pairwise structural similarity between all pairs
        of RNAs

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

    def cluster_split(self, dataset, frac, n=0.05, ids=None):
        """ Fast cluster-based splitting adapted from ProteinShake (https://github.com/BorgwardtLab/proteinshake_release/blob/main/structure_split.py).
        """
        similarity_matrix = self.compute_similarity_matrix(dataset)
        print(f"mean similarity: {similarity_matrix.mean()}")
        nei = NearestNeighbors(radius=1-self.similarity_threshold, metric='precomputed').fit(1 - similarity_matrix)
        neighbors = nei.radius_neighbors(return_distance=False)

        test_size = int(len(dataset) * frac)
        random.seed(self.seed)
        test = set()
        print(test_size, n)
        n = max(1, int(test_size*n))
        print(f"Got test frac {frac}, test set of size {test_size} from {len(dataset)} samples. Max cluster size {n}")
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



