import os
import itertools
import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from rnaglib.data_loading.rna_dataset import RNADataset
from rnaglib.data_loading import RNADataset
from rnaglib.distance_computer import DistanceComputer
from rnaglib.utils import (
    US_align_wrapper,
    cif_remove_residues,
    clean_mmcif,
)


class RNAalignComputer(DistanceComputer):
    def __init__(
            self,
            use_substructures: bool = True,
            structures_path: Path = os.path.join(os.path.expanduser("~"), ".rnaglib/structures"),
            n_jobs: int = -1,
            **kwargs,
    ):
        self.use_substructures = use_substructures
        self.structures_path = structures_path
        self.n_jobs = n_jobs
        super().__init__(name="rna_align", **kwargs)

    def forward(self, dataset: RNADataset):
        """Computes pairwise structural similarity between all pairs of RNAs with rna-align. Stalls with RNAs > 200 nts.

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        pdbids = [rna["rna"].graph["pdbid"] for rna in dataset]
        pdb_paths = (Path(self.structures_path) / f"{pdbid.lower()}.cif" for pdbid in pdbids)

        with tempfile.TemporaryDirectory() as tmpdir:
            if self.use_substructures:
                reslists = [[(n.split(".")[1], int(n.split(".")[2])) for n in rna["rna"].nodes()] for rna in dataset]
                new_paths = []
                for idx, (cif_path, reslist) in enumerate(zip(pdb_paths, reslists, strict=False)):
                    new_cif = Path(tmpdir) / f"{idx}.cif"
                    cif_remove_residues(cif_path, reslist, new_cif)
                    new_paths.append(new_cif)
                pdb_paths = new_paths

            pdb_paths_clean = []

            for pdb_path in tqdm(pdb_paths, desc="Cleaning PDB files"):
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

        keep_dataset_names = [rna["rna"].name for i, rna in enumerate(dataset) if i in keep_idx]
        return sim_mat, keep_dataset_names
