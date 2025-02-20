import os
import itertools
import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .distance_computer import DistanceComputer
from rnaglib.utils import (
    US_align_wrapper,
    rna_align_wrapper,
    clean_mmcif,
)
from rnaglib.utils.misc import filter_cif_with_res
from rnaglib.utils.graph_io import get_default_download_dir


class StructureDistanceComputer(DistanceComputer):
    def __init__(
        self,
        name: str = "USalign",
        use_substructures: bool = True,
        structures_path: Path = None,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.name = name
        self.use_substructures = use_substructures
        self.structures_path = structures_path
        self.n_jobs = n_jobs
        super().__init__(name=self.name, **kwargs)

    def forward(self, dataset):
        """Computes pairwise structural similarity between all pairs of RNAs with rna-align.
        Stalls with RNAs > 200 nts.

        :param dataset: RNA dataset to compute similarity over.
        :returns np.array: Array of pairwise similarities in order of given dataset.
        """
        if self.name not in ["RNAalign", "USalign"]:
            raise ValueError("name must be 'RNAalign' or 'USalign'")

        # set default structures dir if a specific directory wasn't specified by the user
        if self.structures_path is None:
            dirname = get_default_download_dir()
            self.structures_path = os.path.join(dirname, "structures")

        with tempfile.TemporaryDirectory() as tmpdir:
            print("dumping structures...")
            # tmpdir = 'debug_persistent'
            os.makedirs(tmpdir, exist_ok=True)
            all_pdb_path = []
            for idx, rna in tqdm(enumerate(dataset), total=len(dataset)):
                rna_graph = rna["rna"]
                cif_path = Path(self.structures_path) / f"{rna_graph.graph['pdbid'].lower()}.cif"
                if self.use_substructures:
                    reslist = [(n.split(".")[1], int(n.split(".")[2])) for n in rna["rna"].nodes()]
                    new_cif = os.path.join(tmpdir, f"{rna_graph.name}.cif")
                    filter_cif_with_res(cif_path, reslist, new_cif)
                    all_pdb_path.append(new_cif)
                else:
                    clean_path = Path(tmpdir) / f"{rna_graph.name}.cif"
                    clean_mmcif(cif_path, clean_path)
                    all_pdb_path.append(clean_path)
            todo = list(itertools.combinations(all_pdb_path, 2))
            if self.name == "USalign":
                sims = Parallel(n_jobs=self.n_jobs)(
                    delayed(US_align_wrapper)(pdbid1, pdbid2)
                    for pdbid1, pdbid2 in tqdm(todo, total=len(todo), desc="USalign")
                )
            elif self.name == "RNAalign":
                sims = Parallel(n_jobs=self.n_jobs)(
                    delayed(rna_align_wrapper)(pdbid1, pdbid2)
                    for pdbid1, pdbid2 in tqdm(todo, total=len(todo), desc="RNAalign")
                )
        sim_mat = np.zeros((len(all_pdb_path), len(all_pdb_path)))
        sim_mat[np.triu_indices(len(all_pdb_path), 1)] = sims
        sim_mat += sim_mat.T
        np.fill_diagonal(sim_mat, 1)

        row_nan_count = np.isnan(sim_mat).sum(axis=1)
        # find rnas that failed against all others
        keep_idx = np.where(row_nan_count != sim_mat.shape[0] - 1)[0]
        sim_mat = sim_mat[keep_idx][:, keep_idx]

        keep_dataset_names = [rna["rna"].name for i, rna in enumerate(dataset) if i in keep_idx]
        return sim_mat, keep_dataset_names
