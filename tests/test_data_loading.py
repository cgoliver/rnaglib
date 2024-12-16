import unittest
import os
from pathlib import Path

import networkx as nx

from rnaglib.data_loading import RNADataset
from rnaglib.data_loading import rna_from_pdbid
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import RNAFMTransform
from rnaglib.transforms import GraphRepresentation


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.default_dataset = RNADataset(debug=True)
        pass

    def test_rna_from_pdbid(self):
        rna_from_pdbid("1fmn", redundancy="debug")  # fetch from RCSB
        rna_from_pdbid("1d0t", redundancy="debug")  # local

    def test_in_memory(self):
        d = RNADataset(debug=True, in_memory=True)
        d[0]
        pass

    def test_on_disk(self):
        d = RNADataset(debug=True, in_memory=False)
        d[0]

    def test_get_pdbds(self):
        d = RNADataset(debug=True, get_pdbs=True, overwrite=True)
        pdbids = [rna["rna"].graph["pdbid"] for rna in d]
        pdb_paths = (Path(d.structures_path) / f"{pdbid.lower()}.cif" for pdbid in pdbids)
        for path in pdb_paths:
            assert os.path.exists(path)
        pass

    def test_rna_get(self):
        rna = self.default_dataset[0]
        assert "rna" in rna

    def test_dataset_from_list(self):
        rnas = [nx.Graph(name="rna1"), nx.Graph(name="rna2")]
        da = RNADataset(rnas=rnas)
        assert len(da) == len(rnas)
        pass

    """
    def test_dataset_from_pdbids(self):
        all_rnas = ['2pwt', '5v3f', '379d',
                    '5bjo', '4pqv', '430d',
                    '1nem', '1q8n', '1f1t',
                    '2juk', '4yaz', '364d',
                    '6ez0', '2tob', '1ddy',
                    '1fmn', '2mis', '4f8u'
                    ]

        da = RNADataset(all_rnas=all_rnas, redundancy='all')
        assert len(da) == len(all_rnas)
    """

    def test_add_representation(self):
        self.default_dataset.add_representation(GraphRepresentation())
        pass

    def test_pre_transform(self):
        """Add rnafm embeddings during dataset construction from database,
        then look up the stored attribute at getitem time.
        """
        tr = RNAFMTransform()
        feat = FeaturesComputer(nt_features=["nt_code", tr.name], custom_encoders={tr.name: tr.encoder})
        dataset = RNADataset(
            debug=True,
            features_computer=feat,
            pre_transforms=tr,
            representations=GraphRepresentation(framework="pyg"),
        )

        assert dataset[0]["graph"].x is not None

    def test_post_transform(self):
        """Apply transform during getitem call."""
        tr = RNAFMTransform()
        feat = FeaturesComputer(nt_features=["nt_code", tr.name], custom_encoders={tr.name: tr.encoder})
        dataset = RNADataset(
            debug=True,
            features_computer=feat,
            transforms=tr,
            representations=GraphRepresentation(framework="pyg"),
        )
        assert dataset[0]["graph"].x is not None
        pass


if __name__ == "__main__":
    unittest.main()
