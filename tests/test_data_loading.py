import unittest
import os
from pathlib import Path

import networkx as nx

from rnaglib.dataset import RNADataset
from rnaglib.dataset import rna_from_pdbid
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import RNAFMTransform
from rnaglib.transforms import GraphRepresentation


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.default_dataset = RNADataset(debug=True)

    def test_rna_from_pdbid(self):
        rna_from_pdbid("1fmn", redundancy="debug")  # fetch from RCSB
        rna_from_pdbid("1d0t", redundancy="debug")  # local

    def test_in_memory(self):
        d = RNADataset(debug=True, in_memory=True)
        d[0]

    def test_on_disk(self):
        d = RNADataset(debug=True, in_memory=False)
        d[0]

    def test_get_pdbds(self):
        d = RNADataset(debug=True, get_pdbs=True)
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

    def test_post_transform(self):
        """Apply transform during getitem call."""
        tr = RNAFMTransform(debug=True)
        feat = FeaturesComputer(nt_features=["nt_code", tr.name], custom_encoders={tr.name: tr.encoder})
        dataset = RNADataset(
            debug=True,
            features_computer=feat,
            transforms=tr,
            representations=GraphRepresentation(framework="pyg"),
        )
        assert dataset[0]["graph"].x is not None

    def test_transforms_list_not_dropped(self):
        """Passing a list of transforms should keep all of them (fix: transforms were silently dropped)."""
        from rnaglib.transforms import RfamTransform
        tr1 = RNAFMTransform(debug=True)
        tr2 = RfamTransform()
        dataset = RNADataset(debug=True, transforms=[tr1, tr2])
        assert len(dataset.transforms) == 2

    def test_transforms_single_kept(self):
        """Passing a single transform should wrap it in a list."""
        tr = RNAFMTransform(debug=True)
        dataset = RNADataset(debug=True, transforms=tr)
        assert len(dataset.transforms) == 1

    def test_transforms_none_is_empty(self):
        """Passing None transforms should give an empty list."""
        dataset = RNADataset(debug=True, transforms=None)
        assert dataset.transforms == []

    def test_rna_class_from_dict(self):
        """Test RNA from_dict properly sets graph properties and creates self.rna_dict."""
        from rnaglib.dataset.rna import RNA
        g = nx.Graph(name="test_rna", pdbid="1abc")
        rna_dict = {"rna": g, "other_attr": "value"}
        rna = RNA(rna_dict=rna_dict)
        assert hasattr(rna, "name")
        assert rna.name == "test_rna"
        assert hasattr(rna, "pdbid")
        assert rna.pdbid == "1abc"
        assert getattr(rna, "other_attr") == "value"
        assert rna.to_dict() == rna_dict

    def test_rna_from_pdbid_no_multigraph_error(self):
        """Test RNA from_pdbid runs without TypeError related to multigraph."""
        from rnaglib.dataset.rna import RNA
        # This should fail if it tries to pass multigraph to rna_from_pdbid which doesn't accept it
        try:
            rna = RNA(pdbid="1fmn")
            assert hasattr(rna, "pdbid")
            assert rna.pdbid == "1fmn"
        except TypeError as e:
            self.fail(f"rna_from_pdbid raised TypeError: {e}")

    def test_rnadataset_save_and_extension(self):
        import tempfile
        import os
        from rnaglib.dataset.rna_dataset import RNADataset
        from rnaglib.dataset.rna import RNA
        
        g1 = nx.Graph(name="test_1", pdbid="1abc")
        g2 = nx.Graph(name="test_2", pdbid="2xyz")
        rnas = [g1, g2]
        
        # Test in-memory extension default
        dataset = RNADataset(rnas=rnas)
        self.assertEqual(dataset.extension, ".json")
        
        with tempfile.TemporaryDirectory() as td:
            dataset.save(td)
            
            # Check files were created with .json extension
            dump_files = set(os.listdir(td))
            self.assertIn("test_1.json", dump_files)
            self.assertIn("test_2.json", dump_files)
            
            # Test that RNADataset correctly loads from the dumped dir
            loaded_dataset = RNADataset(dataset_path=td)
            self.assertEqual(len(loaded_dataset), 2)
            self.assertEqual(loaded_dataset.extension, ".json")

if __name__ == "__main__":
    unittest.main()
