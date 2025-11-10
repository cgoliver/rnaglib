import unittest
from unittest import mock
import tempfile
from types import SimpleNamespace
from pathlib import Path
import os
import numpy as np
import networkx as nx

from rnaglib.prepare_data import fr3d_to_graph, chop_all, annotate_all, build_graph_from_cif
from rnaglib.prepare_data.main import prepare_data_main, dir_setup
from rnaglib.prepare_data.chopper import (
    block_pca, pca_chop, chop, graph_filter, graph_clean, chop_one_rna
)
from rnaglib.prepare_data.khop_annotate import node_2_unordered_rings
from rnaglib.prepare_data.retrieve_structures import load_csv
from rnaglib.utils import load_graph, dump_json


class TestPrepareData(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        args = {
            "debug": True,
            "continu": False,
            "n_debug": 10,
            "num_workers": 4,
            "annotate": False,
            "chop": False,
            "one_mmcif": None,
            "tag": "test",
            "rna_source": "rcsb",
            "nr": True,
        }
        self.args = SimpleNamespace(**args)

    def test_fr3d_to_graph(self):
        """Test fr3d_to_graph function."""
        cif_path = "./src/rnaglib/data/1evv.cif"
        if os.path.exists(cif_path):
            graph = fr3d_to_graph(cif_path)
            assert graph is not None
            assert isinstance(graph, nx.Graph)

    def test_database_build(self):
        """Test database build."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.args.structures_dir = Path(tmpdir) / "structures"
            self.args.output_dir = Path(tmpdir) / "build"
            prepare_data_main(self.args)

    def test_dir_setup(self):
        """Test directory setup function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=Path(tmpdir) / "output",
                tag="test_tag",
                annotate=False,
                chop=False
            )
            build_dir = dir_setup(args)
            assert os.path.exists(build_dir)
            assert os.path.exists(os.path.join(build_dir, "graphs"))

    def test_dir_setup_with_annotate(self):
        """Test directory setup with annotation enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=Path(tmpdir) / "output",
                tag="test_tag",
                annotate=True,
                chop=False
            )
            build_dir = dir_setup(args)
            assert os.path.exists(os.path.join(build_dir, "chops"))
            assert os.path.exists(os.path.join(build_dir, "annot"))

    def test_dir_setup_with_chop(self):
        """Test directory setup with chop enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                output_dir=Path(tmpdir) / "output",
                tag="test_tag",
                annotate=False,
                chop=True
            )
            build_dir = dir_setup(args)
            assert os.path.exists(os.path.join(build_dir, "chops"))

    def test_build_graph_from_cif(self):
        """Test build_graph_from_cif function."""
        cif_path = "./src/rnaglib/data/1evv.cif"
        if os.path.exists(cif_path):
            with tempfile.TemporaryDirectory() as tmpdir:
                dump_dir = Path(tmpdir)
                result = build_graph_from_cif(cif_path, dump_dir=dump_dir)
                assert result is not None
                assert os.path.exists(result)


class TestChopper(unittest.TestCase):
    """Test chopper module functions."""

    def test_block_pca(self):
        """Test block_pca function."""
        residues = [
            ("node1", np.array([1.0, 2.0, 3.0])),
            ("node2", np.array([2.0, 3.0, 4.0])),
            ("node3", np.array([3.0, 4.0, 5.0])),
        ]
        pca_coords = block_pca(residues)
        assert pca_coords.shape == (3, 3)
        assert len(pca_coords) == len(residues)

    def test_pca_chop(self):
        """Test pca_chop function."""
        residues = [
            ("node1", np.array([1.0, 2.0, 3.0])),
            ("node2", np.array([2.0, 3.0, 4.0])),
            ("node3", np.array([3.0, 4.0, 5.0])),
            ("node4", np.array([4.0, 5.0, 6.0])),
        ]
        s1, s2 = pca_chop(residues)
        assert len(s1) + len(s2) == len(residues)
        assert len(s1) > 0
        assert len(s2) > 0

    def test_chop(self):
        """Test recursive chop function."""
        # Create residues with coordinates
        residues = [
            (f"node{i}", np.array([float(i), float(i+1), float(i+2)]))
            for i in range(100)
        ]
        chops = list(chop(residues, max_size=50))
        assert len(chops) > 0
        # All chops should be <= max_size
        for chop_residues in chops:
            assert len(chop_residues) <= 50

    def test_graph_filter(self):
        """Test graph_filter function."""
        # Create a graph with non-canonical edges and enough nodes
        G = nx.Graph()
        for i in range(15):  # More than max_nodes=10
            G.add_node(i)
        G.add_edge(1, 2, LW="CWW")
        G.add_edge(2, 3, LW="B35")
        G.add_edge(3, 4, LW="TWW")  # Non-canonical
        assert graph_filter(G, max_nodes=10) is True

    def test_graph_filter_too_small(self):
        """Test graph_filter with too small graph."""
        G = nx.Graph()
        G.add_edge(1, 2, LW="CWW")
        assert graph_filter(G, max_nodes=10) is False

    def test_graph_filter_only_canonical(self):
        """Test graph_filter with only canonical edges."""
        G = nx.Graph()
        G.add_edge(1, 2, LW="CWW")
        G.add_edge(2, 3, LW="B35")
        G.add_edge(3, 4, LW="B53")
        assert graph_filter(G, max_nodes=10) is False

    def test_graph_clean(self):
        """Test graph_clean function."""
        # Create a test graph
        G = nx.Graph()
        G.add_edge(1, 2, LW="CWW")
        G.add_edge(2, 3, LW="B35")
        G.add_edge(3, 4, LW="TWW")
        G.add_node(5)  # Isolated node
        G.graph['pdbid'] = 'test'
        
        subG = G.subgraph([1, 2, 3, 4]).copy()
        cleaned = graph_clean(G, subG, thresh=8)
        assert cleaned is not None
        assert isinstance(cleaned, nx.Graph)

    def test_chop_one_rna(self):
        """Test chop_one_rna function."""
        # Create a test graph with coordinates
        G = nx.Graph()
        G.graph['pdbid'] = 'test'
        for i in range(100):
            G.add_node(f"test.A.{i}", C5prime_xyz=np.array([float(i), float(i+1), float(i+2)]))
        # Add some edges
        for i in range(99):
            G.add_edge(f"test.A.{i}", f"test.A.{i+1}", LW="CWW")
        
        subgraphs = chop_one_rna(G)
        # Should return a list (may be empty if filtering removes all)
        assert subgraphs is not None
        assert isinstance(subgraphs, list)

    def test_chop_one_rna_missing_coords(self):
        """Test chop_one_rna with missing coordinates."""
        G = nx.Graph()
        G.graph['pdbid'] = 'test'
        G.add_node("test.A.1")  # No coordinates
        subgraphs = chop_one_rna(G)
        assert subgraphs is not None


class TestKhopAnnotate(unittest.TestCase):
    """Test khop_annotate module functions."""

    def test_node_2_unordered_rings(self):
        """Test node_2_unordered_rings function."""
        # Load a real graph to test with proper structure
        test_data_path = "./tests/data/1fmn.json"
        if os.path.exists(test_data_path):
            G = load_graph(test_data_path)
            # Get a node from the graph
            node = list(G.nodes())[0]
            rings = node_2_unordered_rings(G, node, depth=2, hasher=None, hash_table=None)
            assert 'node' in rings
            assert 'edge' in rings
            assert 'graphlet' in rings
            assert len(rings['node']) == 3  # depth 0, 1, 2
            assert len(rings['edge']) == 3
            assert len(rings['graphlet']) == 3

    def test_node_2_unordered_rings_with_depth(self):
        """Test node_2_unordered_rings with different depth."""
        test_data_path = "./tests/data/1fmn.json"
        if os.path.exists(test_data_path):
            G = load_graph(test_data_path)
            node = list(G.nodes())[0]
            rings = node_2_unordered_rings(G, node, depth=3, hasher=None, hash_table=None)
            assert len(rings['node']) == 4  # depth 0, 1, 2, 3


class TestRetrieveStructures(unittest.TestCase):
    """Test retrieve_structures module functions."""

    def test_load_csv(self):
        """Test load_csv function."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("header1,header2\n")
            f.write("value1,1a9n\n")
            f.write("value2,1evv\n")
            csv_path = f.name
        
        try:
            repr_set = load_csv(csv_path, quiet=True)
            # load_csv reads all rows including header, so we get 3 items
            assert len(repr_set) == 3
            assert "header2" in repr_set  # Header row
            assert "1a9n" in repr_set
            assert "1evv" in repr_set
        finally:
            os.unlink(csv_path)

    def test_load_csv_empty(self):
        """Test load_csv with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("header1,header2\n")
            csv_path = f.name
        
        try:
            repr_set = load_csv(csv_path, quiet=True)
            assert isinstance(repr_set, list)
        finally:
            os.unlink(csv_path)
