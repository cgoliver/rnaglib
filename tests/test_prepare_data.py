import unittest
from unittest import mock
import tempfile
from types import SimpleNamespace
from pathlib import Path

from rnaglib.prepare_data import fr3d_to_graph
from rnaglib.prepare_data.main import prepare_data_main


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
        fr3d_to_graph("./src/rnaglib/data/1evv.cif")
        pass

    def test_database_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.args.structures_dir = Path(tmpdir) / "structures"
            self.args.output_dir = Path(tmpdir) / "build"
            prepare_data_main(self.args)
        pass

    pass
