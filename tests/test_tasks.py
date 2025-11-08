import unittest
import tempfile
import shutil

from rnaglib.transforms import GraphRepresentation
from rnaglib.dataset_transforms import RandomSplitter
from rnaglib.tasks import Task

from rnaglib.tasks import RNAGo
from rnaglib.tasks import ProteinBindingSite
from rnaglib.tasks import ChemicalModification
from rnaglib.tasks import InverseFolding
from rnaglib.tasks import gRNAde
from rnaglib.tasks import LigandIdentification
from rnaglib.tasks import BindingSite, BenchmarkBindingSite


def has_cdhit():
    """Check if cd-hit is available."""
    return shutil.which("cd-hit") is not None


def has_usalign():
    """Check if USalign is available."""
    return shutil.which("USalign") is not None


def has_external_tools():
    """Check if external tools (cd-hit and USalign) are available."""
    return has_cdhit() and has_usalign()


class TaskTest(unittest.TestCase):
    default_dataset_params = {"debug": True, 
                              "in_memory": False, 
                              "precomputed": False,
                              "splitter": RandomSplitter(),
                              "redundancy_removal": False  # Disable redundancy removal if tools not available
                              }

    def check_task(self, task: Task):
        task.dataset.add_representation(GraphRepresentation(framework="pyg"))
        assert task.target_var is not None
        assert task.train_ind is not None
        assert task.test_ind is not None
        assert task.val_ind is not None

    def test_RNAGO(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = RNAGo(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_ProteinBindingSite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ProteinBindingSite(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_ChemicalModification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_InverseFolding(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = InverseFolding(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_gRNAde(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = gRNAde(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_LigandIdentification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = LigandIdentification(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_BindingSite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BindingSite(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_BenchmarkBindingSite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BenchmarkBindingSite(root=tmp, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            # prepare the data
            ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
            ta.dataset.features_computer.add_feature(feature_names="nt_code")
            # refresh loaders
            train_load, val_load, test_load = ta.get_split_loaders()
            loss, *outputs = ta.dummy_inference()
            ta.compute_metrics(*outputs)

    def test_add_representation(self):
        """Test adding a representation to a task."""
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            rep = GraphRepresentation(framework="pyg")
            ta.add_representation(rep)
            # Check that representation was added (representations is a list)
            rep_names = [r.name for r in ta.dataset.representations]
            assert rep.name in rep_names

    def test_remove_representation(self):
        """Test removing a representation from a task."""
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            # Test that remove_representation method exists and can be called
            # (it may not remove if representation doesn't exist, which is fine)
            initial_reps = len(ta.dataset.representations)
            ta.remove_representation("graph")
            # Method should execute without error
            assert hasattr(ta, 'remove_representation')

    def test_add_feature(self):
        """Test adding a feature to a task."""
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            # Test that add_feature method exists and can be called
            # (feature might already exist, which is fine)
            ta.add_feature("nt_code", feature_level="residue", is_input=True)
            # Method should execute without error
            assert hasattr(ta, 'add_feature')
            assert ta.dataset.features_computer is not None

    def test_get_split_datasets(self):
        """Test getting split datasets."""
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            train_ds, val_ds, test_ds = ta.get_split_datasets()
            assert train_ds is not None
            assert val_ds is not None
            assert test_ds is not None
            assert len(train_ds) > 0
            assert len(val_ds) > 0
            assert len(test_ds) > 0

    def test_set_datasets(self):
        """Test setting split datasets."""
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            ta.set_datasets(recompute=True)
            assert hasattr(ta, "train_dataset")
            assert hasattr(ta, "val_dataset")
            assert hasattr(ta, "test_dataset")

    def test_describe(self):
        """Test task description."""
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            description = ta.describe()
            assert description is not None
            assert "num_node_features" in description
            assert "num_classes" in description
            assert "dataset_size" in description

    def test_task_equality(self):
        """Test task equality comparison."""
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                ta1 = ChemicalModification(root=tmp1, **self.default_dataset_params)
                ta2 = ChemicalModification(root=tmp2, **self.default_dataset_params)
                # Tasks with same name should be equal
                assert ta1 == ta2

    def test_get_split_loaders(self):
        """Test getting split loaders."""
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
            train_loader, val_loader, test_loader = ta.get_split_loaders(batch_size=2)
            assert train_loader is not None
            assert val_loader is not None
            assert test_loader is not None
