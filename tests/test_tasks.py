import unittest
import tempfile

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


class TaskTest(unittest.TestCase):
    default_dataset_params = {"debug": True, 
                              "in_memory": False, 
                              "precomputed": False,
                              "splitter": RandomSplitter()
                              }

    def check_task(self, task: Task):
        task.dataset.add_representation(GraphRepresentation(framework="pyg"))
        assert task.target_var is not None
        assert task.train_ind is not None
        assert task.test_ind is not None
        assert task.val_ind is not None

    def test_RNAGO(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = RNAGo(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_ProteinBindingSite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ProteinBindingSite(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_ChemicalModification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_InverseFolding(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = InverseFolding(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_gRNAde(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = gRNAde(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_LigandIdentification(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_filename = 'binding_pockets.csv'
            ta = LigandIdentification(root=tmp, data_filename=data_filename,
                **self.default_dataset_params)
            self.check_task(ta)

    def test_BindingSite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BindingSite(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_BenchmarkBindingSite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BenchmarkBindingSite(root=tmp, **self.default_dataset_params)
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
