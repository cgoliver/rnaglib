import os
import unittest
import tempfile
import pandas as pd

from rnaglib.transforms import GraphRepresentation
from rnaglib.tasks import Task
from rnaglib.tasks import RNAFamily
from rnaglib.tasks import ProteinBindingSiteDetection
from rnaglib.tasks import ChemicalModification

from rnaglib.tasks import RNAFamily
from rnaglib.tasks import ProteinBindingSiteDetection
from rnaglib.tasks import ChemicalModification
from rnaglib.tasks import InverseFolding
from rnaglib.tasks import gRNAde
from rnaglib.tasks import LigandIdentification
from rnaglib.tasks import BindingSiteDetection, BenchmarkBindingSiteDetection


class TaskTest(unittest.TestCase):
    default_dataset_params = {"debug": True, "in_memory": False}

    def check_task(self, task: Task):
        task.dataset.add_representation(GraphRepresentation(framework="pyg"))
        assert task.target_var is not None
        assert task.train_ind is not None
        assert task.test_ind is not None
        assert task.val_ind is not None

    def test_RNAFamily(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = RNAFamily(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_ProteinBindingSiteDetectionTask(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ProteinBindingSiteDetection(root=tmp, **self.default_dataset_params)
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
            # data = pd.DataFrame({'idx':[0,0,229],'label':[0,0,1],'nid':['165d.B.15','165d.B.16','2h0z.A.6']})
            data = pd.read_csv(
                os.path.join(os.path.dirname(__file__), "../src/rnaglib/tasks/RNA_Ligand/data/gmsm_dataset.csv")
            )
            ta = LigandIdentification(root=tmp, data=data, recompute=True, **self.default_dataset_params)
            self.check_task(ta)

    def test_BindingSiteDetection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BindingSiteDetection(root=tmp, **self.default_dataset_params)
            self.check_task(ta)

    def test_BenchmarkBindingSiteDetection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BenchmarkBindingSiteDetection(root=tmp, **self.default_dataset_params)
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
