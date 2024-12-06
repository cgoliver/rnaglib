import unittest
import tempfile

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

    def check_task(self, task: Task):
        task.dataset.add_representation(GraphRepresentation(framework="pyg"))
        assert task.target_var is not None
        assert task.train_ind is not None
        assert task.test_ind is not None
        assert task.val_ind is not None

    def test_RNAFamily(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = RNAFamily(root=tmp, debug=True)
            self.check_task(ta)
        pass

    def test_ProteinBindingSiteDetectionTask(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ProteinBindingSiteDetection(root=tmp, debug=True)
            self.check_task(ta)

    def test_ChemicalModification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, debug=False)
            self.check_task(ta)

    def test_InverseFolding(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = InverseFolding(root=tmp, debug=True)
            self.check_task(ta)

    def test_gRNAde(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = gRNAde(root=tmp, debug=True)
            self.check_task(ta)

    def test_LigandIdentification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = LigandIdentification(root=tmp, debug=True)
            self.check_task(ta)

    def test_BindingSiteDetection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BindingSiteDetection(root=tmp, debug=True)
            self.check_task(ta)

    def test_BenchmarkBindingSiteDetection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = BenchmarkBindingSiteDetection(root=tmp, debug=True)
            self.check_task(ta)

    def test_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, debug=True)
            # prepare the data
            ta.dataset.add_representation(GraphRepresentation(framework="pyg"))
            ta.dataset.features_computer.add_feature(feature_names="nt")
            # refresh loaders
            train_load, val_load, test_load = ta.get_split_loaders()
            ta.evaluate(ta.dummy_model, test_load)
        pass
