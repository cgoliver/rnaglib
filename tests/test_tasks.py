import unittest
import tempfile

from rnaglib.transforms import GraphRepresentation
from rnaglib.tasks import Task
from rnaglib.tasks import RNAFamilyTask
from rnaglib.tasks import ProteinBindingSiteDetection
from rnaglib.tasks import ChemicalModification


class TaskTest(unittest.TestCase):

    def check_task(self, task: Task):
        task.dataset.add_representation(GraphRepresentation(framework="pyg"))
        assert task.target_var is not None

        assert task.train_ind is not None
        assert task.test_ind is not None
        assert task.val_ind is not None

    def test_RNAFamilyTask(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = RNAFamilyTask(root=tmp, debug=True)
            self.check_task(ta)
        pass

    def test_ProteinBindingSiteDetectionTask(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ProteinBindingSiteDetection(root=tmp, debug=True)
            self.check_task(ta)

    def test_ChemicalModification(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = ChemicalModification(root=tmp, debug=True)
            self.check_task(ta)
