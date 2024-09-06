import unittest
import tempfile

from rnaglib.tasks import Task
from rnaglib.tasks import RNAFamilyTask

class TaskTest(unittest.TestCase):

    def check_task(self, task: Task):
        assert task.target_var is not None

        assert task.train_ind is not None
        assert task.test_ind is not None
        assert task.val_ind is not None

    def test_RNAFamilyTask(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = RNAFamilyTask(root=tmp, debug=True)
            self.check_task(ta)
        pass
