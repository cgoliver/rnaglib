import unittest
import tempfile

from rnaglib.tasks import RNAFamilyTask

class TaskTest(unittest.TestCase):

    def test_RNAFamilyTask(self):
        with tempfile.TemporaryDirectory() as tmp:
            ta = RNAFamilyTask(root=tmp, debug=True)
        pass
