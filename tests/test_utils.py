import unittest
import tempfile

from rnaglib.utils import load_graph


class TestUtils(unittest.TestCase):
    def test_load_graph(self):
        g = load_graph("examples/1fmn.json", multigraph=False)
        pass
