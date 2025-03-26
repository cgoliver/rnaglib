import os
import unittest

from rnaglib.utils import load_graph


class TestUtils(unittest.TestCase):
    def test_load_graph(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', "1fmn.json")
        g = load_graph(data_path, multigraph=False)
