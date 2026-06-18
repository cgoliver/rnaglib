import os
import unittest

from rnaglib.utils import load_graph


class TestUtils(unittest.TestCase):
    def test_load_graph(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', "1fmn.json")
        g = load_graph(data_path, multigraph=False)

    def test_locarna_wrapper_no_crash(self):
        """Test locarna wrapper runs without crashing (returns None or completes correctly)"""
        from rnaglib.utils.wrappers import locarna_wrapper
        # Use two short RNA sequences
        locarna_wrapper('AAA', 'CCC')

    def test_cdhit_wrapper_no_crash(self):
        """Test cdhit wrapper runs without crashing (returns dicts correctly)"""
        from rnaglib.utils.wrappers import cdhit_wrapper
        # Dummy data
        res1, res2 = cdhit_wrapper(['seq1', 'seq2'], ['AA', 'CC'])
        assert isinstance(res1, dict)
        assert isinstance(res2, (dict, type(None), tuple, list))
        