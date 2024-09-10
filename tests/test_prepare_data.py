import unittest

from rnaglib.prepare_data import fr3d_to_graph


class TestPrepareData(unittest.TestCase):
    def test_fr3d_to_graph(self):
        fr3d_to_graph("./src/rnaglib/data/1evv.cif")
        pass

    pass
