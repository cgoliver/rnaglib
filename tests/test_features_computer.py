import unittest
import tempfile

from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import RNAFMTransform


class TransformsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.dataset = RNADataset(debug=True)

    def test_default_feature(self):
        ft = FeaturesComputer(nt_features=["nt_code"])
        ft(self.dataset[0])
        pass

    def test_add_feature(self):
        ft = FeaturesComputer(nt_features=["nt_code"])
        ft.add_feature("alpha")
        ft(self.dataset[0])
        pass

    def test_add_custom_feature(self):
        t = RNAFMTransform()
        ft = FeaturesComputer()
        ft.add_feature(t.name, custom_encoders={t.name: t.encoder})
        feat_dict = ft(t(self.dataset[0]))

    def test_add_dataset_features(self):
        self.dataset.features_computer.add_feature(["nt_code"])
        pass

    def test_add_dataset_custom_features(self):
        t = RNAFMTransform()
        self.dataset.features_computer.add_feature(
            t.name, custom_encoders={t.name: t.encoder}
        )
        pass
