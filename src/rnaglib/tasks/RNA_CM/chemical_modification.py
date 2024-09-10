from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ResidueAttributeFilter
from rnaglib.transforms import PDBIDNameTransform


class ChemicalModification(ResidueClassificationTask):
    """Residue-level binary classification task to predict whether or not a given
    residue is chemically modified.
    """

    target_var = "is_modified"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def build_dataset(self, root):
        rnas = ResidueAttributeFilter(attribute=self.target_var)(
            RNADataset(debug=self.debug)
        )
        rnas = PDBIDNameTransform()(rnas)
        features_computer = FeaturesComputer(nt_targets=self.target_var)
        dataset = RNADataset(
            rnas=[r["rna"] for r in rnas], features_computer=features_computer
        )
        return dataset
