import os

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ResidueAttributeFilter
from rnaglib.transforms import DummyFilter
from rnaglib.transforms import PDBIDNameTransform
from rnaglib.utils import dump_json


class ChemicalModification(ResidueClassificationTask):
    """Residue-level binary classification task to predict whether or not a given
    residue is chemically modified.
    """

    target_var = "is_modified"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def get_task_vars(self):
        return FeaturesComputer(nt_targets=self.target_var, nt_features=self.input_var)

    def process(self):
        # Define your transforms
        rna_filter = ResidueAttributeFilter(attribute=self.target_var, value_checker=lambda val: val == True)
        if self.debug:
            rna_filter = DummyFilter()
        add_name = PDBIDNameTransform()

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=self.in_memory)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if rna_filter.forward(rna):
                rna = add_name(rna)["rna"]
                if self.in_memory:
                    all_rnas.append(rna)
                else:
                    all_rnas.append(rna.name)
                    dump_json(os.path.join(self.dataset_path, f"{rna.name}.json"), rna)
        if self.in_memory:
            dataset = RNADataset(rnas=all_rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_rnas)
        return dataset
