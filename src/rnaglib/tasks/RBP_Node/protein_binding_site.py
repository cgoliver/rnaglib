import os

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ComposeFilters
from rnaglib.transforms import RibosomalFilter
from rnaglib.transforms import DummyFilter
from rnaglib.transforms import PDBIDNameTransform
from rnaglib.transforms import ResidueAttributeFilter
from rnaglib.utils import dump_json


class ProteinBindingSiteDetection(ResidueClassificationTask):
    """Residue-level task where the job is to predict a binary variable
    at each residue representing the probability that a residue belongs to
    a protein-binding interface
    """

    target_var = "binding_protein"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def get_task_vars(self):
        return FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)

    def process(self):
        # build the filters
        ribo_filter = RibosomalFilter()
        non_bind_filter = ResidueAttributeFilter(attribute=self.target_var, value_checker=lambda val: val is not None)
        filters = ComposeFilters([ribo_filter, non_bind_filter])
        if self.debug:
            filters = DummyFilter()

        # Define your transforms
        add_name = PDBIDNameTransform()

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=self.in_memory)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if filters.forward(rna):
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
