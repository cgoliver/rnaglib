import os
from tqdm import tqdm

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import DummyFilter, ResidueAttributeFilter
from rnaglib.transforms import ConnectedComponentPartition
from rnaglib.dataset_transforms import ClusterSplitter


class ProteinBindingSite(ResidueClassificationTask):
    """Residue-level task where the job is to predict a binary variable
    at each residue representing the probability that a residue belongs to
    a protein-binding interface
    """

    target_var = "protein_binding"
    input_var = "nt_code"
    name = "rna_prot"

    def __init__(self, root,
                 size_thresholds=(15, 500),
                 **kwargs):
        super().__init__(root=root, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        return ClusterSplitter(distance_name="USalign")

    def get_task_vars(self):
        return FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)

    def process(self):
        # Define your transforms
        filters = ResidueAttributeFilter(attribute=self.target_var,
            value_checker=lambda val: val is not None)
        if self.debug:
            filters = DummyFilter()
        connected_components_partition = ConnectedComponentPartition()

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=False)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in tqdm(dataset, total=len(dataset)):
            if filters.forward(rna):
                for rna_connected_component in connected_components_partition(rna):
                    if self.size_thresholds is not None:
                        if not self.size_filter.forward(rna_connected_component):
                            continue
                    rna = rna_connected_component["rna"]
                    self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(rnas=all_rnas)
        return dataset
