import os
from tqdm import tqdm

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ResidueAttributeFilter, DummyFilter, ComposeFilters
from rnaglib.transforms import ConnectedComponentPartition
from rnaglib.dataset_transforms import ClusterSplitter, StructureDistanceComputer, RedundancyRemover


class ChemicalModification(ResidueClassificationTask):
    """Residue-level binary classification task to predict whether a given
    residue is chemically modified.
    """

    target_var = "is_modified"
    input_var = "nt_code"
    
    def __init__(self, root, splitter=ClusterSplitter(distance_name="USalign"), size_thresholds=[5, 500], distance_computers=[StructureDistanceComputer(name="USalign")], redundancy_remover=RedundancyRemover(distance_name="USalign"), **kwargs):
        super().__init__(root=root, splitter=splitter, size_thresholds=size_thresholds, distance_computers=distance_computers, redundancy_remover=redundancy_remover, **kwargs)

    def get_task_vars(self):
        return FeaturesComputer(nt_targets=self.target_var, nt_features=self.input_var)

    def process(self):
        # Define your transforms
        residue_attribute_filter = ResidueAttributeFilter(attribute=self.target_var,
            value_checker=lambda val: val == True)
        if self.debug:
            residue_attribute_filter = DummyFilter()
        connected_components_partition = ConnectedComponentPartition()

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=self.in_memory)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in tqdm(dataset):
            for rna_connected_component in connected_components_partition(rna):
                if residue_attribute_filter.forward(rna_connected_component):
                    if self.size_thresholds is not None and not self.size_filter.forward(rna_connected_component):
                        continue
                    rna = rna_connected_component["rna"]
                    self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(all_rnas)


        # Apply the distances computations specified in self.distance_computers
        for distance_computer in self.distance_computers:
            dataset = distance_computer(dataset)
        dataset.save(self.dataset_path, recompute=False)

        # Remove redundancy if specified
        if self.redundancy_remover is not None:
            dataset = self.redundancy_remover(dataset)

        return dataset
