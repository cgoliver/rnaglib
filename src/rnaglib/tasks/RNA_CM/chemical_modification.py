import os
from tqdm import tqdm

from rnaglib.dataset import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ResidueAttributeFilter, DummyFilter
from rnaglib.transforms import ConnectedComponentPartition
from rnaglib.dataset_transforms import ClusterSplitter


class ChemicalModification(ResidueClassificationTask):
    """Residue-level binary classification task to predict whether a given residue is chemically modified.

    Task type: binary classification
    Task level: residue-level

    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset(default (15, 500))
    """

    target_var = "is_modified"
    input_var = "nt_code"
    name = "rna_cm"
    default_metric = "balanced_accuracy"
    version = "2.0.2"

    def __init__(self, size_thresholds=(15, 500), **kwargs):
        meta = {'multi_label': False}
        super().__init__(additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        return ClusterSplitter(distance_name="USalign")

    def get_task_vars(self):
        """Specifies the `FeaturesComputer` object of the tasks which defines the features which have to be added to the RNAs
        (graphs) and nucleotides (graph nodes)
        
        :return: the features computer of the task
        :rtype: FeaturesComputer
        """
        return FeaturesComputer(nt_targets=self.target_var, nt_features=self.input_var)

    def process(self) -> RNADataset:
        """
        Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """
        # Define your transforms
        residue_attribute_filter = ResidueAttributeFilter(
            attribute=self.target_var, value_checker=lambda val: val == True
        )
        if self.debug:
            residue_attribute_filter = DummyFilter()
        connected_components_partition = ConnectedComponentPartition()

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=self.in_memory, version=self.version)
        all_rnas = []
        for rna in tqdm(dataset):
            for rna_connected_component in connected_components_partition(rna):
                if residue_attribute_filter.forward(rna_connected_component):
                    if self.size_thresholds is not None and not self.size_filter.forward(rna_connected_component):
                        continue
                    rna = rna_connected_component["rna"]
                    self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(all_rnas)
        print(f"len of process: {len(dataset)}")
        return dataset
