import os
from tqdm import tqdm

from rnaglib.dataset import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ResidueAttributeFilter, DummyFilter
from rnaglib.transforms import ConnectedComponentPartition
from rnaglib.transforms import BiologicalModificationAnnotator
from rnaglib.dataset_transforms import ClusterSplitter


class ChemicalModification(ResidueClassificationTask):
    """Residue-level binary classification task to predict whether a given residue carries a
    natural, enzymatically installed RNA modification.

    A positive is a residue whose PDB code is a curated natural modification (pseudouridine,
    dihydrouridine, methylations, ...). The database-level ``is_modified`` flag is not used as
    the target because it is set from residue-name length and also fires on synthetic analogs
    (halogen/2'-F/LNA), DNA residues and nucleotide ligands. The stricter target is derived on
    the fly by :class:`BiologicalModificationAnnotator`, so no database rebuild is needed.

    Task type: binary classification
    Task level: residue-level

    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset(default (15, 500))
    :param allowed_modifications: iterable of PDB residue codes to count as positives. Defaults
        to :data:`rnaglib.config.NATURAL_RNA_MODIFICATIONS`; widen it to change the scope
        (e.g. to include synthetic analogs).
    """

    target_var = "is_biological_modification"
    input_var = "nt_code"
    name = "rna_cm"
    default_metric = "balanced_accuracy"
    version = "3.0.0"

    def __init__(self, size_thresholds=(15, 500), graph_path=None, allowed_modifications=None, **kwargs):
        meta = {'multi_label': False}
        self.graph_path = graph_path
        self.allowed_modifications = allowed_modifications
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
        modification_annotator = BiologicalModificationAnnotator(
            allowed_modifications=self.allowed_modifications
        )
        residue_attribute_filter = ResidueAttributeFilter(
            attribute=self.target_var, value_checker=lambda val: val == True
        )
        if self.debug:
            residue_attribute_filter = DummyFilter()
        connected_components_partition = ConnectedComponentPartition()

        # Run through database, applying our filters
        dataset = RNADataset(dataset_path=self.graph_path, debug=self.debug, in_memory=self.in_memory, version=self.version)
        all_rnas = []
        for rna in tqdm(dataset):
            for rna_connected_component in connected_components_partition(rna):
                modification_annotator(rna_connected_component)
                if residue_attribute_filter.forward(rna_connected_component):
                    if self.size_thresholds is not None and not self.size_filter.forward(rna_connected_component):
                        continue
                    rna = rna_connected_component["rna"]
                    self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(all_rnas)
        print(f"len of process: {len(dataset)}")
        return dataset
