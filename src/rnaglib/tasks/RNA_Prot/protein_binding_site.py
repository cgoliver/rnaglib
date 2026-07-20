import os

from tqdm import tqdm

from rnaglib.dataset import RNADataset
from rnaglib.dataset_transforms import ClusterSplitter
from rnaglib.encoders import BoolEncoder
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import ConnectedComponentPartition, DummyFilter, FeaturesComputer, ResidueAttributeFilter, ComposeFilters

# Lowercase pdbid -> set of author chain IDs, confirmed crystallization
# chaperones (U1A RRM domain / Fab-Fv frameworks) by sequence identity, with
# native-job structures (real spliceosomal assemblies, U1A's own PIE-element
# autoregulation) excluded. Seeded from the RNA_Prot task-validity audit
# (2026-07-18). This is a scope choice (a chaperone is a real, folded protein
# making a real physicochemical interface, just not this RNA's evolved
# partner), not a bug -- it belongs here, in the task, not in the annotation
# transform.
CHAPERONE_CHAINS = {
    "1drz": {"A"},
    "1m5k": {"C"},
    "1u6b": {"A"},
    "2nz4": {"A", "B", "C"},
    "2r8s": {"H"},
    "3cul": {"A"},
    "3egz": {"A"},
    "3hhn": {"B"},
    "3irw": {"P"},
    "3k0j": {"A", "C"},
    "3p49": {"B"},
    "4kzd": {"H"},
    "4w90": {"B"},
    "5ddo": {"G"},
    "5fj4": {"A"},
    "6db8": {"H"},
    "6las": {"D", "E"},
    "6mwn": {"C", "H"},
    "6u8d": {"H"},
    "6x5m": {"H"},
    "6xh1": {"A"},
    "6xjq": {"C", "H"},
    "7d7v": {"C"},
    "7dlz": {"A"},
    "7mlx": {"H"},
    "7szu": {"H"},
    "7xlt": {"H"},
    "8d29": {"A", "H"},
    "8dp3": {"H"},
    "8gxb": {"C", "D", "F"},
    "8jy0": {"A", "C"},
    "8sh5": {"H"},
    "8t29": {"A"},
    "8uiw": {"H"},
    "8vm9": {"A", "H"},
    "8vma": {"A", "H"},
    "8vmb": {"H"},
    "9aur": {"H"},
    "9dn4": {"A"},
}


class ProteinBindingSite(ResidueClassificationTask):
    """The job is to predict a binary variable
    at each residue representing the probability that a residue belongs to
    a protein-binding interface

    Task type: binary classification
    Task level: residue-level

    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset(default (15, 500))
    :param bool exclude_chaperone_contacts: if True, relabel as non-binding any residue whose only contributing protein chain(s) are known crystallization chaperones (see CHAPERONE_CHAINS), rather than a biological partner of this RNA (default False)
    """

    target_var = "protein_binding"
    input_var = "nt_code"
    name = "rna_prot"
    default_metric = "balanced_accuracy"
    version = "2.0.2"

    def __init__(self, size_thresholds=(15, 500), graph_path=None, exclude_chaperone_contacts=False, **kwargs):
        meta = {"multi_label": False}
        self.graph_path = graph_path
        self.exclude_chaperone_contacts = exclude_chaperone_contacts
        super().__init__(additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        return ClusterSplitter(distance_name="USalign", debug=self.debug)

    def get_task_vars(self):
        """Specifies the `FeaturesComputer` object of the tasks which defines the features which have to be added to the RNAs
        (graphs) and nucleotides (graph nodes)
        
        :return: the features computer of the task
        :rtype: FeaturesComputer
        """
        return FeaturesComputer(
            nt_features=self.input_var,
            nt_targets=self.target_var,
            custom_encoders={self.target_var: BoolEncoder()},
        )

    def process(self) -> RNADataset:
        """"
        Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """
        if self.size_thresholds is not None:
            connected_component_filters_list = [self.size_filter]
        else:
            connected_component_filters_list = []
        if not self.debug:
            # Check that the selected RNA subsets have at least one protein-binding residue
            min_one_binding_residue_filter = ResidueAttributeFilter(attribute=self.target_var, value_checker=lambda val: val, aggregation_mode="min_valid", min_valid=1)
            connected_component_filters_list.append(min_one_binding_residue_filter)
            # Check that the selected RNA subsets have at least one non-protein-binding residue
            min_one_non_binding_residue_filter = ResidueAttributeFilter(attribute=self.target_var, value_checker=lambda val: not val, aggregation_mode="min_valid", min_valid=1)
            connected_component_filters_list.append(min_one_non_binding_residue_filter)
        filters = ComposeFilters(connected_component_filters_list)
        connected_components_partition = ConnectedComponentPartition()

        # Run through database, applying our filters
        dataset = RNADataset(dataset_path=self.graph_path, debug=self.debug, in_memory=False, version=self.version)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in tqdm(dataset, total=len(dataset)):
            if self.exclude_chaperone_contacts:
                g = rna["rna"]
                pdbid = g.graph["pdbid"].lower()
                chaperone_chains = CHAPERONE_CHAINS.get(pdbid, set())
                if chaperone_chains:
                    for node, data in g.nodes(data=True):
                        contributing = set(data["protein_binding_chains"])
                        if contributing and contributing <= chaperone_chains:
                            data[self.target_var] = False  # every contributing chain is a known chaperone
            for rna_connected_component in connected_components_partition(rna):
                if not filters.forward(rna_connected_component):
                    continue
                rna = rna_connected_component["rna"]
                self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(rnas=all_rnas)
        return dataset
