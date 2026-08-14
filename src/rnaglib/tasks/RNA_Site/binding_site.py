import os
import numpy as np
from tqdm import tqdm
import networkx as nx

from rnaglib.dataset import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.encoders import BoolEncoder
from rnaglib.transforms import FeaturesComputer
from rnaglib.transforms import ChainNameTransform
from rnaglib.transforms import BindingSiteAnnotator, SmallMoleculeBindingTransform
from rnaglib.transforms import ResidueAttributeFilter, ChainFilter, ComposeFilters
from rnaglib.transforms import ConnectedComponentPartition
from rnaglib.dataset_transforms import SPLITTING_VARS, default_splitter_tr60_tr18, RandomSplitter, ClusterSplitter


class BenchmarkBindingSite(ResidueClassificationTask):
    """
    Version of RNA-Site implemented using the data and splitting of the experiment by Su et al. (2021)

    Hong Su, Zhenling Peng, and Jianyi Yang. Recognition of small molecule–rna binding sites using
    rna sequence and structure. Bioinformatics, 37(1):36–42, 2021. <https://doi.org/10.1093/bioinformatics/btaa1092>

    Task type: binary classification
    Task level: residue-level

    :param float cutoff: distance (in Angstroms) between an RNA atom and any small molecule atom below which the RNA residue is considered as part of a binding site (default 6.0)
    """
    target_var = "binding_site"
    input_var = "nt_code"
    name = "rna_site_bench"
    version = "2.0.2"
    default_metric = "balanced_accuracy"

    def __init__(self, cutoff=4.0,graph_path=None, **kwargs):
        self.cutoff = cutoff
        self.graph_path = graph_path
        meta = {"multi_label": False}
        super().__init__(additional_metadata=meta, **kwargs)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        if self.debug:
            return RandomSplitter()
        else:
            return default_splitter_tr60_tr18()

    def process(self) -> RNADataset:
        """"Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """
        dataset = RNADataset(
            dataset_path=self.graph_path, 
            debug=self.debug,
            in_memory=self.in_memory,
            redundancy="all",
            rna_id_subset=SPLITTING_VARS["PDB_TO_CHAIN_TR60_TE18"].keys(),
            version=self.version
        )

        chain_filter = ChainFilter(SPLITTING_VARS["PDB_TO_CHAIN_TR60_TE18"])
        bs_finder = SmallMoleculeBindingTransform(
            structures_dir=dataset.structures_path,
            additional_atoms=["CO", "S", "P"],
            mass_lower_limit=30,
            mass_upper_limit=1400
        )
        bs_annotator = BindingSiteAnnotator(include_ions=True)
        namer = ChainNameTransform()

        # Run through database, applying our filters
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if chain_filter.forward(rna):
                rna = bs_finder(rna)
                rna = bs_annotator(rna)
                rna = namer(rna)["rna"]
                self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(all_rnas)
        return dataset

    def post_process(self):
        pass

    def get_task_vars(self) -> FeaturesComputer:
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


class BindingSite(ResidueClassificationTask):
    """
    Predict the RNA residues which are the most likely to be part of binding sites for small molecule ligands

    Task type: binary classification
    Task level: residue-level

    :param float cutoff: distance (in Angstroms) between an RNA atom and any small molecule atom below which the RNA residue is considered as part of a binding site (default 6.0)
    :param tuple[int] size_thresholds: range of RNA sizes to keep in the task dataset(default (15, 500))
    """
    input_var = "nt_code"
    name = "rna_site"
    version = "2.0.2"
    default_metric = "balanced_accuracy"

    def __init__(self, cutoff=6.0, size_thresholds=(15, 500), graph_path=None, **kwargs):
        self.target_var = f"binding_small-molecule-{cutoff}A"
        self.graph_path = graph_path
        meta = {"multi_label": False}
        super().__init__(additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    def process(self) -> RNADataset:
        """"
        Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """
        # Define your transforms
        rna_filter = ResidueAttributeFilter(attribute=self.target_var, value_checker=lambda val: val is not None, aggregation_mode="min_valid", min_valid=10)
        connected_components_partition = ConnectedComponentPartition()

        connected_component_filters_list = []
        if self.size_thresholds is not None:
            size_filter = self.size_filter

        # Run through database, applying our filters
        dataset = RNADataset(dataset_path=self.graph_path, debug=self.debug, in_memory=self.in_memory, redundancy="all", version=self.version)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in tqdm(dataset, total=len(dataset), desc="Processing RNAs"):
            for rna_connected_component in connected_components_partition(rna):
                if self.size_thresholds is not None and not size_filter.forward(rna_connected_component):
                    continue
                if rna_filter.forward(rna_connected_component):
                    rna_g = rna_connected_component["rna"]
                    self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna_g)
        dataset = self.create_dataset_from_list(all_rnas)
        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        """Specifies the `FeaturesComputer` object of the tasks which defines the features which have to be added to the RNAs
        (graphs) and nucleotides (graph nodes)
        
        :return: the features computer of the task
        :rtype: FeaturesComputer
        """
        return FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        if self.debug:
            return RandomSplitter()
        else:
            return ClusterSplitter(distance_name="USalign")
