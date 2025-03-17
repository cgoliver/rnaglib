import os
import numpy as np
from tqdm import tqdm
from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer
from rnaglib.dataset_transforms import SPLITTING_VARS, default_splitter_tr60_tr18, RandomSplitter
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.encoders import BoolEncoder
from rnaglib.transforms import ResidueAttributeFilter
from rnaglib.transforms import ChainNameTransform
from rnaglib.transforms import BindingSiteAnnotator
from rnaglib.transforms import SmallMoleculeBindingTransform
from rnaglib.transforms import ChainFilter, ComposeFilters
from rnaglib.transforms import ConnectedComponentPartition
from rnaglib.dataset_transforms import ClusterSplitter


class BenchmarkBindingSite(ResidueClassificationTask):
    target_var = "binding_site"
    input_var = "nt_code"
    name = "rna_site_bench"
    version = "2.0.2"

    def __init__(self, root, cutoff=6.0, **kwargs):
        self.cutoff = cutoff
        meta = {"task_name": "rna_site_bench", "multi_label":False}
        super().__init__(root=root, additional_metadata=meta, **kwargs)

    @property
    def default_splitter(self):
        if self.debug:
            return RandomSplitter()
        else:
            return default_splitter_tr60_tr18()

    def process(self) -> RNADataset:
        dataset = RNADataset(
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
        bs_annotator = BindingSiteAnnotator(include_ions=True,
                                            include_covalent=True)
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
        return FeaturesComputer(
            nt_features=self.input_var,
            nt_targets=self.target_var,
            custom_encoders={self.target_var: BoolEncoder()},
        )


class BindingSite(ResidueClassificationTask):
    input_var = "nt_code"
    name = "rna_site"
    version = "2.0.2"

    def __init__(self, root, cutoff=6.0,size_thresholds=(15, 500), **kwargs):
        self.target_var = f"binding_small-molecule-{cutoff}A"
        meta = {"task_name": "rna_site", "multi_label":False}
        super().__init__(root=root, additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    def process(self) -> RNADataset:
        # Define your transforms
        rna_filter = ResidueAttributeFilter(attribute=self.target_var, value_checker=lambda val: val is not None)
        connected_components_partition = ConnectedComponentPartition()

        protein_content_filter = ResidueAttributeFilter(
            attribute="protein_content_8.0", aggregation_mode="aggfunc", value_checker=lambda x: x < 10, aggfunc=np.mean
        )
        connected_component_filters_list = [protein_content_filter]
        if self.size_thresholds is not None:
            connected_component_filters_list.append(self.size_filter)
        connected_component_filters = ComposeFilters(connected_component_filters_list)

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=self.in_memory, redundancy="all", version=self.version)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in tqdm(dataset, total=len(dataset), desc="Processing RNAs"):
            if rna_filter.forward(rna):
                for rna_connected_component in connected_components_partition(rna):
                    if not connected_component_filters.forward(rna_connected_component):
                        continue
                    rna = rna_connected_component["rna"]
                    self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(all_rnas)
        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)

    @property
    def default_splitter(self):
        if self.debug:
            return RandomSplitter()
        else:
            return ClusterSplitter(distance_name="USalign")
