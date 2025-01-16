import os
import numpy as np
import json

from rnaglib.tasks import RNAClassificationTask
from rnaglib.data_loading import RNADataset
from rnaglib.encoders import IntEncoder
from rnaglib.transforms import (FeaturesComputer, AnnotatorFromDict, PartitionFromDict, ResolutionFilter,
                                ComposeFilters, ResidueAttributeFilter)


class LigandIdentification(RNAClassificationTask):
    """Binding pocket-level task where the job is to predict the (small molecule) ligand which is the most likely
    to bind a binding pocket with a given structure
    """
    input_var = "nt_code"
    target_var = "ligand_code"
    num_classes = 10

    def __init__(self, root, bp_dict_path, ligands_dict_path, splitter=None, **kwargs):
        self.bp_dict_path  = os.path.join(os.path.dirname(__file__), bp_dict_path)
        self.ligands_dict_path = os.path.join(os.path.dirname(__file__), ligands_dict_path)

        # bp_dict is a dictionary where a key is an RNA and a value is a list of binding pockets, each binding pocket
        # being itself a list of names of residues
        with open(self.bp_dict_path, 'r') as f1:
            self.bp_dict = json.load(f1)

        # ligands_dict is a dictionary where a key is a residue name and a value is the ligand code of the ligand binding
        # this residue
        with open(self.ligands_dict_path, 'r') as f2:
            self.ligands_dict = json.load(f2)
        self.mapping = {i: i for i in range(self.num_classes)}
        self.nodes_keep = list(self.bp_dict.keys())
        super().__init__(root=root, splitter=splitter, **kwargs)

    def process(self):
        # Initialize dataset with in_memory=False to avoid loading everything at once
        dataset = RNADataset(debug=self.debug, in_memory=False, redundancy="all", rna_id_subset=self.nodes_keep)

        # Instantiate filters to apply
        resolution_filter = ResolutionFilter(resolution_threshold=4.0)

        # Instantiate transforms to apply
        nt_partition = PartitionFromDict(partition_dict=self.bp_dict)
        annotator = AnnotatorFromDict(annotation_dict=self.ligands_dict, name="ligand_code")

        # Run through database, applying our filters
        all_binding_pockets = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if resolution_filter.forward(rna):
                for binding_pocket_dict in nt_partition(rna):
                    if self.size_thresholds is not None:
                        if not self.size_filter.forward(binding_pocket_dict):
                            continue
                    annotated_binding_pocket = annotator(binding_pocket_dict)
                    self.add_rna_to_building_list(all_rnas=all_binding_pockets, rna=annotated_binding_pocket["rna"])
        dataset = self.create_dataset_from_list(all_binding_pockets)
        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntEncoder(mapping=self.mapping)},
        )
