import os

import pandas as pd

from rnaglib.tasks import RNAClassificationTask
from rnaglib.data_loading import RNADataset
from rnaglib.encoders import IntEncoder
from rnaglib.transforms import FeaturesComputer, AnnotatorFromDict, PartitionFromDict, ResidueNameFilter
from rnaglib.utils import dump_json


class LigandIdentification(RNAClassificationTask):
    input_var = "nt_code"
    target_var = "ligand_code"
    num_classes = 44

    def __init__(self, root, data, splitter=None, **kwargs):
        self.data = data
        self.nodes_keep = set(data.nid.values)
        self.bp_dict, self.ligands_dict = self.parse_data()
        super().__init__(root=root, splitter=splitter, **kwargs)

    def parse_data(self):
        self.data["RNA"] = self.data.nid.apply(lambda id: id.split(".")[0])
        binding_pockets = self.data[["RNA", "idx", "nid"]].groupby(["RNA", "idx"])["nid"].apply(lambda x: x.to_list())
        # create a dict where key is RNA name and values are lists of lists [[residue 1 of binding pocket 1,...,residue N of BP 1],...,[residue 1 of BP k,...]]
        bp_dict = {
            rna: [binding_pockets[rna, bp_idx] for bp_idx in binding_pockets[rna].index]
            for rna in binding_pockets.index.droplevel(1)
        }
        ligands_dict = {residue_label[0]: residue_label[1] for residue_label in self.data[["nid", "label"]].values}
        return bp_dict, ligands_dict

    def process(self):
        rna_filter = ResidueNameFilter(value_checker=lambda name: name in self.nodes_keep, min_valid=1)
        nt_partition = PartitionFromDict(partition_dict=self.bp_dict)
        annotator = AnnotatorFromDict(annotation_dict=self.ligands_dict, name="ligand_code")

        # Run through database, applying our filters
        # Initialize dataset with in_memory=False to avoid loading everything at once
        dataset = RNADataset(
            debug=self.debug, in_memory=False, redundancy="all", rna_id_subset=list(self.data["RNA"].unique())
        )
        all_binding_pockets = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if rna_filter.forward(rna):
                for binding_pocket_dict in nt_partition(rna):
                    annotated_binding_pocket = annotator(binding_pocket_dict)["rna"]
                    if self.in_memory:
                        all_binding_pockets.append(annotated_binding_pocket)
                    else:
                        all_binding_pockets.append(annotated_binding_pocket.name)
                        dump_json(
                            os.path.join(self.dataset_path, f"{annotated_binding_pocket.name}.json"),
                            annotated_binding_pocket,
                        )
        if self.in_memory:
            dataset = RNADataset(rnas=all_binding_pockets)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_binding_pockets)

        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        self.mapping = {i: i for i in range(self.num_classes)}
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntEncoder(mapping=self.mapping)},
        )
