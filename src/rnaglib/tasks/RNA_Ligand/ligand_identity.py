import os
import pandas as pd
import numpy as np
import json

from rnaglib.tasks import RNAClassificationTask
from rnaglib.data_loading import RNADataset
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import FeaturesComputer, AnnotatorFromDict, PartitionFromDict, ResolutionFilter
from rnaglib.dataset_transforms import ClusterSplitter, StructureDistanceComputer, RedundancyRemover

class LigandIdentification(RNAClassificationTask):
    """Binding pocket-level task where the job is to predict the (small molecule) ligand which is the most likely
    to bind a binding pocket with a given structure
    """
    input_var = "nt_code"
    target_var = "ligand"
    ligands_nb = 10

    def __init__(self, root, data_filename, splitter=ClusterSplitter(distance_name="USalign"), size_thresholds=[5, 500], distance_computers=[StructureDistanceComputer(name="USalign")], redundancy_remover=RedundancyRemover(distance_name="USalign"), **kwargs):
        self.data_path  = os.path.join(os.path.dirname(__file__), "data", data_filename)
        binding_pockets = pd.read_csv(self.data_path)
        binding_pockets3 = binding_pockets[["RNA", "bp_id", "nid"]].groupby(["RNA", "bp_id"])["nid"].apply(lambda x: x.to_list())

        # create a dict where key is RNA name and values are lists of lists [[residue 1 of binding pocket 1,...,residue N of BP 1],...,[residue 1 of BP k,...]]
        self.bp_dict = {
            rna: [binding_pockets3[rna, bp_idx] for bp_idx in binding_pockets3[rna].index]
            for rna in binding_pockets3.index.droplevel(1)
        }
        self.ligands_dict = {rna_ligand[0]:rna_ligand[1] for rna_ligand in binding_pockets[["nid","ligand"]].values}
        self.nodes_keep = list(self.bp_dict.keys())
        super().__init__(root=root, splitter=splitter, size_thresholds=size_thresholds, distance_computers=distance_computers, redundancy_remover=redundancy_remover, **kwargs)

    def process(self):
        # Initialize dataset with in_memory=False to avoid loading everything at once
        dataset = RNADataset(debug=self.debug, in_memory=False, redundancy="all", rna_id_subset=self.nodes_keep)

        # Instantiate filters to apply
        resolution_filter = ResolutionFilter(resolution_threshold=4.0)

        # Instantiate transforms to apply
        nt_partition = PartitionFromDict(partition_dict=self.bp_dict)
        # annotator = AnnotatorFromDict(annotation_dict=self.ligands_dict, name="ligand_code")
        annotator = AnnotatorFromDict(annotation_dict=self.ligands_dict, name="ligand")

        # Run through database, applying our filters
        all_binding_pockets = []
        ligands_dict = {}
        idx = 0
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if resolution_filter.forward(rna):
                for binding_pocket_dict in nt_partition(rna):
                    if self.size_thresholds is not None:
                        if not self.size_filter.forward(binding_pocket_dict):
                            continue
                    annotated_binding_pocket = annotator(binding_pocket_dict)
                    self.add_rna_to_building_list(all_rnas=all_binding_pockets, rna=annotated_binding_pocket["rna"])
                    try:
                        ligands_dict[annotated_binding_pocket["rna"].graph["ligand"]].append(idx)
                    except:
                        ligands_dict[annotated_binding_pocket["rna"].graph["ligand"]] = [idx]
                    idx += 1
        ligands_binding_pockets_nb = np.array([len(ligands_dict[ligand]) for ligand in ligands_dict])
        ligands_to_keep_indices = np.argsort(ligands_binding_pockets_nb)[-self.ligands_nb:]
        ligands_to_keep = list(np.array(list(ligands_dict.keys()))[ligands_to_keep_indices])
        indices_to_keep = sorted([bp_idx for ligand in ligands_to_keep for bp_idx in ligands_dict[ligand]])
        top_ligand_binding_pockets = [all_binding_pockets[i] for i in indices_to_keep]

        dataset = self.create_dataset_from_list(top_ligand_binding_pockets)

        # Apply the distances computations specified in self.distance_computers
        for distance_computer in self.distance_computers:
            dataset = distance_computer(dataset)
        dataset.save(self.dataset_path, recompute=False)

        # Remove redundancy if specified
        if self.redundancy_remover is not None:
            dataset = self.redundancy_remover(dataset)
            
        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        represented_values = set()
        for rna in self.dataset:
            represented_values.add(rna['rna'].graph[self.target_var])
        self.mapping = {target_value: i for i, target_value in enumerate(represented_values)}
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(mapping=self.mapping)},
        )