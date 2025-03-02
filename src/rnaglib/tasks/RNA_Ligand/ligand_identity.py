import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch

from rnaglib.dataset_transforms import RandomSplitter
from rnaglib.tasks import RNAClassificationTask
from rnaglib.data_loading import RNADataset, Collater
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import FeaturesComputer, AnnotatorFromDict, PartitionFromDict, ResolutionFilter
from rnaglib.dataset_transforms import ClusterSplitter, CDHitComputer, StructureDistanceComputer
from .prepare_dataset import PrepareDataset


class LigandIdentification(RNAClassificationTask):
    """Binding pocket-level task where the job is to predict the (small molecule) ligand which is the most likely
    to bind a binding pocket with a given structure
    """
    input_var = "nt_code"
    target_var = "ligand"
    name = "rna_ligand"

    def __init__(self, 
        root, 
        data_filename = 'binding_pockets.csv',
        size_thresholds=(15, 500),
        admissible_ligands = ['PAR','LLL','8UZ'],
        use_balanced_sampler=False,
        **kwargs
    ):
        self.admissible_ligands = admissible_ligands
        self.data_path = os.path.join(os.path.dirname(__file__), "data", data_filename)
        self.use_balanced_sampler = use_balanced_sampler
        binding_pockets = pd.read_csv(self.data_path)
        binding_pockets3 = binding_pockets[["RNA", "bp_id", "nid"]].groupby(["RNA", "bp_id"])["nid"].apply(
            lambda x: x.to_list())

        # create a dict where key is RNA name and values are lists of lists [[residue 1 of binding pocket 1,...,residue N of BP 1],...,[residue 1 of BP k,...]]
        self.bp_dict = {
            rna: [binding_pockets3[rna, bp_idx] for bp_idx in binding_pockets3[rna].index]
            for rna in binding_pockets3.index.droplevel(1)
        }
        self.ligands_dict = {rna_ligand[0]: rna_ligand[1] for rna_ligand in binding_pockets[["nid", "ligand"]].values}
        self.nodes_keep = list(self.bp_dict.keys())
        super().__init__(root=root, size_thresholds=size_thresholds, **kwargs)

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
        for rna in tqdm(dataset):
            if resolution_filter.forward(rna):
                for binding_pocket_dict in nt_partition(rna):
                    if self.size_thresholds is not None:
                        if not self.size_filter.forward(binding_pocket_dict):
                            continue
                    annotated_binding_pocket = annotator(binding_pocket_dict)
                    current_ligand = binding_pocket_dict["rna"].graph["ligand"]
                    if current_ligand in self.admissible_ligands:
                        self.add_rna_to_building_list(all_rnas=all_binding_pockets, rna=annotated_binding_pocket["rna"])
                        try:
                            ligands_dict[current_ligand].append(idx)
                        except:
                            ligands_dict[current_ligand] = [idx]
                        idx += 1
        dataset = self.create_dataset_from_list(all_binding_pockets)
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

    def post_process(self):
        """
        The most common post_processing steps to remove redundancy.

        Other tasks should implement their own if this is not the desired default behavior
        """
        cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
        prepare_dataset = PrepareDataset(distance_name="cd_hit", threshold=0.9)
        us_align_computer = StructureDistanceComputer(name="USalign")
        self.dataset = cd_hit_computer(self.dataset)
        self.dataset = prepare_dataset(self.dataset)
        self.dataset = us_align_computer(self.dataset)

    def set_loaders(self, recompute=True, **dataloader_kwargs):
        """Sets the dataloader properties.
        Call this each time you modify ``self.dataset``.
        """
        self.set_datasets(recompute=recompute)

        # If no collater is provided we need one
        if dataloader_kwargs is None:
            dataloader_kwargs = {"collate_fn": Collater(self.train_dataset)}
        if "collate_fn" not in dataloader_kwargs:
            collater = Collater(self.train_dataset)
            dataloader_kwargs["collate_fn"] = collater

        targets = np.array([self.mapping[rna['rna'].graph["ligand"]] for rna in self.train_dataset])


        samples_weight = np.array([1./self.metadata["description"]["class_distribution"][str(i)] for i in targets])
        samples_weight = torch.from_numpy(samples_weight)
        balanced_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # Now build the loaders
        if self.use_balanced_sampler:
            self.train_dataloader = DataLoader(dataset=self.train_dataset, sampler=balanced_sampler,  **dataloader_kwargs)
        else:
            self.train_dataloader = DataLoader(dataset=self.train_dataset,  **dataloader_kwargs)
        dataloader_kwargs["shuffle"] = False
        self.val_dataloader = DataLoader(dataset=self.val_dataset, **dataloader_kwargs)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, **dataloader_kwargs)

    @property
    def default_splitter(self):
        return ClusterSplitter(distance_name="USalign")
