"""Inverse Folding task definitions"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

from rnaglib.data_loading import RNADataset
from rnaglib.dataset_transforms import (
    CDHitComputer,
    ClusterSplitter,
    NameSplitter,
    RedundancyRemover,
    StructureDistanceComputer,
)
from rnaglib.encoders import BoolEncoder, NucleotideEncoder
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.transforms import (
    ChainFilter,
    ConnectedComponentPartition,
    DummyAnnotator,
    FeaturesComputer,
)


class InverseFolding(ResidueClassificationTask):
    target_var = "nt_code"  # in rna graph
    input_var = "dummy"  # should be dummy variable
    nucs = ["A", "C", "G", "U"]
    name = "rna_if"

    def __init__(self, root,
                 size_thresholds=(15, 300),
                 additional_metadata=None,
                 **kwargs):
        if additional_metadata is None:
            meta = {"multi_label": False, "task_name": "rna_if"}
        else:
            meta = additional_metadata
        super().__init__(root=root, additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        return ClusterSplitter(distance_name="USalign")

    def process(self) -> RNADataset:
        # Define your transforms
        annotate_rna = DummyAnnotator()
        connected_components_partition = ConnectedComponentPartition()

        # Run through database, applying our filters
        dataset = RNADataset(in_memory=self.in_memory,
                             redundancy=self.redundancy)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for i, rna in tqdm(enumerate(dataset)):
            if self.debug:
                if i > 200: break
            for rna_connected_component in connected_components_partition(rna):
                if self.size_thresholds is not None and not self.size_filter.forward(rna_connected_component):
                    continue
                rna = annotate_rna(rna_connected_component)["rna"]
                self.add_rna_to_building_list(all_rnas=all_rnas, rna=rna)
        dataset = self.create_dataset_from_list(all_rnas)
        return dataset

    def post_process(self):
        cd_hit_computer = CDHitComputer(similarity_threshold=0.99)
        cd_hit_rr = RedundancyRemover(distance_name="cd_hit", threshold=0.9)
        self.dataset = cd_hit_computer(self.dataset)
        self.dataset = cd_hit_rr(self.dataset)

        us_align_computer = StructureDistanceComputer(name="USalign")
        self.dataset = us_align_computer(self.dataset)
        self.dataset.save_distances()

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(
            nt_features=self.input_var,
            nt_targets=self.target_var,
            custom_encoders={
                self.input_var: BoolEncoder(),
                self.target_var: NucleotideEncoder(),
            },
        )

    def compute_one_metric(self, preds, unfiltered_preds, probs, labels, unfiltered_labels):
        # Calculate metrics only on standard nucleotides
        # Note that accuracy is equivalent to sequence recovery rate
        one_metric = {
            "accuracy": accuracy_score(labels, preds),
            "mcc": matthews_corrcoef(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted"),
            # Calculate coverage (percentage of predictions that are standard nucleotides)
            "coverage": (unfiltered_preds != 0).mean(),
            # Add non-standard nucleotide statistics
            "non_standard_ratio": (unfiltered_labels == 0).mean(),
        }

        # Only calculate AUC for standard nucleotides, don't forget to offset i
        for i, nuc in enumerate(self.nucs):
            binary_labels = labels == i + 1
            binary_probs = probs[:, i + 1]
            binary_preds = preds == i + 1
            try:
                one_metric[f"auc_{nuc}"] = roc_auc_score(binary_labels, binary_probs)
                one_metric[f"f1_{nuc}"] = f1_score(binary_labels, binary_preds)
            except ValueError:
                one_metric[f"auc_{nuc}"] = float("nan")
                one_metric[f"f1_{nuc}"] = float("nan")
        # Add average AUC
        valid_aucs = [v for k, v in one_metric.items() if k.startswith("auc_") and not np.isnan(v)]
        one_metric["mean_auc"] = np.mean(valid_aucs) if valid_aucs else float("nan")
        return one_metric

    def compute_metrics(self, all_preds, all_probs, all_labels):
        """Evaluate model performance on nucleotide prediction task.

        Returns:
            dict: Dictionary containing metrics including loss if criterion provided

        Note: Label 0 represents non-standard/unknown nucleotides and is excluded
        from performance metrics to focus on ACGU prediction quality.
        """
        # Some metrics are computed only on standard nucleotides
        # Compute filtered versions of the predictions
        filtered_all_preds, filtered_all_probs, filtered_all_labels = [], [], []
        for pred, prob, label in zip(all_preds, all_probs, all_labels, strict=False):
            valid_mask = label != 0
            if len(valid_mask) > 0:
                filt_pred = pred[valid_mask]
                filt_prob = prob[valid_mask]
                filt_label = label[valid_mask]
                filtered_all_preds.append(filt_pred)
                filtered_all_probs.append(filt_prob)
                filtered_all_labels.append(filt_label)

        # Here we have a list of preds [(n1,), (n2,)...] for each residue in each RNA
        # Either compute the overall flattened results, or aggregate by system
        sorted_keys = []
        metrics = []
        for pred, filt_pred, prob, label, filt_label in zip(
            all_preds,
            filtered_all_preds,
            all_probs,
            all_labels,
            filtered_all_labels,
            strict=False,
        ):
            # Can't compute metrics over just one class
            if len(np.unique(label)) == 1:
                continue
            one_metric = self.compute_one_metric(pred, filt_pred, prob, label, filt_label)
            metrics.append([v for k, v in sorted(one_metric.items())])
            # metrics.append(np.array([v for k, v in sorted(one_metric.items())]))
            sorted_keys = sorted(one_metric.keys())
        metrics = np.array(metrics)
        mean_metrics = np.nanmean(metrics, axis=0)
        metrics = {k: v for k, v in zip(sorted_keys, mean_metrics, strict=False)}

        # Get the flattened result, renamed to include "global"
        filtered_all_preds = np.concatenate(filtered_all_preds)
        all_preds = np.concatenate(all_preds)
        filtered_all_probs = np.concatenate(filtered_all_probs)
        all_labels = np.concatenate(all_labels)
        filtered_all_labels = np.concatenate(filtered_all_labels)
        global_metrics = self.compute_one_metric(
            filtered_all_preds,
            all_preds,
            filtered_all_probs,
            filtered_all_labels,
            all_labels,
        )
        metrics_global = {f"global_{k}": v for k, v in global_metrics.items()}
        metrics.update(metrics_global)

        # Add confusion matrix (including non-standard nucleotides)
        cm = confusion_matrix(all_labels, all_preds)
        metrics["confusion_matrix"] = cm.tolist()
        return metrics


class gRNAde(InverseFolding):
    """This class is a subclass of InverseFolding and is used to train a model on the gRNAde dataset."""

    # everything is inherited except for process and splitter.
    name = "rna_if_bench"

    def __init__(self, root, size_thresholds=(15, 300), **kwargs):
        self.splits = {
            # Use sets instead of lists for chains since order doesn't matter
            "pdb_to_chain_train": defaultdict(set),
            "pdb_to_chain_test": defaultdict(set),
            "pdb_to_chain_val": defaultdict(set),
            "pdb_to_chain_all": defaultdict(set),
            "pdb_to_chain_all_single": defaultdict(set),
        }
        # Populate the structure
        data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
        for split in ["train", "test", "val"]:
            file_path = data_dir / f"{split}_ids_das.txt"
            with open(file_path) as f:
                for i, line in enumerate(f):
                    if kwargs['debug'] and i > 10:
                        break
                    line = line.strip()
                    pdb_id = line.split("_")[0].lower()
                    chain = line.split("_")[-1]
                    chain_components = list(chain.split("-"))

                    # Using update for sets automatically handles duplicates
                    self.splits[f"pdb_to_chain_{split}"][pdb_id].add(chain)
                    self.splits["pdb_to_chain_all"][pdb_id].add(chain)
                    self.splits["pdb_to_chain_all_single"][pdb_id].update(chain_components)

        meta = {"multi_label": False, "task_name": "rna_if_bench"}
        super().__init__(root=root, additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        train_names = [
            f"{pdb.lower()}_{chain}"
            for pdb in self.splits["pdb_to_chain_train"]
            for chain in self.splits["pdb_to_chain_train"][pdb]
        ]

        val_names = [
            f"{pdb.lower()}_{chain}"  # .upper()
            for pdb in self.splits["pdb_to_chain_val"]
            for chain in self.splits["pdb_to_chain_val"][pdb]
        ]

        test_names = [
            f"{pdb.lower()}_{chain}"  # .upper()
            for pdb in self.splits["pdb_to_chain_test"]
            for chain in self.splits["pdb_to_chain_test"][pdb]
        ]

        return NameSplitter(train_names, val_names, test_names)

    def process(self) -> RNADataset:
        """Returns a filtered and processed RNADataset."""
        pdb_to_single_chains = {
            pdb.lower(): [chain for chain in self.splits["pdb_to_chain_all_single"][pdb]]
            for pdb in self.splits["pdb_to_chain_all_single"]
        }

        chain_filter = ChainFilter(pdb_to_single_chains)
        annote_dummy = DummyAnnotator()

        # Initialize dataset with in_memory=False to avoid loading everything at once
        print(self.redundancy)
        source_dataset = RNADataset(rna_id_subset=list(pdb_to_single_chains.keys()),
                   redundancy=self.redundancy, in_memory=False)

        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)

        for rna in tqdm(source_dataset):
            if chain_filter.forward(rna):
                rna = annote_dummy(rna)
                base_graph = rna["rna"]
                pdb = base_graph.name
                for chain in self.splits["pdb_to_chain_all"][pdb]:
                    chain_components = set(chain.split("-"))
                    selected_nodes = [node for node in base_graph.nodes() if node.split(".")[1] in chain_components]
                    selected_chains = base_graph.copy().subgraph(selected_nodes)
                    selected_chains.name = f"{pdb.lower()}_{chain}"
                    self.add_rna_to_building_list(all_rnas=all_rnas, rna=selected_chains)
        dataset = self.create_dataset_from_list(all_rnas)
        return dataset

    def post_process(self):
        pass
