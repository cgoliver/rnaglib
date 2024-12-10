"""Inverse Folding task definitions"""

import os
from collections import defaultdict
import numpy as np
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score, confusion_matrix

from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer, DummyAnnotator, ComposeFilters, RibosomalFilter, RNAAttributeFilter
from rnaglib.transforms import NameFilter, ChainFilter, ChainSplitTransform, ChainNameTransform
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.encoders import BoolEncoder, NucleotideEncoder

from rnaglib.splitters import NameSplitter
from rnaglib.utils import dump_json


class InverseFolding(ResidueClassificationTask):
    target_var = "nt_code"  # in rna graph
    input_var = "dummy"  # should be dummy variable
    nucs = ["A", "C", "G", "U"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self) -> RNADataset:
        # build the filters
        ribo_filter = RibosomalFilter()
        resolution_filter = RNAAttributeFilter(
            attribute="resolution_high", value_checker=lambda val: float(val[0]) < 4.0
        )
        filters = ComposeFilters([ribo_filter, resolution_filter])

        # Define your transforms
        annotate_rna = DummyAnnotator()

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=self.in_memory)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if filters.forward(rna):
                rna = annotate_rna(rna)["rna"]
                if self.in_memory:
                    all_rnas.append(rna)
                else:
                    all_rnas.append(rna.name)
                    dump_json(os.path.join(self.dataset_path, f"{rna.name}.json"), rna)
        if self.in_memory:
            dataset = RNADataset(rnas=all_rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_rnas)
        return dataset

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
        """
        Evaluate model performance on nucleotide prediction task
        Returns:
            dict: Dictionary containing metrics including loss if criterion provided

        Note: Label 0 represents non-standard/unknown nucleotides and is excluded
        from performance metrics to focus on ACGU prediction quality.
        """

        # Some metrics are computed only on standard nucleotides
        # Compute filtered versions of the predictions
        filtered_all_preds, filtered_all_probs, filtered_all_labels = [], [], []
        for pred, prob, label in zip(all_preds, all_probs, all_labels):
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
            all_preds, filtered_all_preds, all_probs, all_labels, filtered_all_labels
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
        metrics = {k: v for k, v in zip(sorted_keys, mean_metrics)}

        # Get the flattened result, renamed to include "global"
        filtered_all_preds = np.concatenate(filtered_all_preds)
        all_preds = np.concatenate(all_preds)
        filtered_all_probs = np.concatenate(filtered_all_probs)
        all_labels = np.concatenate(all_labels)
        filtered_all_labels = np.concatenate(filtered_all_labels)
        global_metrics = self.compute_one_metric(
            filtered_all_preds, all_preds, filtered_all_probs, filtered_all_labels, all_labels
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

    def __init__(self, **kwargs):
        self.splits = {
            "train": [],
            "test": [],
            "val": [],
            "all": [],
            # Use sets instead of lists for chains since order doesn't matter
            "pdb_to_chain_train": defaultdict(set),
            "pdb_to_chain_test": defaultdict(set),
            "pdb_to_chain_val": defaultdict(set),
            "pdb_to_chain_all": defaultdict(set),
        }
        # Populate the structure
        data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "data"
        for split in ["train", "test", "val"]:
            file_path = data_dir / f"{split}_ids_das.txt"
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    pdb_id = line.split("_")[0].lower()
                    chain = line.split("_")[-1]  # .upper()
                    chain_components = list(chain.split("-"))
                    # [c.upper() for c in chain.split("-")]

                    if pdb_id not in self.splits[split]:
                        self.splits[split].append(pdb_id)
                    if pdb_id not in self.splits["all"]:
                        self.splits["all"].append(pdb_id)

                    # Using update for sets automatically handles duplicates
                    self.splits[f"pdb_to_chain_{split}"][pdb_id].update(chain_components)
                    self.splits["pdb_to_chain_all"][pdb_id].update(chain_components)

        super().__init__(**kwargs)

    @property
    def default_splitter(self):
        train_names = [
            f"{pdb.lower()}_{chain}"  # .upper()
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
        """
        Process the dataset in batches to avoid memory issues.
        Returns a filtered and processed RNADataset.
        """
        name_filter = NameFilter(self.splits["train"] + self.splits["test"] + self.splits["val"])
        chain_filter = ChainFilter(self.splits["pdb_to_chain_all"])
        filters = ComposeFilters([name_filter, chain_filter])

        annote_dummy = DummyAnnotator()
        split_chain = ChainSplitTransform()
        add_name_chains = ChainNameTransform()

        # Initialize dataset with in_memory=False to avoid loading everything at once
        source_dataset = RNADataset(debug=self.debug, redundancy="all", in_memory=False)

        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        import tqdm

        for rna in tqdm.tqdm(source_dataset):
            if filters.forward(rna):
                rna = annote_dummy(rna)
                rna_chains = split_chain(rna)  # Split by chain
                renamed_chains = list(add_name_chains(rna_chains))  # Rename
                for rna_chain in renamed_chains:
                    rna_chain = rna_chain["rna"]
                    if self.in_memory:
                        all_rnas.append(rna_chain)
                    else:
                        all_rnas.append(rna_chain.name)
                        dump_json(os.path.join(self.dataset_path, f"{rna_chain.name}.json"), rna_chain)
        if self.in_memory:
            dataset = RNADataset(rnas=all_rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_rnas)
        return dataset
