"""Inverse Folding task definitions"""

import torch
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer, DummyAnnotator
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.encoders import BoolEncoder, NucleotideEncoder


class InverseFolding(ResidueClassificationTask):
    target_var = "nt_code"  # in rna graph
    input_var = "dummy"  # should be dummy variable

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def process(self) -> RNADataset:
        dataset = RNADataset(debug=self.debug)
        rnas = DummyAnnotator()(dataset)
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
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

    def evaluate(self, model: torch.nn, loader) -> dict:
        """
        Evaluate model performance on nucleotide prediction task
        Args:
            model: The model to evaluate
            loader: Data loader to use
        Returns:
            dict: Dictionary containing metrics including loss if criterion provided

        Note: Label 0 represents non-standard/unknown nucleotides and is excluded
        from performance metrics to focus on ACGU prediction quality.
        """
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"]
                graph = graph.to(model.device)
                out = model(graph)

                if model.criterion is not None:
                    loss = model.criterion(out, graph.y.long())
                    total_loss += loss.item()

                # Get probabilities and predictions
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(out, dim=1)

                # Store results
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(graph.y.long().cpu().numpy())

        # Convert to numpy arrays for metric calculation
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Create mask for standard nucleotides (exclude 0s)
        valid_mask = all_labels != 0

        # Calculate metrics only on standard nucleotides
        metrics = {
            # Note that accuracy is equivalent to sequence recovery rate
            "accuracy": accuracy_score(all_labels[valid_mask], all_preds[valid_mask]),
            "mcc": matthews_corrcoef(all_labels[valid_mask], all_preds[valid_mask]),
            "macro_f1": f1_score(
                all_labels[valid_mask], all_preds[valid_mask], average="macro"
            ),
            "weighted_f1": f1_score(
                all_labels[valid_mask], all_preds[valid_mask], average="weighted"
            ),
            "per_class_f1": f1_score(
                all_labels[valid_mask], all_preds[valid_mask], average=None
            ).tolist(),
        }

        # Add confusion matrix (including non-standard nucleotides)
        cm = confusion_matrix(all_labels, all_preds)
        metrics["confusion_matrix"] = cm.tolist()

        # Calculate coverage (percentage of predictions that are standard nucleotides)
        metrics["coverage"] = (all_preds != 0).mean()

        # Calculate per-class metrics for standard nucleotides
        for i, nuc in enumerate(["X", "A", "C", "G", "U"]):  # Include 'X' for 0
            if i == 0:  # Skip AUC calculation for non-standard class
                continue

            binary_labels = all_labels == i
            binary_probs = all_probs[:, i]

            # Only calculate AUC for standard nucleotides where they should appear
            valid_positions = all_labels != 0
            try:
                metrics[f"auc_{nuc}"] = roc_auc_score(
                    binary_labels[valid_positions], binary_probs[valid_positions]
                )
            except ValueError:
                metrics[f"auc_{nuc}"] = float("nan")

        # Add average AUC
        valid_aucs = [
            v for k, v in metrics.items() if k.startswith("auc_") and not np.isnan(v)
        ]
        if valid_aucs:
            metrics["mean_auc"] = np.mean(valid_aucs)

        if model.criterion is not None:
            metrics["loss"] = total_loss / len(loader)

        # Add non-standard nucleotide statistics
        metrics["non_standard_ratio"] = (all_labels == 0).mean()
        metrics["non_standard_prediction_rate"] = (all_preds == 0).mean()

        return metrics
