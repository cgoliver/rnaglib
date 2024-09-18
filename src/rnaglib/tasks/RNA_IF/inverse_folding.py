from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import RandomSplitter
from networkx import set_node_attributes
from rnaglib.encoders import BoolEncoder

import torch
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score


class InverseFolding(ResidueClassificationTask):
    target_var = "nt_code"  # in rna graph
    input_var = "dummy"  # should be dummy variable

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def evaluate(self, model, loader, criterion, device):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                graph = batch["graph"]
                graph = graph.to(device)
                out = model(graph)
                loss = criterion(out, graph.y)  # torch.flatten(graph.y).long())
                total_loss += loss.item()
                probs = torch.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.tolist())
                labels = graph.y.argmax(dim=1)
                all_labels.extend(labels.tolist())
                all_probs.append(probs.cpu())

            avg_loss = total_loss / len(loader)

            all_preds = torch.tensor(all_preds)
            all_labels = torch.tensor(all_labels)
            all_probs = torch.cat(all_probs, dim=0)

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="weighted")
            auc = roc_auc_score(
                all_labels, all_probs, average="weighted", multi_class="ovr"
            )
            mcc = matthews_corrcoef(all_labels, all_preds)

            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1 Score: {f1:.4f}")
            print(f"Test AUC: {auc:.4f}")
            print(f"Test MCC: {mcc:.4f}")

            return accuracy, f1, auc, avg_loss, mcc

    def default_splitter(self):
        return RandomSplitter()

    def _annotator(self, x):
        dummy = {node: 1 for node, nodedata in x.nodes.items()}
        set_node_attributes(x, dummy, "dummy")
        return x

    def build_dataset(self):
        print("building dataset task")

        features_computer = FeaturesComputer(
            nt_targets=[self.target_var],
            nt_features=[self.input_var],
            custom_encoders_features={self.input_var: BoolEncoder()},
        )
        dataset = RNADataset.from_database(
            features_computer=features_computer,
            rna_filter=lambda x: x.graph["pdbid"][0],
            annotator=self._annotator,
        )
        return dataset
