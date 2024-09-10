from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import Splitter
import pandas as pd
import ast
import os
import torch
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score
from networkx import set_node_attributes
from rnaglib.encoders import BoolEncoder


class DasSplitter(Splitter):
    def __init__(self, seed=0, **kwargs):
        super().__init__(**kwargs)
        print('Initialising DasSplitter')
        self.seed = seed

        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        splits_path = os.path.join(parent_dir, 'tasks/data', 'das_split.pt')
        metadata_path = os.path.join(parent_dir, 'tasks/data', 'gRNAde_metadata.csv')

        # Note that preprocessing is needed since splits contain indices of compounds that may contain multiple pdbs.
        # Our approach treats each pdb as an individual sample.
        splits = torch.load(splits_path)
        metadata = pd.read_csv(metadata_path)
        metadata_ids = metadata['id_list'].apply(ast.literal_eval)
        train_pdbs = self._process_split(metadata_ids, splits[0])
        val_pdbs = self._process_split(metadata_ids, splits[1])
        test_pdbs = self._process_split(metadata_ids, splits[2])
        # If you want to convince yourself that this is the right order, see this notebook:
        # https://github.com/chaitjo/geometric-rna-design/blob/deccaa0139f7f9130487858ece2fbca331100369/notebooks/split_das.ipynb
        self.train_pdbs = train_pdbs
        self.val_pdbs = val_pdbs
        self.test_pdbs = test_pdbs
        pass

    def __call__(self, dataset):
        print('Generating split indices')
        dataset_map = {value['rna'].graph['pdbid'][0]: idx for idx, value in enumerate(dataset)}
        train_ind = [dataset_map[item] for item in self.train_pdbs if item in dataset_map]
        val_ind = [dataset_map[item] for item in self.val_pdbs if item in dataset_map]
        test_ind = [dataset_map[item] for item in self.test_pdbs if item in dataset_map]
        return train_ind, val_ind, test_ind

    def _process_split(self, metadata_ids, indices):
        return [x.split('_')[0] for x in sum(metadata_ids.iloc[indices].to_list(), [])]


class gRNAde(ResidueClassificationTask):
    target_var = "nt_code"  # in rna graph
    input_var = "dummy"  # in rna graph

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    def sequence_recovery(predictions, target_sequence):
        # predictions are a tensor of designed sequences with shape `(n_samples, seq_len)`
        recovery = predictions.eq(target_sequence).float().mean(dim=1).cpu().numpy()
        return recovery

    def evaluate(self, model, loader, criterion, device):
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                graph = batch['graph']
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
            f1 = f1_score(all_labels, all_preds, average='weighted')
            auc = roc_auc_score(all_labels, all_probs, average='weighted', multi_class='ovr')
            mcc = matthews_corrcoef(all_labels, all_preds)

            print(f'Test Accuracy: {accuracy:.4f}')
            print(f'Test F1 Score: {f1:.4f}')
            print(f'Test AUC: {auc:.4f}')
            print(f'Test MCC: {mcc:.4f}')

            return accuracy, f1, auc, avg_loss, mcc

    def default_splitter(self):
        return DasSplitter()
        # SingleStateSplit
        # MultiStateSplit

    def _annotator(self, x):
        dummy = {
            node: 1
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, dummy, 'dummy')
        return x

    def build_dataset(self, root):
        # load metadata from gRNAde if it fails, print link
        try:
            current_dir = os.path.dirname(__file__)
            metadata = pd.read_csv(os.path.join(current_dir, 'data/gRNAde_metadata.csv'))
        except FileNotFoundError:
            print(
                'Download the metadata from https://drive.google.com/file/d/1lbdiE1LfWPReo5VnZy0zblvhVl5QhaF4/ and place it in the ./data dir')

        # generate list
        rnas_keep = []

        for sample in metadata['id_list']:
            per_sample_list = ast.literal_eval(sample)
            rnas_keep.extend(per_sample_list)
        # remove extra info from strings
        rnas_keep = [x.split('_')[0] for x in rnas_keep]

        features_computer = FeaturesComputer(nt_targets=[self.target_var],
                                             nt_features=[self.input_var],
                                             custom_encoders_features={self.input_var: BoolEncoder()})
        dataset = RNADataset.from_database(features_computer=features_computer,
                                       redundancy='all',
                                       annotator=self._annotator,
                                       rna_filter=lambda x: x.graph['pdbid'][0] in rnas_keep)

        return dataset
