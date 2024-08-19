import os
import hashlib
import json
from functools import cached_property
from typing import Union, Optional

import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, roc_auc_score

from rnaglib.data_loading import RNADataset
from rnaglib.splitters import Splitter


class Task:
    """ Abstract class for a benchmarking task using the rnaglib datasets. This class handles the logic for building the underlying dataset which is held in an rnaglib.data_loading.RNADataset object. Onc e the dataset is created, the splitter is invoked to create the train/val/test indices. Tasks also define an evaluate() function to yield appropriate model performance metrics.

    :param root: path to a folder where the task information will be stored for fast loading.
    :param recompute: whether to recompute the task info from scratch or use what is stored in root.
    :param splitter: rnaglib.splitters.Splitter object that handles splitting of data into train/val/test indices. If None uses task's default_splitter() attribute.
    """
    def __init__(self, 
                 root: Union[str, os.PathLike],
                 recompute: bool = False,
                 splitter: Optional[Splitter] = None):
        self.root = root
        self.recompute = recompute

        if splitter is None:
            self.splitter = self.default_splitter()
        else:
            self.splitter = splitter

        self.dataset = self._build_dataset(root)
        self.split()

        if not os.path.exists(root) or recompute:
            os.makedirs(root, exist_ok=True)
            self.write()

    def write(self):
        """ Save task data and splits to root. Creates a folder in ``root`` called
        ``'graphs'`` which stores the RNAs that form the dataset, and three `.txt` files (`'{train, val, test}_idx.txt'`,
        one for each split with a list of indices.

        """
        print(">>> Saving dataset.")
        os.makedirs(os.path.join(self.root, 'graphs'), exist_ok=True)
        self.dataset.save(os.path.join(self.root, 'graphs'))

        with open(os.path.join(self.root, 'train_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.train_ind]
        with open(os.path.join(self.root, 'val_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.val_ind]
        with open(os.path.join(self.root, 'test_idx.txt'), 'w') as idx:
            [idx.write(str(ind) + "\n") for ind in self.test_ind]
        with open(os.path.join(self.root, "task_id.txt"), "w") as tid:
            tid.write(self.task_id)
        print(">>> Done")

    def split(self):
        """ Sets train, val, and test indices as attributes of the task. Can be accessed
        as ``self.train_ind``, etc. Will load splits if they are saved in `root` otherwise,
        recomputes from scratch by invoking ``self.splitter()``.
        """
        if not os.path.exists(os.path.join(self.root, "train_idx.txt")) or self.recompute:
            print(">>> Computing splits...")
            train_ind, val_ind, test_ind = self.splitter(self.dataset)
        else:
            print(">>> Loading splits...")
            train_ind = [int(ind) for ind in open(os.path.join(self.root, "train_idx.txt"), 'r').readlines()]
            val_ind = [int(ind) for ind in open(os.path.join(self.root, "val_idx.txt"), 'r').readlines()]
            test_ind = [int(ind) for ind in open(os.path.join(self.root, "test_idx.txt"), 'r').readlines()]
        self.train_ind = train_ind
        self.val_ind = val_ind
        self.test_ind = test_ind
        return train_ind, val_ind, test_ind

    def _build_dataset(self, root):
        # check if dataset exists and load
        if os.path.exists(os.path.join(self.root, 'graphs')) and not self.recompute:
            return RNADataset(nt_targets=[self.target_var],
                              nt_features=[self.input_var],
                              saved_dataset=os.path.join(self.root, 'graphs')
                              )
        return self.build_dataset(root)

    def build_dataset(self, root):
        raise NotImplementedError

    @cached_property
    def task_id(self):
        """ Task hash is a hash of all RNA ids and node IDs in the dataset"""
        h = hashlib.new('sha256')
        for rna in self.dataset.rnas:
            h.update(rna.graph['pdbid'][0].encode("utf-8"))
            for nt in sorted(rna.nodes()):
                h.update(nt.encode("utf-8"))
        [h.update(str(i).encode("utf-8")) for i in self.train_ind]
        [h.update(str(i).encode("utf-8")) for i in self.val_ind]
        [h.update(str(i).encode("utf-8")) for i in self.test_ind]
        return h.hexdigest()

    def __eq__(self, other):
        return self.task_id == other.task_id


    def evaluate(self, model, test_loader, criterion, device):
        raise NotImplementedError


class ResidueClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, loader, criterion, device):
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        for batch in loader:
            graph = batch['graph']
            graph = graph.to(device)
            out = model(graph)
            loss = criterion(out, torch.flatten(graph.y).long())
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(graph.y.tolist()) 
        
        avg_loss = total_loss / len(loader)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)

        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Test F1 Score: {f1:.4f}')
        print(f'Test AUC: {auc:.4f}')
        print(f'Test MCC: {mcc:.4f}')  
        
        return accuracy, f1, auc, avg_loss, mcc


class RNAClassificationTask(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    '''
    def evaluate(self, graph_level_attribute, test_predictions):
        from sklearn.metrics import matthews_corrcoef
        true = [self.dataset[idx]['graph'][graph_level_attribute] for idx in self.test_ind]
        mcc_scores = [matthews_corrcoef([true[i]], [test_predictions[i]]) for i in range(len(self.test_ind))]
        return {'mcc': sum(mcc_scores) / len(mcc_scores)}
    '''
    def evaluate(self, model, test_loader, criterion, device):
        model.eval()  
        
        all_preds = []
        all_labels = []
        all_probs = []
        test_loss = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                test_loss += loss.item()
                
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = (all_preds == all_labels).mean()
        avg_loss = test_loss / len(test_loader)
        
        # Calculate MCC
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        # Calculate F1 score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f'Test Loss: {avg_loss:.4f}')
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'Matthews Correlation Coefficient: {mcc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        return avg_loss, accuracy, mcc, f1
