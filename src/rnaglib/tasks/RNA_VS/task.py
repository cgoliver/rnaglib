import os
import sys

import numpy as np
import pickle
from torch.utils.data import DataLoader

from rnaglib.tasks.RNA_VS.build_data import build_data
from rnaglib.tasks.RNA_VS.data import VSRNATestDataset, VSRNATrainDataset, VSCollater
from rnaglib.tasks.RNA_VS.evaluate import run_virtual_screen
from rnaglib.tasks.RNA_VS.ligands import MolGraphEncoder


class VSTask:
    script_dir = os.path.dirname(__file__)
    json_dump = os.path.join(script_dir, "../data/rna_vs/dataset_as_json.json")
    trainval_groups, test_groups = pickle.load(open(json_dump, 'rb'))

    def __init__(self, root, ligand_framework='dgl', recompute=False):
        self.root = root
        self.recompute = recompute
        self.build_dataset()
        train_cut = int(0.9 * len(self.trainval_groups))
        train_groups_keys = set(np.random.choice(list(self.trainval_groups.keys()), size=train_cut, replace=False))
        self.train_groups = {k: v for k, v in self.trainval_groups.items() if k in train_groups_keys}
        self.val_groups = {k: v for k, v in self.trainval_groups.items() if k not in train_groups_keys}
        # TODO: add support for pyg ligand graphs
        self.ligand_encoder = MolGraphEncoder(cache_path=os.path.join(self.root, 'ligands.p'))

    def build_dataset(self):
        # check if dataset exists and load
        if not os.path.exists(os.path.join(self.root, 'graphs')) or self.recompute:
            build_data(root=self.root, recompute=self.recompute)

    def get_split_datasets(self, dataset_kwargs=None):
        train_dataset = VSRNATrainDataset(groups=self.train_groups,
                                          ligand_embedder=self.ligand_encoder,
                                          dataset_path=os.path.join(self.root, 'graphs'),
                                          **dataset_kwargs)
        val_dataset = VSRNATrainDataset(groups=self.val_groups,
                                        ligand_embedder=self.ligand_encoder,
                                        dataset_path=os.path.join(self.root, 'graphs'),
                                        **dataset_kwargs)
        test_dataset = VSRNATestDataset(groups=self.test_groups,
                                        ligand_embedder=self.ligand_encoder,
                                        dataset_path=os.path.join(self.root, 'graphs'),
                                        **dataset_kwargs)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        return train_dataset, val_dataset, test_dataset

    def get_split_loaders(self, dataset_kwargs=None, dataloader_kwargs=None):
        # If datasets were not already precomputed
        if 'train_dataset' not in self.__dict__:
            self.get_split_datasets(dataset_kwargs=dataset_kwargs)
        if dataloader_kwargs is None:
            dataloader_kwargs = {'collate_fn': VSCollater(self.train_dataset)}
        if 'collate_fn' not in dataloader_kwargs:
            collater = VSCollater(self.train_dataset)
            dataloader_kwargs['collate_fn'] = collater
        train_loader = DataLoader(dataset=self.train_dataset, **dataloader_kwargs)
        val_loader = DataLoader(dataset=self.val_dataset, **dataloader_kwargs)
        test_dataloader_kwargs = dataloader_kwargs.copy()
        test_dataloader_kwargs['batch_size'] = 1
        test_loader = DataLoader(dataset=self.test_dataset, **test_dataloader_kwargs)
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.test_dataloader = test_loader
        return train_loader, val_loader, test_loader

    def evaluate(self, model):
        return run_virtual_screen(model, self.test_dataloader)


if __name__ == '__main__':
    from rnaglib.representations.graph import GraphRepresentation

    # Create a task
    root = "../../data/tasks/rna_vs"
    ef_task = VSTask(root)

    # Build corresponding datasets and dataloader
    representations = [GraphRepresentation(framework='dgl')]
    rna_dataset_args = {'representations': representations, 'nt_features': 'nt_code'}
    rna_loader_args = {'batch_size': 2}
    # train_dataset, val_dataset, test_dataset = ef_task.get_split_datasets(rna_dataset_args)
    train_dataloader, val_dataloader, test_dataloader = ef_task.get_split_loaders(dataset_kwargs=rna_dataset_args,
                                                                                  dataloader_kwargs=rna_loader_args)

    # Check both models work well
    for i, elt in enumerate(train_dataloader):
        # print(elt)
        a = 1
        # if i > 3:
        #     break
        if not i % 50:
            print(i, len(train_dataloader))

    for i, elt in enumerate(test_dataloader):
        # print(elt)
        a = 1
        # if i > 3:
        #     break
        if not i % 10:
            print(i, len(train_dataloader))
