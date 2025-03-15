import os

import numpy as np
import json
from torch.utils.data import DataLoader

from rnaglib.tasks.RNA_VS.build_data import dump_rna_jsons, precompute_ligand_graphs
from rnaglib.tasks.RNA_VS.data import VSRNATestDataset, VSRNATrainDataset, VSCollater
from rnaglib.tasks.RNA_VS.evaluate import run_virtual_screen
from rnaglib.tasks.RNA_VS.ligands import MolGraphEncoder


class VirtualScreening:

    name = "rna_vs"

    def __init__(self, root, ligand_framework='dgl', recompute=False):
        self.root = root
        self.recompute = recompute
        self.ligand_framework = ligand_framework

        # If not present, dump RNA and molecules as graphs
        self.build_dataset()
        script_dir = os.path.dirname(__file__)
        json_dump = os.path.join(script_dir, "data/dataset_as_json.json")
        whole_data = json.load(open(json_dump, 'r'))
        self.trainval_groups, self.test_groups = whole_data["trainval"], whole_data["test"]

        # Get data splits
        train_val_cut = int(0.9 * len(self.trainval_groups))
        train_groups_keys = set(np.random.choice(list(self.trainval_groups.keys()), size=train_val_cut, replace=False))
        self.train_groups = {k: v for k, v in self.trainval_groups.items() if k in train_groups_keys}
        self.val_groups = {k: v for k, v in self.trainval_groups.items() if k not in train_groups_keys}

        self.ligand_encoder = MolGraphEncoder(framework=ligand_framework,
                                              cache_path=os.path.join(self.root, f'ligands_{self.ligand_framework}.p'))

    def build_dataset(self):
        # check if dataset exists and load
        if not os.path.exists(os.path.join(self.root, 'graphs')) or self.recompute:
            dump_rna_jsons(root=self.root, recompute=self.recompute)
        if not os.path.exists(os.path.join(self.root, f'ligands_{self.ligand_framework}.p')) or self.recompute:
            precompute_ligand_graphs(root=self.root, recompute=self.recompute, framework=self.ligand_framework)



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
    from rnaglib.transforms import GraphRepresentation

    # Create a task
    root = "../../data/tasks/rna_vs"
    ef_task = VirtualScreening(root)

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
