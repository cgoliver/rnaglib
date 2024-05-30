import os
import sys

import numpy as np
import pickle
import random
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..")] + sys.path

from rnaglib.data_loading import RNADataset, Collater
from rnaglib.tasks.rna_ef.build_data import build_data
from rnaglib.tasks.rna_ef.ligands import MolGraphEncoder


class VSCollater:
    def __init__(self, vs_dataset):
        self.rna_collater = Collater(vs_dataset.rna_dataset)
        self.ligand_collater = vs_dataset.ligand_embedder.collate_fn

    def collate(self, samples):
        batch = dict()
        for key in samples[0].keys():
            if key == 'pocket':
                batch[key] = self.rna_collater.collate([sample[key] for sample in samples])
            elif key in ('ligand', "active_ligands", "inactive_ligands"):
                batch[key] = self.ligand_collater([sample[key] for sample in samples])
            else:
                batch[key] = [sample[key] for sample in samples]
        return batch


# class VSTrainSampler(Sampler):
#     def __init__(self, num_pos, num_neg):
#         super().__init__(data_source=None)
#         self.num_pos = num_pos
#         self.num_neg = num_neg
#
#     def __iter__(self):
#         # Sample active/inactive indices and put them all in a complex list
#         selected_pos = np.random.randint(0, self.num_pos)
#         selected_neg = np.random.randint(0, self.num_neg)
#         pocket_idx = np.array(range(len(selected_pos)))
#         actives = np.ones(len(pocket_idx))
#         inactives = np.zeros(len(pocket_idx))
#         pos_stacked = np.stack((pocket_idx, selected_pos, actives), axis=-1)
#         neg_stacked = np.stack((pocket_idx, selected_neg, inactives), axis=-1)
#         systems = np.concatenate((pos_stacked, neg_stacked))
#         np.random.shuffle(systems)
#         yield from systems
#
#     def __len__(self) -> int:
#         return len(self.num_pos) * 2

class VSRNATrainDataset(Dataset):
    def __init__(self, groups, ligand_embedder, saved_dataset, decoy_mode='pdb', **kwargs):
        # To load RNAs:
        self.rna_dataset = RNADataset(saved_dataset=saved_dataset, **kwargs)
        self.name_id_mapping = {rna['rna'].graph['pocket_name']: idx for idx, rna in enumerate(iter(self.rna_dataset))}

        # To load ligands
        self.ligand_embedder = ligand_embedder

        # To get the right pairs
        # The difficulty here arises from the need to train on pocket/ligand pairs obtained
        # from a pool of possible actives/inactives
        self.groups = groups
        self.sorted_groups = np.sort(list(groups.keys()))
        num_pos, num_neg = [], []
        for group_rep in self.sorted_groups:
            actives = groups[group_rep]['actives']
            inactives = groups[group_rep][f'{decoy_mode}_decoys']
            num_pos.append(len(actives))
            num_neg.append(len(inactives))
        self.num_pos = np.array(num_pos)
        self.num_neg = np.array(num_neg)
        self.decoy_mode = decoy_mode

    def __len__(self):
        return len(self.num_pos * 2)

    def __getitem__(self, idx):
        group_rep = self.sorted_groups[idx]
        pocket_representations = self.rna_dataset[self.name_id_mapping[group_rep]]
        group = self.groups[group_rep]
        actives = group['actives']
        inactives = group[f'{self.decoy_mode}_decoys']
        is_active = random.random() > 0.5
        ligands_to_use = actives if is_active else inactives
        ligand_id = random.randint(0, len(ligands_to_use) - 1)
        ligand = ligands_to_use[ligand_id]
        ligand_graph = self.ligand_embedder.smiles_to_graph_one(ligand)

        # Sampler-based solution
        # group_idx, ligand_id, is_active = idx
        # ligand = actives[ligand_id] if is_active else inactives[ligand_id]
        return {'pocket': pocket_representations, 'ligand': ligand_graph, 'active': is_active}


class VSRNADataset(Dataset):
    def __init__(self, groups, ligand_embedder, saved_dataset, decoy_mode='pdb', **kwargs):
        # To load RNAs:
        self.rna_dataset = RNADataset(saved_dataset=saved_dataset, **kwargs)
        self.name_id_mapping = {rna['rna'].graph['pocket_name']: idx for idx, rna in enumerate(iter(self.rna_dataset))}

        # To load ligands
        self.ligand_embedder = ligand_embedder

        # To get the right data
        self.groups = groups
        self.sorted_groups = np.sort(list(groups.keys()))
        self.decoy_mode = decoy_mode

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        group_rep = self.sorted_groups[i]
        pocket_representations = self.rna_dataset[self.name_id_mapping[group_rep]]

        actives = self.groups[group_rep]['actives']
        inactives = self.groups[group_rep][f'{self.decoy_mode}_decoys']
        active_ligands = self.ligand_embedder.smiles_to_graph_list(actives)
        inactive_ligands = self.ligand_embedder.smiles_to_graph_list(inactives)
        return {'pocket': pocket_representations,
                'active_ligands': active_ligands,
                'inactive_ligands': inactive_ligands}


class EFTask:
    script_dir = os.path.dirname(__file__)
    json_dump = os.path.join(script_dir, "../../data/tasks/rna_ef/dataset_as_json.json")
    trainval_groups, test_groups = pickle.load(open(json_dump, 'rb'))

    def __init__(self, root, recompute=False, **kwargs):
        self.root = root
        self.recompute = recompute
        self.build_dataset()
        train_cut = int(0.8 * len(self.trainval_groups))
        train_groups_keys = set(np.random.choice(list(self.trainval_groups.keys()), size=train_cut, replace=False))
        self.train_groups = {k: v for k, v in self.trainval_groups.items() if k in train_groups_keys}
        self.val_groups = {k: v for k, v in self.trainval_groups.items() if k not in train_groups_keys}
        self.ligand_encoder = MolGraphEncoder(cache_path=os.path.join(self.root, 'ligands.p'))

    def build_dataset(self):
        # check if dataset exists and load
        if not os.path.exists(os.path.join(self.root, 'graphs')) or self.recompute:
            build_data(root=self.root)

    def get_split_datasets(self, dataset_kwargs=None):
        train_dataset = VSRNATrainDataset(groups=self.train_groups,
                                          ligand_embedder=self.ligand_encoder,
                                          saved_dataset=os.path.join(self.root, 'graphs'),
                                          **dataset_kwargs)
        val_dataset = VSRNATrainDataset(groups=self.val_groups,
                                        ligand_embedder=self.ligand_encoder,
                                        saved_dataset=os.path.join(self.root, 'graphs'),
                                        **dataset_kwargs)
        test_dataset = VSRNADataset(groups=self.test_groups,
                                    ligand_embedder=self.ligand_encoder,
                                    saved_dataset=os.path.join(self.root, 'graphs'),
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
            dataloader_kwargs = {'collate_fn': VSCollater(self.train_dataset).collate}
        if 'collate_fn' not in dataloader_kwargs:
            collater = VSCollater(self.train_dataset).collate
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


if __name__ == '__main__':
    from rnaglib.representations.graph import GraphRepresentation

    root = "../../data/tasks/rna_ef"
    ef_task = EFTask(root)
    representations = [GraphRepresentation()]
    rna_dataset_args = {'representations': representations}
    rna_loader_args = {'batch_size': 2}
    # train_dataset, val_dataset, test_dataset = ef_task.get_split_datasets(rna_dataset_args)
    train_dataset, val_dataset, test_dataset = ef_task.get_split_loaders(dataset_kwargs=rna_dataset_args,
                                                                         dataloader_kwargs=rna_loader_args)

    for i, elt in enumerate(train_dataset):
        # print(elt)
        a = 1
        # if i > 3:
        #     break
        if not i % 50:
            print(i, len(train_dataset))

    for i, elt in enumerate(test_dataset):
        # print(elt)
        a = 1
        # if i > 3:
        #     break
        if not i % 10:
            print(i, len(test_dataset))
