import os
import sys

import numpy as np
import random
from torch.utils.data import Dataset

from rnaglib.data_loading import RNADataset, Collater


class VSCollater:
    def __init__(self, vs_dataset):
        self.rna_collater = Collater(vs_dataset.rna_dataset)
        self.ligand_collater = vs_dataset.ligand_embedder.collate_fn

    def __call__(self, samples):
        batch = dict()
        # TODO make this better by falling back to default collate_fn on other keys
        for key in samples[0].keys():
            if key == 'pocket':
                batch[key] = self.rna_collater([sample[key] for sample in samples])
            elif key == 'ligand':
                batch[key] = self.ligand_collater([sample[key] for sample in samples])
            else:
                batch[key] = [sample[key] for sample in samples]
        return batch


class VSRNADataset(Dataset):
    def __init__(self, groups, ligand_embedder, dataset_path, decoy_mode='pdb', features_computer=None, **kwargs):
        # To load RNAs. We don't pass the features_encoder for graph construction, to avoid discarding other fields
        self.rna_dataset = RNADataset.from_database(dataset_path=dataset_path, features_computer=None, **kwargs)
        self.rna_dataset.features_computer = features_computer
        self.name_id_mapping = {rna['rna'].graph['pocket_name']: idx for idx, rna in enumerate(iter(self.rna_dataset))}

        # To load ligands
        self.ligand_embedder = ligand_embedder

        # To get the right data
        self.groups = groups
        self.sorted_groups = np.sort(list(groups.keys()))
        self.decoy_mode = decoy_mode

    def add_inpocket_flag(self, graph, rna):
        # TODO also add for point representation and pyg graphs
        import dgl
        pocket_graph_dgl = dgl.from_networkx(nx_graph=rna,
                                             node_attrs=['in_pocket'])
        graph.ndata['in_pocket'] = pocket_graph_dgl.ndata['in_pocket']

    def __len__(self):
        return len(self.groups)

    def get_pocket_representations(self, group_rep):
        pocket_representations = self.rna_dataset[self.name_id_mapping[group_rep]]
        self.add_inpocket_flag(pocket_representations['graph'], pocket_representations['rna'])
        return pocket_representations

    def __getitem__(self, idx):
        raise NotImplementedError


class VSRNATrainDataset(VSRNADataset):
    def __getitem__(self, idx):
        # Get pocket representation
        group_rep = self.sorted_groups[idx]
        pocket_representations = self.get_pocket_representations(group_rep)

        # Pick either active or inactive at random, then sample a ligand of the right group and encode it
        group = self.groups[group_rep]
        is_active = random.random() > 0.5
        ligands_to_use = group['actives'] if is_active else group[f'{self.decoy_mode}_decoys']
        ligand = ligands_to_use[random.randint(0, len(ligands_to_use) - 1)]
        ligand_graph = self.ligand_embedder.smiles_to_graph_one(ligand)
        return {'group_rep': group_rep,
                'pocket': pocket_representations,
                'ligand': ligand_graph,
                'active': is_active}


class VSRNATestDataset(VSRNADataset):
    def __getitem__(self, idx):
        # Get pocket representation
        group_rep = self.sorted_groups[idx]
        pocket_representations = self.get_pocket_representations(group_rep)

        # Pick either active or inactive at random, then sample a ligand of the right group and encode it
        actives = self.groups[group_rep]['actives']
        inactives = self.groups[group_rep][f'{self.decoy_mode}_decoys']
        active_ligands = self.ligand_embedder.smiles_to_graph_list(actives)
        inactive_ligands = self.ligand_embedder.smiles_to_graph_list(inactives)
        return {'group_rep': group_rep,
                'pocket': pocket_representations,
                'active_ligands': active_ligands,
                'inactive_ligands': inactive_ligands}


if __name__ == '__main__':
    import pickle
    from rnaglib.transforms import FeaturesComputer
    from rnaglib.transforms import GraphRepresentation
    from rnaglib.tasks.RNA_VS.build_data import build_data
    from rnaglib.tasks.RNA_VS.ligands import MolGraphEncoder

    script_dir = os.path.dirname(__file__)
    json_dump = os.path.join(script_dir, "../data/rna_vs/dataset.p")
    trainval_groups, test_groups = pickle.load(open(json_dump, 'rb'))

    root = "../data/rna_vs"
    build_data(root=root, recompute=False)
    ligand_encoder = MolGraphEncoder(cache_path=os.path.join(root, 'ligands.p'))
    features_computer = FeaturesComputer(nt_features=['nt_code'])
    representations = GraphRepresentation(framework='dgl')
    rna_dataset_args = {'representations': representations, 'features_computer': features_computer}
    train_dataset_debug = VSRNATrainDataset(groups=test_groups,
                                            ligand_embedder=ligand_encoder,
                                            dataset_path=os.path.join(root, 'graphs'),
                                            **rna_dataset_args)
    test_dataset_debug = VSRNATrainDataset(groups=test_groups,
                                           ligand_embedder=ligand_encoder,
                                           dataset_path=os.path.join(root, 'graphs'),
                                           **rna_dataset_args)

    for i, elt in enumerate(train_dataset_debug):
        a = 1
        # if i > 3:
        #     break
        if not i % 50:
            print(i, len(train_dataset_debug))
