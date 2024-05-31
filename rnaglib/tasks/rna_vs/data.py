import os
import sys

import numpy as np
import random
from torch.utils.data import Dataset

if __name__ == "__main__":
    sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..")] + sys.path

from rnaglib.data_loading import RNADataset, Collater


class VSCollater:
    def __init__(self, vs_dataset):
        self.rna_collater = Collater(vs_dataset.rna_dataset)
        self.ligand_collater = vs_dataset.ligand_embedder.collate_fn

    def __call__(self, samples):
        batch = dict()
        for key in samples[0].keys():
            if key == 'pocket':
                batch[key] = self.rna_collater([sample[key] for sample in samples])
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

    def add_inpocket_flag(self, graph, rna):
        # TODO also add for point representation and pyg graphs
        import dgl
        pocket_graph_dgl = dgl.from_networkx(nx_graph=rna,
                                             node_attrs=['in_pocket'])
        graph.ndata['in_pocket'] = pocket_graph_dgl.ndata['in_pocket']

    def __getitem__(self, idx):
        group_rep = self.sorted_groups[idx]
        pocket_representations = self.rna_dataset[self.name_id_mapping[group_rep]]
        self.add_inpocket_flag(pocket_representations['graph'], pocket_representations['rna'])
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
        return {'group_rep': group_rep,
                'pocket': pocket_representations,
                'ligand': ligand_graph,
                'active': is_active}


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

    def add_inpocket_flag(self, graph, rna):
        # TODO also add for point representation and pyg graphs
        import dgl
        pocket_graph_dgl = dgl.from_networkx(nx_graph=rna,
                                             node_attrs=['in_pocket'])
        graph.ndata['in_pocket'] = pocket_graph_dgl.ndata['in_pocket']

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        group_rep = self.sorted_groups[i]
        pocket_representations = self.rna_dataset[self.name_id_mapping[group_rep]]
        self.add_inpocket_flag(pocket_representations['graph'], pocket_representations['rna'])

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
    from rnaglib.representations.graph import GraphRepresentation
    from rnaglib.tasks.rna_vs.build_data import build_data
    from rnaglib.tasks.rna_vs.ligands import MolGraphEncoder

    script_dir = os.path.dirname(__file__)
    json_dump = os.path.join(script_dir, "../../data/tasks/rna_vs/dataset_as_json.json")
    trainval_groups, test_groups = pickle.load(open(json_dump, 'rb'))

    root = "../../data/tasks/rna_vs"
    build_data(root=root, recompute=False)
    ligand_encoder = MolGraphEncoder(cache_path=os.path.join(root, 'ligands.p'))
    representations = [GraphRepresentation(framework='pyg')]
    rna_dataset_args = {'representations': representations, 'nt_features': 'nt_code'}
    test_dataset = VSRNATrainDataset(groups=test_groups,
                                     ligand_embedder=ligand_encoder,
                                     saved_dataset=os.path.join(root, 'graphs'),
                                     **rna_dataset_args)

    for i, elt in enumerate(test_dataset):
        a = 1
        # if i > 3:
        #     break
        if not i % 50:
            print(i, len(test_dataset))
