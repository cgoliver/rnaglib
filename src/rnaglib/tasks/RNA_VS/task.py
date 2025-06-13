import os

import json
import networkx as nx
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader

from rnaglib.dataset import RNADataset
from rnaglib.tasks import Task
from rnaglib.transforms import Representation, FeaturesComputer
from rnaglib.dataset_transforms import RandomSplitter, NameSplitter, Collater
from rnaglib.tasks.RNA_VS import build_data
from rnaglib.tasks.RNA_VS.ligands import LigandRepresentation, TestLigandRepresentation


class VirtualScreening(Task):
    input_var = "nt_code"
    target_var = "dummy"
    name = "rna_vs"
    version = "2.0.2"
    default_metric = "auroc"

    def __init__(self, ligand_framework="pyg", **kwargs):
        self.ligand_framework = ligand_framework
        meta = {"multi_label": False, 'graph_level': True}
        super().__init__(additional_metadata=meta, **kwargs)

    def process(self) -> RNADataset:
        """"
        Creates the task-specific dataset.

        :return: the task-specific dataset
        :rtype: RNADataset
        """

        # check if dataset exists and load
        build_data.dump_rna_jsons(root=self.root, recompute=self.recompute, version=self.version)
        if not os.path.exists(os.path.join(self.root, f'ligands_{self.ligand_framework}.p')) or self.recompute:
            build_data.precompute_ligand_graphs(root=self.root, recompute=self.recompute,
                                                framework=self.ligand_framework)
        self.load_groups()

        dataset = RNADataset(dataset_path=self.dataset_path, in_memory=self.in_memory)
        dataset.add_representation(InPocketRepresentation())
        return dataset

    def load_groups(self):
        """
        The data is extracted from the RNAmigos repo directly, and preprocessed in build_data.

        It is then stored as a json, and stores ids of RNA pockets as keys of a dictionary.
        It also stores the center of these pockets as a binary mask.
        These pockets are split into training and testing systems.

        Values of the pocket dictionary contain lists of active and inactive ligands for each RNA pocket.

        :return:
        """
        script_dir = os.path.dirname(__file__)
        json_dump = os.path.join(script_dir, "data/dataset_as_json.json")
        whole_data = json.load(open(json_dump, 'r'))
        trainval_groups, test_groups = whole_data["trainval"], whole_data["test"]
        self.trainval_groups, self.test_groups = trainval_groups, test_groups
        return trainval_groups, test_groups

    def post_process(self):
        pass

    def get_task_vars(self) -> FeaturesComputer:
        """Specifies the `FeaturesComputer` object of the tasks which defines the features which have to be added to the RNAs
        (graphs) and nucleotides (graph nodes)

        :return: the features computer of the task
        :rtype: FeaturesComputer
        """
        return FeaturesComputer(nt_features=self.input_var,
                                nt_targets=self.target_var)

    @property
    def default_splitter(self):
        """Returns the splitting strategy to be used for this specific task.

        Here, we follow the splits proposed in RNAmigos2

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        if self.debug:
            return RandomSplitter()
        else:
            # Get data splits
            train_val_cut = int(0.9 * len(self.trainval_groups))
            train_groups_keys = set(
                np.random.choice(list(self.trainval_groups.keys()), size=train_val_cut, replace=False))
            val_groups = [k for k in self.trainval_groups.keys() if k not in train_groups_keys]
            return NameSplitter(train_names=list(train_groups_keys),
                                val_names=val_groups,
                                test_names=list(self.test_groups.keys()))

    def load(self):
        """
        We need to adapt the loading a bit so that the groups are loaded again
        :return:
        """
        dataset, metadata, (train_ind, val_ind, test_ind) = super().load()
        self.dataset.add_representation(InPocketRepresentation())
        self.load_groups()
        return self.dataset, metadata, (train_ind, val_ind, test_ind)

    def set_datasets(self, recompute=True):
        """
        This is a specific for this task.
        Since we do not only make inference on an RNA, but also on one or several ligands, we frame these annotations
        as additional, specific, stateful RNA representations.
        Moreover, we need a different format for training on pairs (rna, ligand) and
         for testing on pairs (rna, list of ligands).
        Thus, we need to include those representations in the datasets at construction time.
        :param recompute:
        :return:
        """
        super().set_datasets(recompute=recompute)
        trainval_ligand_rep = LigandRepresentation(framework=self.ligand_framework,
                                                   groups=self.trainval_groups,
                                                   root=self.root)
        self.train_dataset.add_representation(trainval_ligand_rep)
        self.val_dataset.add_representation(trainval_ligand_rep)
        test_rep = TestLigandRepresentation(framework=self.ligand_framework,
                                            groups=self.test_groups,
                                            root=self.root)
        self.test_dataset.add_representation(test_rep)
        return self.train_dataset, self.val_dataset, self.test_dataset

    def set_loaders(self, recompute=True, **dataloader_kwargs):
        """
        We need to override because we have a test collater (since train and test representations are not the same)
        """
        self.set_datasets(recompute=recompute)

        # If no collater is provided we need one
        if dataloader_kwargs is None:
            dataloader_kwargs = {"collate_fn": Collater(self.train_dataset)}
        if "collate_fn" not in dataloader_kwargs:
            collater = Collater(self.train_dataset)
            dataloader_kwargs["collate_fn"] = collater

        # Now build the loaders
        self.train_dataloader = DataLoader(dataset=self.train_dataset, **dataloader_kwargs)
        dataloader_kwargs["shuffle"] = False
        self.val_dataloader = DataLoader(dataset=self.val_dataset, **dataloader_kwargs)
        collater = Collater(self.test_dataset)
        dataloader_kwargs["collate_fn"] = collater
        dataloader_kwargs["batch_size"] = 1
        self.test_dataloader = DataLoader(dataset=self.test_dataset, **dataloader_kwargs)

    def evaluate(self, model, loader=None):
        """
        Run_virtual_screen.

        :param model: trained affinity prediction model
        :param loader: Loader of VirtualScreenDataset object
        :returns efs: list of efs, one for each pocket in the dataset
        """
        dataloader = loader if loader is not None else self.test_dataloader


        def mean_active_rank(scores, is_active):
            scores = (scores - scores.min()) / (scores.max() - scores.min())
            fpr, tpr, thresholds = metrics.roc_curve(is_active, scores, drop_intermediate=True)
            return metrics.auc(fpr, tpr)

        efs = list()
        failed_set = set()
        print(f"Doing VS on {len(dataloader)} pockets.")
        for i, data in enumerate(dataloader):
            if not i % 20:
                print(f"Done {i}/{len(dataloader)}")

            ligands = data['ligands']["ligands"][0]
            actives = data['ligands']["actives"][0]
            if ligands.batch_size < 10:
                print(f"Skipping pocket{i}, not enough decoys")
                continue

            pocket = data['graph']
            in_pocket = torch.tensor(data['in_pocket'])
            pocket.in_pocket = in_pocket

            scores = model.predict_ligands(pocket, ligands)[:, 0].numpy()
            efs.append(mean_active_rank(scores, actives))
        if len(failed_set) > 0:
            print(f"VS failed on {failed_set}")
        print('Mean EF :', np.mean(efs))
        return efs


class InPocketRepresentation(Representation):
    """
    Nodes close to the center of the pocket are flagged with a Boolean
    This representation ensures that this is taken into account.
    """
    name = 'in_pocket'

    def __call__(self, rna_graph, features_dict):
        in_pocket = nx.get_node_attributes(rna_graph, 'in_pocket')
        in_pocket_flag = [flag for node, flag in sorted(in_pocket.items())]
        return in_pocket_flag

    def batch(self, samples):
        return [x for y in samples for x in y]
