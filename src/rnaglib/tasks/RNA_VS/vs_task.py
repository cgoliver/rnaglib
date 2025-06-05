import os
import numpy as np

from rnaglib.dataset import RNADataset
from rnaglib.tasks import Task
from rnaglib.transforms import FeaturesComputer
from rnaglib.dataset_transforms import RandomSplitter, NameSplitter
from rnaglib.tasks.RNA_VS import build_data
from rnaglib.tasks.RNA_VS.data import InPocketRepresentation
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

        dataset = RNADataset(dataset_path=self.dataset_path, in_memory=self.in_memory)
        dataset.add_representation(InPocketRepresentation())
        return dataset

    def load_groups(self):
        import json
        script_dir = os.path.dirname(__file__)
        json_dump = os.path.join(script_dir, "data/dataset_as_json.json")
        whole_data = json.load(open(json_dump, 'r'))
        trainval_groups, test_groups = whole_data["trainval"], whole_data["test"]
        self.trainval_groups, self.test_groups = trainval_groups, test_groups
        return trainval_groups, test_groups

    def load(self):
        dataset, metadata, (train_ind, val_ind, test_ind) = super().load()
        self.dataset.add_representation(InPocketRepresentation())
        self.load_groups()
        return self.dataset, metadata, (train_ind, val_ind, test_ind)

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
        """Returns the splitting strategy to be used for this specific task. Canonical splitter is ClusterSplitter which is a
        similarity-based splitting relying on clustering which could be refined into a sequence or structure-based clustering
        using distance_name argument

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

    def set_datasets(self, recompute=True):
        super().set_datasets(recompute=recompute)
        trainval_ligand_rep = LigandRepresentation(framework=self.ligand_framework,
                                                   groups=self.trainval_groups,
                                                   root=self.root)
        self.train_dataset.add_representation(trainval_ligand_rep)
        self.val_dataset.add_representation(trainval_ligand_rep)
        test_rep = TestLigandRepresentation(framework=self.ligand_framework, groups=self.test_groups)
        self.test_dataset.add_representation(test_rep)
        return self.train_dataset, self.val_dataset, self.test_dataset
