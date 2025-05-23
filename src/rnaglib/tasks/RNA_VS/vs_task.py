import os
import numpy as np

from rnaglib.dataset import RNADataset
from rnaglib.tasks import Task
from rnaglib.transforms import FeaturesComputer
from rnaglib.dataset_transforms import RandomSplitter, NameSplitter
from rnaglib.tasks.RNA_VS import build_data


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
        return RNADataset(dataset_path=self.dataset_path)

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
        similarity-based splitting relying on clustering which could be refined into a sequencce- or structure-based clustering
        using distance_name argument

        :return: the default splitter to be used for the task
        :rtype: Splitter
        """
        if self.debug:
            return RandomSplitter()
        else:
            import json
            script_dir = os.path.dirname(__file__)
            json_dump = os.path.join(script_dir, "data/dataset_as_json.json")
            whole_data = json.load(open(json_dump, 'r'))
            trainval_groups, test_groups = whole_data["trainval"], whole_data["test"]

            # Get data splits
            train_val_cut = int(0.9 * len(trainval_groups))
            train_groups_keys = set(
                np.random.choice(list(trainval_groups.keys()), size=train_val_cut, replace=False))
            val_groups = [k for k in trainval_groups.keys() if k not in train_groups_keys]
            return NameSplitter(train_names=list(train_groups_keys),
                                val_names=val_groups,
                                test_names=list(test_groups.keys()))
