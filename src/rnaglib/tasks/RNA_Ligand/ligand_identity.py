import os

import pandas as pd
from networkx import set_node_attributes

from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter
from rnaglib.data_loading import RNADataset
from rnaglib.encoders import OneHotEncoder
from rnaglib.transforms import FeaturesComputer, NTSubgraphTransform, LigandAnnotator, NameFilter


class LigandNTFilter(NTSubgraphTransform):

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/gmsm_dataset.csv"))
    nodes_keep = set(data.nid.values)

    def node_filter(self, node, ndata):
        if node in self.nodes_keep:
            return True
        return False


class GMSM(RNAClassificationTask):
    input_var = "nt_code"
    target_var = "ligand_code"

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/gmsm_dataset.csv"))
    rnas_keep = set([id[0] for id in data.nid.str.split(".")])
    nodes_keep = set(data.nid.values)
    mapping = {i: i for i in range(len(data.label.unique()))}

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    def default_splitter(self):
        return RandomSplitter()

    def _annotator(self, x):
        ligand_codes = {
            node: int(self.data.loc[self.data.nid == node, "label"].values[0])
            # remove .values[0] when playing with _nt_filter
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, ligand_codes, "ligand_code")
        return x

    def _nt_filter(self, x):
        wrong_nodes = [node for node in x if node not in self.nodes_keep]
        x.remove_nodes_from(wrong_nodes)
        return [x]

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(
            nt_targets=self.target_var,
            custom_encoders_targets={self.target_var: OneHotEncoder(mapping=self.mapping)},
        )

    def process(self):
        dataset = RNADataset.from_database(
            annotator=self._annotator,
            nt_filter=self._nt_filter,
            rna_filter=lambda x: x.graph["pdbid"][0].lower() in self.rnas_keep,
            all_rnas=[name + ".json" for name in self.rnas_keep],  # for testing [0:10]
            redundancy="all",
        )
        return dataset
