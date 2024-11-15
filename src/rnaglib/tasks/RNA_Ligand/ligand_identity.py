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

    def process(self):
        features_computer = FeaturesComputer(
            nt_features=self.input_var,
            nt_targets=self.target_var,
            custom_encoders={
                self.target_var: OneHotEncoder(mapping=self.mapping)
            },
        )
        rna_filter = NameFilter(
            names = self.rnas_keep
        )
        rnas = RNADataset(debug=False, redundancy='all', rna_id_subset=[name for name in self.rnas_keep], features_computer=features_computer)
        rnas = LigandAnnotator()(rnas)
        rnas = LigandNTFilter()(rnas)
        rnas = rna_filter(rnas)
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        return dataset
    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)

