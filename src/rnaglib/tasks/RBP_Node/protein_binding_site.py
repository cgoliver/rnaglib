import numpy as np
from networkx import set_node_attributes

from rnaglib.data_loading.create_dataset import annotator_add_embeddings, nt_filter_split_chains
from rnaglib.data_loading import RNADataset, FeaturesComputer
from rnaglib.splitters import RandomSplitter, SPLITTING_VARS, get_ribosomal_rnas, default_splitter_tr60_tr18
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.utils import load_index
from rnaglib.utils.feature_maps import BoolEncoder


class ProteinBindingSiteDetection(ResidueClassificationTask):
    input_var = "nt_code"
    target_var = "binding_protein"

    def __init__(self, root, splitter=None, **kwargs):
        self.ribosomal_rnas = get_ribosomal_rnas()
        super().__init__(root=root, splitter=splitter, **kwargs)

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            rna_id = graph.split(".")[0]
            if "node_" + self.target_var in graph_attrs and rna_id not in self.ribosomal_rnas:
                rnas_keep.append(rna_id)

        features_computer = FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)
        dataset = RNADataset.from_database(features_computer=features_computer,
                                           rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep)
        return dataset
