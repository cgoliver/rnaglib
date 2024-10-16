from networkx import set_node_attributes

from rnaglib.data_loading.create_dataset import (
    annotator_add_embeddings,
    nt_filter_split_chains,
)
from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer
from rnaglib.splitters import RandomSplitter, SPLITTING_VARS, default_splitter_tr60_tr18
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.utils import load_index
from rnaglib.encoders import BoolEncoder
from rnaglib.transforms import ResidueAttributeFilter
from rnaglib.transforms import PDBIDNameTransform
from rnaglib.transforms import BindingSiteAnnotator
from rnaglib.transforms import NameFilter

class BenchmarkBindingSiteDetection(ResidueClassificationTask):
    target_var = "binding_site"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
    
    '''
    @property
    def default_splitter(self):
        return default_splitter_tr60_tr18()
    '''
    #TODO: Implement the default splitter
    
    def process(self) -> RNADataset:
        rnas = RNADataset(debug=False, redundancy='all', rna_id_subset=SPLITTING_VARS['PDB_TO_CHAIN_TR60_TE18'].keys())
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        # TODO: remove wrong chains using  SPLITTING_VARS["PDB_TO_CHAIN_TR60_TE18"]
        rnas = BindingSiteAnnotator()(dataset)
        rnas = PDBIDNameTransform()(rnas)
        dataset = RNADataset(rnas=[r["rna"] for r in rnas]) 
        return dataset

    #TODO Implement addition of RNA-FM embeddings, if requested

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(
            nt_features=["nt_code"],
            nt_targets=self.target_var,
            custom_encoders={self.target_var: BoolEncoder()},
        )

class BindingSiteDetection(ResidueClassificationTask):
    #TODO: The more logical target variable is binding_small-molecule, but not discussed yet.
    target_var = "binding_site"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def process(self) -> RNADataset:
        rnas = BindingSiteAnnotator()(RNADataset(debug=self.debug))
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        rnas = ResidueAttributeFilter(
            attribute=self.target_var, value_checker=lambda val: val == True
        )(dataset)
        rnas = PDBIDNameTransform()(rnas)

        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(
            nt_features=["nt_code"],
            nt_targets=self.target_var,
            custom_encoders={self.target_var: BoolEncoder()},
        )

'''
class BenchmarkBindingSiteDetection(ResidueClassificationTask):
    input_var = "nt_code"
    target_var = "binding_site"
    rnaskeep = SPLITTING_VARS["ID_TR60_TE18"]
    rna_id_to_chains = SPLITTING_VARS["PDB_TO_CHAIN_TR60_TE18"]

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def default_splitter(self):
        return default_splitter_tr60_tr18()

    def _nt_filter(self, x):
        yield from nt_filter_split_chains(x, self.rna_id_to_chains)

    def _annotator(self, x):
        binding_sites = {
            node: (
                not (
                    nodedata.get("binding_small-molecule", None) is None
                    and nodedata.get("binding_ion", None) is None
                )
            )
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, binding_sites, "binding_site")

        # Add RNA-FM embeddings
        annotator_add_embeddings(x)
        return x

    def build_dataset(self):
        features_computer = FeaturesComputer(
            nt_features=self.input_var,
            custom_encoders_targets={self.target_var: BoolEncoder()},
            extra_useful_keys=["embeddings"],
        )
        dataset = RNADataset.from_database(
            features_computer=features_computer,
            dataset_path=self.dataset_path,
            nt_filter=self._nt_filter,
            annotator=self._annotator,
            all_rnas_db=[name[:-1] + ".json" for name in self.rnaskeep],
            redundancy="all",
            recompute=self.recompute,
        )
        return dataset


class BindingSiteDetection(ResidueClassificationTask):
    target_var = "binding_small-molecule"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            if "node_" + self.target_var in graph_attrs:
                rnas_keep.append(graph.split(".")[0])

        features_computer = FeaturesComputer(
            nt_features=self.input_var, nt_targets=self.target_var
        )
        dataset = RNADataset.from_database(
            features_computer=features_computer,
            rna_filter=lambda x: x.graph["pdbid"][0].lower() in rnas_keep,
        )

        return dataset
    
'''

