from networkx import (
    set_node_attributes,
)  # check whether needed after rna-fm integration

from rnaglib.data_loading.create_dataset import (
    annotator_add_embeddings,
    nt_filter_split_chains,
)  # check whether needed after rna-fm integration

from rnaglib.data_loading import RNADataset
from rnaglib.transforms import FeaturesComputer
from rnaglib.splitters import SPLITTING_VARS, default_splitter_tr60_tr18
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.encoders import BoolEncoder
from rnaglib.transforms import ResidueAttributeFilter
from rnaglib.transforms import PDBIDNameTransform, ChainNameTransform
from rnaglib.transforms import BindingSiteAnnotator
from rnaglib.transforms import ChainFilter


class BenchmarkBindingSiteDetection(ResidueClassificationTask):
    target_var = "binding_site"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    @property
    def default_splitter(self):
        return default_splitter_tr60_tr18()

    def process(self) -> RNADataset:
        rnas = RNADataset(
            debug=False,
            redundancy="all",
            rna_id_subset=SPLITTING_VARS["PDB_TO_CHAIN_TR60_TE18"].keys(),
        )
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        rnas = ChainFilter(SPLITTING_VARS["PDB_TO_CHAIN_TR60_TE18"])(dataset)
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        rnas = BindingSiteAnnotator()(dataset)
        rnas = ChainNameTransform()(rnas)
        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        return dataset

    # TODO Implement addition of RNA-FM embeddings, if requested

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(
            nt_features=self.input_var,
            nt_targets=self.target_var,
            custom_encoders={self.target_var: BoolEncoder()},
        )


class BindingSiteDetection(ResidueClassificationTask):
    target_var = "binding_small-molecule"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def process(self) -> RNADataset:
        dataset = RNADataset(debug=self.debug)
        rnas = ResidueAttributeFilter(
            attribute=self.target_var, value_checker=lambda val: val is not None
        )(dataset)

        rnas = PDBIDNameTransform()(rnas)

        dataset = RNADataset(rnas=[r["rna"] for r in rnas])
        return dataset

    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)
