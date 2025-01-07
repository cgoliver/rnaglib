import os

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import (
    ChainSplitTransform,
    RfamTransform,
    ChainNameTransform,
    RNAAttributeFilter,
    ComposeFilters
)
from rnaglib.transforms import FeaturesComputer
from rnaglib.utils import dump_json



class RNAFamily(RNAClassificationTask):
    """Predict the Rfam family of a given RNA chain.
    This is a multi-class classification task. Of course, this task is solved
    by definition since families are constructed algorithmically using covariance models. However it can still test the ability of a model to capture characteristic
    structural features from 3D.
    """

    target_var = "rfam"  # graph level attribute
    input_var = "nt_code"  # node level attribute

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_task_vars(self):
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(self.metadata["label_mapping"])}
        )

    def process(self):
        # init filters
        rna_filter = ComposeFilters(self.filters_list)
        # Initialize dataset with in_memory=False to avoid loading everything at once
        dataset = RNADataset(debug=self.debug, in_memory=False)
        tr_rfam = RfamTransform(parallel=True)
        rfam_filter = RNAAttributeFilter(attribute=tr_rfam.name, value_checker=lambda val: val is not None)
        chain_split = ChainSplitTransform()
        chain_annotator = ChainNameTransform()

        # Run through database, applying our filters
        all_rnas = []
        rfams_set = set()
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if rna_filter.forward(rna) and len(rna["rna"].nodes())>0:
                annotated_rna = tr_rfam(rna)
                if rfam_filter.forward(annotated_rna):
                    rfams_set.add(annotated_rna["rna"].graph["rfam"])
                    for chain in chain_split(annotated_rna):
                        annotated_chain = chain_annotator(chain)["rna"]
                        if self.in_memory:
                            all_rnas.append(annotated_chain)
                        else:
                            all_rnas.append(annotated_chain.name)
                            dump_json(
                                os.path.join(self.dataset_path, f"{annotated_chain.name}.json"),
                                annotated_chain,
                            )
        # compute one-hot mapping of labels
        labels = sorted(rfams_set)
        rfam_mapping = {rfam: i for i, rfam in enumerate(labels)}
        self.metadata["label_mapping"] = rfam_mapping

        if self.in_memory:
            dataset = RNADataset(rnas=all_rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_rnas)

        return dataset
