from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import (
    ChainSplitTransform,
    RfamTransform,
    ChainNameTransform,
    RNAAttributeFilter,
)
from rnaglib.transforms import FeaturesComputer


class RNAFamily(RNAClassificationTask):
    """Predict the Rfam family of a given RNA chain.
    This is a multi-class classification task. Of course, this task is solved
    by definition since families are constructed algorithmically using covariance models. However it can still test the ability of a model to capture characteristic
    structural features from 3D.
    """

    target_var = "rfam"  # graph level attribute
    input_var = "nt_code"  # node level attribute

    def __init__(self, max_size: int = 200, **kwargs):
        self.max_size = max_size
        super().__init__(**kwargs)

    def get_task_vars(self):
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(self.metadata["label_mapping"])}, )

    def process(self):
        # Create dataset
        full_dataset = RNADataset(debug=self.debug)
        # compute rfam annotation, only keep ones with an Rfam annot.
        tr_rfam = RfamTransform(parallel=True)
        rnas = tr_rfam(full_dataset)
        rnas = list(RNAAttributeFilter(attribute=tr_rfam.name, value_checker=lambda val: val is not None)(rnas))
        # compute one-hot mapping of labels
        labels = sorted(set([r["rna"].graph["rfam"] for r in rnas]))
        rfam_mapping = {rfam: i for i, rfam in enumerate(labels)}
        self.metadata["label_mapping"] = rfam_mapping

        # split by chain
        rnas = ChainSplitTransform()(rnas)
        rnas = ChainNameTransform()(rnas)

        new_dataset = RNADataset(rnas=list((r["rna"] for r in rnas)))
        return new_dataset
