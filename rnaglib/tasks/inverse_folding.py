from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import DasSplitter

from rnaglib.utils import load_index
import pandas as pd
import ast
import os


class gRNAde(RNAClassificationTask):
    target_var = "nt_code"  # in rna graph
    input_var = "nt_code"  # this is wrong and should be the graph. needs rework of task superclass

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def sequence_recovery(predictions, target_sequence):
        # predictions are a tensor of designed sequences with shape `(n_samples, seq_len)`
        recovery = predictions.eq(target_sequence).float().mean(dim=1).cpu().numpy()
        return recovery

    def evaluate(self, data, predictions):
        sequence_recovery_rates = []
        for pred, true in zip(predictions, data.y):
            result = self.sequence_recovery(true, pred)
            sequence_recovery_rates.append(result)
        average_srr = sum(sequence_recovery_rates) / len(sequence_recovery_rates)

        # at one point, evaluate on these rnas:['1CSL', '1ET4', '1F27', '1L2X', '1LNT', '1Q9A', '1U8D', '1X9C', '1XPE', '2GCS', '2GDI', '2OEU', '2R8S', '354D'] to compare to gRNAde

        return average_srr

    def default_splitter(self):
        return DasSplitter()
        # SingleStateSplit
        # MultiStateSplit

    def build_dataset(self, root):
        # load metadata from gRNAde if it fails, print link
        try:
            current_dir = os.path.dirname(__file__)
            metadata = pd.read_csv(os.path.join(current_dir, 'data/gRNAde_metadata.csv'))
        except FileNotFoundError:
            print(
                'Download the metadata from https://drive.google.com/file/d/1lbdiE1LfWPReo5VnZy0zblvhVl5QhaF4/ and place it in the ./data dir')

        # generate list
        rnas_keep = []

        for sample in metadata['id_list']:
            per_sample_list = ast.literal_eval(sample)
            rnas_keep.extend(per_sample_list)
        # remove extra info from strings
        rnas_keep = [x.split('_')[0] for x in rnas_keep]

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             redundancy='all',
                             rna_filter=lambda x: x.graph['pdbid'][0] in rnas_keep)
        return dataset
