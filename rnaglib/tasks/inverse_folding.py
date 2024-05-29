from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter

from rnaglib.utils import load_index
import requests



class gRNAde(RNAClassificationTask):
    target_var = "sequence" #needs to be replaced with an actual var
    input_var = "coordinates" #needs to be replaced with an actual var

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
        for pred, true in zip(predictions , data.y):
            result = sequence_recovery(true, pred)
            sequence_recovery_rates.append(result)
        average_srr = sum(sequence_recovery_rates) / len(sequence_recovery_rates)
        
        return average_srr
    
    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            if "node_" + self.target_var in graph_attrs:
                rnas_keep.append(graph.split(".")[0])

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep
                             )

        return dataset