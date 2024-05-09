from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import RandomSplitter

from rnaglib.utils import load_index

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
                rnas_keep.append(graph)

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             all_graphs=rnas_keep
                             )

        return dataset
