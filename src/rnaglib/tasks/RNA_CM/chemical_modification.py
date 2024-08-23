from rnaglib.data_loading import RNADataset, FeaturesComputer
from rnaglib.data_loading.create_dataset import nt_filter_split_chains, annotator_add_embeddings
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import RandomSplitter, SPLITTING_VARS, get_ribosomal_rnas, default_splitter_tr60_tr18

from rnaglib.utils import load_index


class BenchmarkChemicalModificationEmbeddings(ResidueClassificationTask):
    input_var = "nt_code"
    target_var = 'is_modified'
    rnaskeep = set(SPLITTING_VARS['TR60'] + SPLITTING_VARS['TE18'])

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    def default_splitter(self):
        return default_splitter_tr60_tr18()

    def _nt_filter(self, x):
        # TODO do we need to ?
        yield from nt_filter_split_chains(x, self.rna_id_to_chains)

    def _annotator(self, x):
        raise NotImplementedError
        # TODO...
        annotator_add_embeddings(x)

    def build_dataset(self, root):
        features_computer = FeaturesComputer(nt_targets=[self.target_var],
                                             nt_features=[self.input_var])
        dataset = RNADataset.from_database(features_computer=features_computer,
                                           rna_filter=lambda x: x.graph['pdbid'][0].lower() in [name[:-1] for name in
                                                                                                self.rnaskeep],
                                           nt_filter=self._nt_filter,
                                           annotator=self._annotator,
                                           redundancy='all',
                                           all_rnas=[name[:-1] + '.json' for name in self.rnaskeep]
                                           # for increased loading speed
                                           )
        return dataset


class ChemicalModification(ResidueClassificationTask):
    """ Residue-level binary classification task to predict whether or not a given
    residue is chemically modified.
    """
    input_var = "nt_code"
    target_var = "is_modified"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            if "node_" + self.target_var in graph_attrs:
                rnas_keep.append(graph.split(".")[0])

        features_computer = FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)
        dataset = RNADataset.from_database(features_computer=features_computer,
                                           rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep)

        return dataset
