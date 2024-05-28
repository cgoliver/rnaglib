from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import BenchmarkBindingSiteSplitter, RandomSplitter

from networkx import set_node_attributes


class BenchmarkLigandBindingSiteDetection(ResidueClassificationTask):
    target_var = 'binding_small-molecule'  # "binding_site" needs to be replaced once dataset modifiable.
    input_var = "nt_code"

    TR60 = ['3sktA', '5u3gB', '5j02A', '2yieZ', '2fcyA', '3gx3A', '4nybA', '1hr2A', '4mgmB', '3oxeB',
            '1y90B', '2quwB', '4megB', '4lvxA', '4rgeB', '4pcjA', '3c44A', '5o69A', '2lwkA', '3vrsA',
            '2g5kA', '5fj1C', '5d5lD', '4frgX', '1ylsB', '3q50A', '4xw7A', '2ktzA', '4qlmA', '3fu2A',
            '5dh8B', '3meiB', '6fz0A', '2mxsA', '2nokC', '1ajuA', '1fypA', '4k31C', '1ntbA', '3bnqC',
            '5vciA', '3q3zV', '1uudB', '1byjA', '1lvjA', '1utsB', '1qd3A', '1arjN', '2l8hA', '6hagA',
            '1yrjA', '1tobA', '1f1tA', '3tzrA', '4qjhC', '2kgpA', '1rawA', '1ehtA', '1nbkA',
            '1ei2A']  # nok is a duplicate. RLBind uses chain C only. ,'2nokB'
    TE18 = ['2pwtA', '5v3fA', '379dB', '5bjoE', '4pqvA', '430dA', '1nemA', '1q8nA', '1f1tA', '2jukA',
            '4yazR', '364dC', '6ez0A', '2tobA', '1ddyA', '1fmnA', '2misA', '4f8uB']
    # The papers also contains a second test dataset not implemented here.
    rnaskeep = TR60 + TE18

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def default_splitter(self):
        train_names = [name[:-1] + '.json' for name in self.TR60[:-6]]
        val_names = [name[:-1] + '.json' for name in self.TR60[-6:]]
        test_names = [name[:-1] + '.json' for name in self.TE18]
        return BenchmarkBindingSiteSplitter(train_names, val_names, test_names)

    def _nt_filter(self, x):
        subgraphs = []
        chains = ['.' + chain[-1] + '.' for chain in self.rnaskeep if chain.startswith(x.graph['pdbid'][0].lower())]
        for chain in chains:
            wrong_chain_nodes = [node for node in list(x) if chain not in node]
            subgraph = x.copy()
            subgraph.remove_nodes_from(wrong_chain_nodes)
            yield subgraph

    def _annotator(self, x):
        binding_sites = {
            node: (not (nodedata.get("binding_small-molecule", None) is None and nodedata.get("binding_ion",
                                                                                              None) is None))
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, binding_sites, 'binding_site')
        return x

    def build_dataset(self):
        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             rna_filter=lambda x: x.graph['pdbid'][0].lower() in [name[:-1] for name in self.rnaskeep],
                             nt_filter=self._nt_filter,
                             annotator=self._annotator,
                             redundancy='all',
                             all_graphs=[name[:-1] + '.json' for name in self.rnaskeep]  # for increased loading speed
                             )
        return dataset
