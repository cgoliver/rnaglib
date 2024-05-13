from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import BenchmarkBindingSiteSplitter, RandomSplitter

from networkx import set_node_attributes

class BenchmarkLigandBindingSiteDetection(ResidueClassificationTask):
    target_var = 'binding_small-molecule' #"binding_site" needs to be replaced once dataset modifiable.
    input_var = "nt_code"

    TR60 = ['3sktA','5u3gB','5j02A','2yieZ','2fcyA','3gx3A','4nybA','1hr2A','4mgmB','3oxeB',
            '1y90B','2quwB','4megB','4lvxA','4rgeB','4pcjA','3c44A','5o69A','2lwkA','3vrsA',
            '2g5kA','5fj1C','5d5lD','4frgX','1ylsB','3q50A','4xw7A','2ktzA','4qlmA','3fu2A',
            '5dh8B','3meiB','6fz0A','2mxsA','2nokC','1ajuA','1fypA','4k31C','2nokB','1ntbA',
            '3bnqC','5vciA','3q3zV','1uudB','1byjA','1lvjA','1utsB','1qd3A','1arjN','2l8hA',
            '6hagA','1yrjA','1tobA','1f1tA','3tzrA','4qjhC','2kgpA','1rawA','1ehtA','1nbkA','1ei2A']
    TE18 = ['2pwtA','5v3fA','379dB','5bjoE','4pqvA','430dA','1nemA','1q8nA','1f1tA','2jukA',
            '4yazR','364dC','6ez0A','2tobA','1ddyA','1fmnA','2misA','4f8uB']
    # The papers also contains a second test dataset not implemented here.
    rnas_keep = TR60 + TE18

    
    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass
    pass

    def default_splitter(self):
        train_names = [name[:-1] + '.json' for name in self.TR60[:-6]] 
        val_names = [name[:-1] + '.json' for name in self.TR60[-6:]] 
        test_names = [name[:-1] + '.json' for name in self.TE18]
        return BenchmarkBindingSiteSplitter(train_names, val_names, test_names)

    def build_dataset(self):
        
        self.rnas_keep = [name[:-1] + '.json' for name in self.rnas_keep]

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             all_graphs=self.rnas_keep,
                             redundancy ='all'
                             )
       
        #Note: The below will only work once dataset is editable.
        '''
        # Removing all nodes not part of the chainns defined in the training and testing set.
        for i in range(len(dataset[i])):
            chain = rnas_keep[i][-1]
            chainstr = '.' + chain + '.'
            wrong_chain_nodes = [nodename for nodename in list(dataset[i]['rna'].nodes()) if chainstr not in nodename]
            dataset[i]['rna'].remove_nodes_from(wrong_chain_nodes)

        # Creating a new binary node attribute determining whether a residues is part of a binding site or not.
        for i in range(len(dataset)):
            binding_sites = {
            node: (not (nodedata.get("binding_small-molecule", None) is None and nodedata.get("binding_ion",None) is None))
            for node, nodedata in dataset[i]['rna'].nodes.items()
            }
            set_node_attributes(dataset[i]['rna'], binding_sites, 'binding_site')  
        '''
        return dataset