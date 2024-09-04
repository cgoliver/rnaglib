import networkx as nx
from networkx import set_node_attributes
from tqdm import tqdm
import requests

from rnaglib.data_loading import RNADataset, FeaturesComputer
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter, get_ribosomal_rnas
from rnaglib.utils import load_index, BoolEncoder, OneHotEncoder
from rnaglib.transforms import ChainSplitTransform, RfamTransform


class RNAFamilyTask(RNAClassificationTask):
    target_var = 'rfam'  # graph level attribute

    def __init__(self, root, max_size: int = 200, splitter=None, **kwargs):
        self.max_size = max_size
        self.ribosomal_rnas = get_ribosomal_rnas()
        if 'debug' in kwargs:
            self.rnas_keep, self.families = self.compute_rfam_families(debug=kwargs['debug'])
        else:
            self.rnas_keep, self.families = self.compute_rfam_families()
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass


    def build_dataset(self, root):
        # Create dataset
        full_dataset = RNADataset()
        # compute rfam annotation
        tr_rfam = RfamTransform()
        rnas = [tr_rfam(r) for r in full_dataset]
        # remove rnas without rfam annotation
        rnas = [r for r in dataset if not r['rna'].graph['rfam'] is None]
        # compute one-hot mapping of labels
        rfam_mapping = {rfam: i for i, rfam in sorted(list(set([r['rna']['rfam'] for r in rnas])))}
        tr_rfam.encoder = OneHotEncoder(rfam_mapping)
        # split by chain
        tr_split = ChainSplitTransform()
        chains = [chain_subg for chain_subg in tr_split(r) for r in rnas]

        ft = FeaturesComputer(rna_targets=[tr_rfam.name], transforms=tr_rfam)
        new_dataset = RNADataset(rnas=chains)

        return dataset
