import networkx as nx
from networkx import set_node_attributes
from tqdm import tqdm
import requests

from rnaglib.data_loading import RNADataset, FeaturesComputer
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter, get_ribosomal_rnas
from rnaglib.utils import load_index, BoolEncoder, OneHotEncoder
from rnaglib.transforms import ChainSplitTransform, RfamTransform, ChainNameTransform, RNAAttributeFilter


class RNAFamilyTask(RNAClassificationTask):
    """ Predict the Rfam family of a given RNA chain.
    This is a multi-class classification task. Of course, this task is solved
    by definition since families are constructted algorithmically using covariance models. However it can still test the ability of a model to capture characteristic
    structural features from 3D.
    """
    target_var = 'rfam'  # graph level attribute

    def __init__(self, root, max_size: int = 200, splitter=None, **kwargs):
        self.max_size = max_size
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass


    def build_dataset(self, root):
        # Create dataset
        full_dataset = RNADataset(debug=True)
        # compute rfam annotation, only keep ones with an Rfam annot.
        tr_rfam = RfamTransform()
        rnas = tr_rfam(full_dataset)
        rnas = RNAAttributeFilter(attribute=tr_rfam.name)(rnas)
        # compute one-hot mapping of labels
        labels = sorted(list(set([r['rna'].graph['rfam'] for r in rnas])))
        rfam_mapping = {rfam: i for i, rfam in enumerate(labels)}
        tr_rfam.encoder = OneHotEncoder(rfam_mapping)
        # split by chain
        tr_split = ChainSplitTransform()
        tr_chain_name = ChainNameTransform()
        chains = [tr_chain_name(chain_subg)['rna'] for r in rnas \
                                            for chain_subg in tr_split(r)]

        ft = FeaturesComputer(rna_targets=[tr_rfam.name], transforms=tr_rfam)
        new_dataset = RNADataset(rnas=chains, features_computer=ft)

        return new_dataset
