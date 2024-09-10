import os
import json
from typing import Dict

import torch
import numpy as np
import networkx as nx
import torch

try:
    import fm
except ModuleNotFoundError:
    print("Please make sure rna-fm is installed with pip install rna-fm")
    sys.exit()

from rnaglib.transforms import Transform
from rnaglib.encoders import ListEncoder
from rnaglib.algorithms import get_sequences

class RNAFMTransform(Transform):
    """ Use the RNA-FM model to compute residue-level embeddings.
    Make sure rna-fm is installed by running ``pip install rna-fm``.
    Sets a node attribute to `'rnafm'` with a numpy array of the resulting
    embedding. Go `here <https://github.com/ml4bio/RNA-FM>`_ for the RNA-FM
    source code.
    """
    name  = 'rnafm'
    encoder = ListEncoder(640)

    def __init__(self, **kwargs):
        # Load RNA-FM model
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        super().__init__(**kwargs)

    def forward(self, rna_dict: Dict) -> Dict:
        # Prepare data
        seq_data = get_sequences(rna_dict['rna'])
        input_seqs = [(chain_id, seq) for chain_id, (seq, _) in seq_data.items()]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(input_seqs)

        # Extract embeddings (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][12]

        for i, (chain_id, (seq, node_ids)) in enumerate(seq_data.items()):
            Z = token_embeddings[i,:len(seq)]
            emb_dict = {n:list(Z[i].detach().numpy()) for i,n in enumerate(node_ids)}
            nx.set_node_attributes(rna_dict['rna'], emb_dict, self.name)

        return rna_dict
