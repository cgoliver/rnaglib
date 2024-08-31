import os
import json

import torch
import numpy as np
import networkx as nx
import torch
import fm

from rnaglib.transforms import Transform
from rnaglib.algorithms import get_sequences

class RNAFMTransform(Transform):
    """" Use the RNA-FM model to compute residue-level embeddings.
    Make sure rna-fm is installed by running ``pip install rna-fm``.
    Sets a node attribute to `'rnafm'` with a numpy array of the resulting
    embedding. Go `here <https://github.com/ml4bio/RNA-FM>`_ for the RNA-FM
    source code.
    """
    def __init__(self):
        # Load RNA-FM model
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

    def forward(self, G: nx.Graph) -> nx.Graph:
        # Prepare data
        data = get_sequences(G)
        input_seqs = [(chain_id, seq) for chain_id, (seq, _) in data.items()]
        print(input_seqs)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(input_seqs)

        # Extract embeddings (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][12]

        for i, (chain_id, (seq, node_ids)) in enumerate(data.items()):
            Z = token_embeddings[i,:len(seq)]
            emb_dict = {n:Z[i].detach().numpy() for i,n in enumerate(node_ids)}
            nx.set_node_attributes(G, emb_dict, 'rnafm')
        pass
