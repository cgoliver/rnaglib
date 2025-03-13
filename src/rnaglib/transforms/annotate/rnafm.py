import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List

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
    """Use the RNA-FM model to compute residue-level embeddings.
    Make sure rna-fm is installed by running ``pip install rna-fm``.
    Sets a node attribute to `'rnafm'` with a numpy array of the resulting
    embedding. Go `here <https://github.com/ml4bio/RNA-FM>`_ for the RNA-FM
    source code.

    :param chunking_strategy: how to process sequences longer than 1024. ``'simple'`` just
    splits into non-overlapping segments.
    :param chunk_size: size of chunks to use (default is 512)
    :param cache_path: a directory containing pre-computed npz embeddings
    :param expand_mean: True

    .. note::
        Maximum size for basic RNA-FM model is 1024. If sequence is larger
        than 1024 we apply ``'chunking_strategy'`` to process the sequence.
    """

    name = "rnafm"
    encoder = ListEncoder(640)

    def __init__(
        self, chunking_strategy: str = "simple", chunk_size: int = 512, cache_path=None, expand_mean=True, verbose=False, **kwargs
    ):
        # Load RNA-FM model
        self.model, self.alphabet = fm.pretrained.rna_fm_t12()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.model.eval()
        self.cache_path = cache_path
        self.expand_mean = expand_mean
        self.verbose = verbose
        super().__init__(**kwargs)

    def basic_chunking(self, seq):
        return [seq[i : i + self.chunk_size] for i in range(0, len(seq), self.chunk_size)]

    def chunk(self, seq_data: List[Tuple]) -> List[Tuple]:
        """Apply a chunking strategy to sequences longer than 1024."""

        chunked = {}
        for chain_id, (seq, nodes) in seq_data.items():
            if self.chunking_strategy == "simple":
                chunks = self.basic_chunking(list(zip(seq, nodes)))
            for i, chunk in enumerate(chunks):
                nodelist = [n for _, n in chunk]
                seq = "".join([s for s, _ in chunk])
                chunked[chain_id + "_" + str(i)] = (seq, nodelist)
        return chunked

    def forward(self, rna_dict: Dict) -> Dict:
        chain_seqs = get_sequences(rna_dict["rna"], verbose=self.verbose)

        # Try to load the embs if possible.
        if self.cache_path is not None:
            if not self.quiet:
                print(f"Loading embeddings from {self.cache_path}.")
            chains = list(chain_seqs.keys())
            for chain in chains:
                embs_path = Path(self.cache_path) / f"{chain}.npz"
                if embs_path.exists():
                    embs = np.load(embs_path)
                    # If they are complete, remove from the chains to do and put in the graph
                    if len(chain_seqs[chain][0]) == len(embs):
                        nx.set_node_attributes(rna_dict["rna"], embs, self.name)
                        chain_seqs.pop(chain)
            if len(chain_seqs) == 0:
                return rna_dict

        # Otherwise make the actual computations
        # Prepare data
        seq_data = self.chunk(chain_seqs)
        input_seqs = [(chain_id, seq) for chain_id, (seq, _) in seq_data.items()]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(input_seqs)

        # Extract embeddings (on CPU)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][12]

        all_embs = []
        for i, (_, (seq, node_ids)) in enumerate(seq_data.items()):
            Z = token_embeddings[i, : len(seq)]
            emb_dict = {n: Z[i].clone().detach() for i, n in enumerate(node_ids)}
            for i, n in enumerate(node_ids):
                z = Z[i].clone().detach()
                rna_dict["rna"].nodes[n][self.name] = list(z.numpy())
                all_embs.append(z)

        # Add mean embedding to missing nodes, if self.expand mean
        if self.expand_mean:
            existing_nodes, _ = list(zip(*nx.get_node_attributes(rna_dict["rna"], self.name).items()))
            missing_nodes = set(rna_dict["rna"].nodes()) - set(existing_nodes)
            if len(missing_nodes) > 0:
                embs = torch.stack(all_embs, dim=0)
                mean_emb = torch.mean(embs, dim=0)
                missing_embs = {node: mean_emb.tolist() for node in missing_nodes}
                nx.set_node_attributes(rna_dict["rna"], name=self.name, values=missing_embs)
        return rna_dict
