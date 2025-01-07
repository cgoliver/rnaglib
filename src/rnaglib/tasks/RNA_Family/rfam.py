import os
from rnaglib.utils import dump_json

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.encoders import IntMappingEncoder
from rnaglib.transforms import (
    ChainSplitTransform,
    RfamTransform,
    ChainNameTransform,
    RNAAttributeFilter,
)
from rnaglib.utils.rfam_utils import get_frequent_go_pdbsel
from rnaglib.transforms import FeaturesComputer


class RNAFamily(RNAClassificationTask):
    """
    Predict the Rfam family of a given RNA chain. This is a multi-class classification task.
     Of course, this task is solved by definition since families are constructed using covariance models.
     However, it can still test the ability of a model to capture characteristic structural features from 3D.
    """

    input_var = "nt_code"  # node level attribute
    target_var = "go_terms"  # graph level attribute

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_task_vars(self):
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(self.metadata["label_mapping"])}, )

    def process(self):
        # Get initial mapping files:
        df, rfam_go_mapping = get_frequent_go_pdbsel()
        if self.debug:
            df = df.sample(n=50, random_state=0)
        dataset = RNADataset(redundancy='all', in_memory=self.in_memory, rna_id_subset=df['pdb_id'].unique())

        # Create dataset
        # Run through database, applying our filters
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            rna_graph = rna['rna']
            lines = df.loc[df['pdb_id'] == rna_graph.name]
            for pdbsel, pdb, rfam_id in list(lines.values):
                # subset the graph. RFAM has a weird numbering convention: they give chain ids that seem like a range
                # and not the actual numbering. Hence, if you have residues [110, 111... 160], the RFAM
                # numbering can look like 2-16. I assume this is residues 111-125.
                _, chain, start, end = pdbsel.split('_')
                pdb_chain_numbers = [node_name for node_name in list(sorted(rna['rna'].nodes())) if
                                     node_name.startswith(f'{pdb}.{chain}')]
                chunk_nodes = pdb_chain_numbers[int(start) - 1: int(end)]
                subgraph = rna_graph.subgraph(chunk_nodes).copy()
                subgraph.name = pdbsel
                subgraph.go_terms = rfam_go_mapping[rfam_id]
                if len(subgraph) < 5 or len(subgraph.edges()) < 5:
                    continue
                if self.in_memory:
                    all_rnas.append(subgraph)
                else:
                    all_rnas.append(subgraph.name)
                    dump_json(os.path.join(self.dataset_path, f"{subgraph.name}.json"), subgraph)
        if self.in_memory:
            dataset = RNADataset(rnas=all_rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_rnas)
        # compute one-hot mapping of labels
        unique_gos = sorted({go for system_gos in rfam_go_mapping.values() for go in system_gos})
        rfam_mapping = {rfam: i for i, rfam in enumerate(unique_gos)}
        self.metadata["label_mapping"] = rfam_mapping
        return dataset


class RNAFamilyOld(RNAClassificationTask):
    """
    Predict the Rfam family of a given RNA chain. This is a multi-class classification task.
     Of course, this task is solved by definition since families are constructed using covariance models.
     However, it can still test the ability of a model to capture characteristic structural features from 3D.
    """

    target_var = "rfam"  # graph level attribute
    input_var = "nt_code"  # node level attribute

    def __init__(self, max_size: int = 200, **kwargs):
        self.max_size = max_size
        super().__init__(**kwargs)

    def get_task_vars(self):
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: IntMappingEncoder(self.metadata["label_mapping"])}, )

    def process(self):
        # Create dataset
        full_dataset = RNADataset(debug=self.debug)
        # Compute rfam annotation, only keep ones with an Rfam annot.
        tr_rfam = RfamTransform(parallel=True)
        rnas = tr_rfam(full_dataset)
        rnas = list(RNAAttributeFilter(attribute=tr_rfam.name, value_checker=lambda val: val is not None)(rnas))
        # compute one-hot mapping of labels
        labels = sorted(set([r["rna"].graph["rfam"] for r in rnas]))
        rfam_mapping = {rfam: i for i, rfam in enumerate(labels)}
        self.metadata["label_mapping"] = rfam_mapping

        # split by chain
        rnas = ChainSplitTransform()(rnas)
        rnas = ChainNameTransform()(rnas)

        new_dataset = RNADataset(rnas=list((r["rna"] for r in rnas)))
        return new_dataset
