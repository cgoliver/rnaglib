import os

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.encoders import MultiLabelOneHotEncoder
from rnaglib.transforms import ComposeFilters, FeaturesComputer
from rnaglib.utils.rfam_utils import get_frequent_go_pdbsel
from rnaglib.utils import dump_json


class RNAGo(RNAClassificationTask):
    """
    Predict the Rfam family of a given RNA chain. This is a multi-class classification task.
     Of course, this task is solved by definition since families are constructed using covariance models.
     However, it can still test the ability of a model to capture characteristic structural features from 3D.
    """

    input_var = "nt_code"  # node level attribute
    target_var = "go_terms"  # graph level attribute

    def __init__(self, **kwargs):
        super().__init__(multi_label=True, **kwargs)

    def get_task_vars(self):
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var: MultiLabelOneHotEncoder(self.metadata["label_mapping"])}, )

    def process(self):
        # Fetch task level filters
        rna_filter = ComposeFilters(self.filters_list)

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
            for pdbsel in lines['pdbsel'].unique():
                pdb = pdbsel.split('_')[0]
                # Subset the graph on the RFAM labeled part.
                # RFAM has a weird numbering convention: they give chain ids that seem like a range
                # and not the actual numbering. Hence, if you have residues [110, 111... 160], the RFAM
                # numbering can look like 2-16. I assume this is residues 111-125.
                _, chain, start, end = pdbsel.split('_')
                pdb_chain_numbers = [node_name for node_name in list(sorted(rna['rna'].nodes())) if
                                     node_name.startswith(f'{pdb}.{chain}')]
                chunk_nodes = pdb_chain_numbers[int(start) - 1: int(end)]
                subgraph = rna_graph.subgraph(chunk_nodes).copy()
                subgraph.name = pdbsel

                # Get the corresponding GO-terms for this RFAM selection
                # Needs a bit of caution because one pdbsel could have more than one rfam_id
                rfams_pdbsel = lines.loc[lines['pdbsel'] == pdbsel]['rfam_acc'].values
                go_terms = [go for rfam_id in rfams_pdbsel for go in rfam_go_mapping[rfam_id]]
                subgraph.graph['go_terms'] = list(set(go_terms))
                
                # Finally, apply quality filters
                if len(subgraph) < 5 or len(subgraph.edges()) < 5:
                    continue
                # A feature dict (including structure path) is needed for the filtering
                chunk_dict = {k: v for k, v in rna.items() if k != 'graph'}
                chunk_dict['graph'] = subgraph
                if rna_filter.forward(chunk_dict):
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
