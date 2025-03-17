import os

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.encoders import MultiLabelOneHotEncoder
from rnaglib.transforms import FeaturesComputer
from rnaglib.dataset_transforms import ClusterSplitter, CDHitComputer
from rnaglib.utils.rfam_utils import get_frequent_go_pdbsel


class RNAGo(RNAClassificationTask):
    """
    Predict the Rfam family of a given RNA chain. This is a multi-class classification task.
     Of course, this task is solved by definition since families are constructed using covariance models.
     However, it can still test the ability of a model to capture characteristic structural features from 3D.
    """

    input_var = "nt_code"  # node level attribute
    target_var = "go_terms"  # graph level attribute
    name = "rna_go"

    def __init__(self, root, size_thresholds=(15, 500), **kwargs):
        meta = {"task_name": "rna_go", "multi_label":True}
        super().__init__(root=root, additional_metadata=meta, size_thresholds=size_thresholds, **kwargs)

    @property
    def default_splitter(self):
        return ClusterSplitter(similarity_threshold=0.6, distance_name="cd_hit")

    def get_task_vars(self):
        label_mapping = self.metadata["label_mapping"]
        if self.debug:
            label_mapping = {"0000353": 0,
                             "0005682": 1,
                             "0005686": 2,
                             "0005688": 3,
                             "0010468": 4
                             }
        return FeaturesComputer(
            nt_features=self.input_var,
            rna_targets=self.target_var,
            custom_encoders={self.target_var:
                             MultiLabelOneHotEncoder(label_mapping)}, )

    def process(self):
        # Get initial mapping files:
        df, rfam_go_mapping = get_frequent_go_pdbsel()
        if self.debug:
            df = df.sample(n=50, random_state=0)
        dataset = RNADataset(redundancy=self.redundancy, in_memory=self.in_memory, rna_id_subset=df['pdb_id'].unique())

        # Create dataset
        # Run through database, applying our filters
        all_rnas = []
        go_terms_dict = {}
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
                pdb_chain_numbers = [node_name for node_name in list(sorted(rna_graph.nodes())) if
                                     node_name.startswith(f'{pdb}.{chain}')]
                chunk_nodes = pdb_chain_numbers[int(start) - 1: int(end)]
                subgraph = rna_graph.subgraph(chunk_nodes).copy()
                subgraph.name = pdbsel

                # Get the corresponding GO-terms for this RFAM selection
                # Needs a bit of caution because one pdbsel could have more than one rfam_id
                rfams_pdbsel = lines.loc[lines['pdbsel'] == pdbsel]['rfam_acc'].values
                
                #top_rfam_go_mapping = {rfam:[go_term for go_term in rfam_go_mapping[rfam] if go_term not in ['0000373','0003824','0006396','0006617','0009113']] for rfam in rfam_go_mapping}
                #go_terms = [go for rfam_id in rfams_pdbsel for go in top_rfam_go_mapping[rfam_id]]
                go_terms = [go for rfam_id in rfams_pdbsel for go in rfam_go_mapping[rfam_id]]

                # Finally, apply quality filters
                if len(subgraph) < 5 or len(subgraph.edges()) < 5:
                    continue
                # A feature dict (including structure path) is needed for the filtering
                chunk_dict = {k: v for k, v in rna.items() if k != 'graph'}
                chunk_dict['graph'] = subgraph
                if self.size_thresholds is not None:
                    if not self.size_filter.forward(chunk_dict):
                        continue
                for go_term in go_terms:
                    if go_term in go_terms_dict:
                        go_terms_dict[go_term].append(rna_graph.name)
                    else:
                        go_terms_dict[go_term]=[rna_graph.name]

                subgraph.graph['go_terms'] = list(set(go_terms))
                self.add_rna_to_building_list(all_rnas=all_rnas, rna=subgraph)
        dataset = self.create_dataset_from_list(all_rnas)

        go_terms_to_keep = [key for key in go_terms_dict if len(go_terms_dict[key])>60]

        for rna in dataset:
            rna['rna'].graph['go_terms'] = [go_term for go_term in rna['rna'].graph['go_terms'] if go_term in go_terms_to_keep]

        # compute one-hot mapping of labels
        unique_gos = sorted({go for system_gos in rfam_go_mapping.values() for go in system_gos if go in go_terms_to_keep})
        rfam_mapping = {rfam: i for i, rfam in enumerate(unique_gos)}
        self.metadata["label_mapping"] = rfam_mapping
        return dataset

    def post_process(self):
        cd_hit_computer = CDHitComputer(similarity_threshold=0.9)
        self.dataset = cd_hit_computer(self.dataset)
