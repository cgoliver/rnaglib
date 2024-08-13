from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter

from rnaglib.utils import load_index, BoolEncoder, OneHotEncoder
import requests

from networkx import set_node_attributes

from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter

from rnaglib.utils import load_index
import requests

class ProteinBindingDetection(RNAClassificationTask):
    target_var = 'rfam'   # graph level attribute
    input_var = "dummy"  # node level attribute

    mapping = {}

    def __init__(self, root, splitter=None, **kwargs):
        self.ribosomal_rnas = self.get_ribosomal_rnas()
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def default_splitter(self):
        return RandomSplitter()

    def _nt_filter(self, x):
        chains = list(set(s[s.index('.'):s.rindex('.')+1] for s in x.nodes))

        def get_node_number(node):
            return int(node.split('.')[-1])

        for chain in chains:
            wrong_chain_nodes = [node for node in list(x) if chain not in node]
            subgraph = x.copy()
            subgraph.remove_nodes_from(wrong_chain_nodes)

            if len(subgraph) < 100:
                yield subgraph
            else:
                # Sort nodes based on the number at the end
                sorted_nodes = sorted(subgraph.nodes, key=get_node_number)

                # We split the chains into chains of size 50 or more (for the last chain)
                # This allows graph level classification on attributes that may always be true for a whole RNA molecule but not for parts of it
                for i in range(0, len(sorted_nodes), 50):
                    batch_nodes = sorted_nodes[i:i+50]
                    batch_subgraph = subgraph.subgraph(batch_nodes).copy()
                    yield batch_subgraph

    import requests
    from typing import List, Dict
    import json

    def get_rfam(self, pdb_id):
        base_url = "https://www.ebi.ac.uk/pdbe/api/nucleic_mappings/rfam/"
        results = {}
        url = f"{base_url}{pdb_id.lower()}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            data = response.json()

            if pdb_id.lower() in data and 'Rfam' in data[pdb_id.lower()]:
                # Get the first Rfam accession number found
                rfam_acc = next(iter(data[pdb_id.lower()]['Rfam'].keys()), 'N/A')
            else:
                rfam_acc = None
        except requests.RequestException as e:
            rfam_acc = None
        except (KeyError, IndexError, ValueError) as e:
            rfam_acc = None
        return rfam_acc

    def _annotator(self, x):
        accession_number = self.get_rfam(x.graph['pdbid'][0])
        rfam = {
            node: accession_number
            for node, nodedata in x.nodes.items()}
        set_node_attributes(x, rfam, 'rfam')
        
        dummy = {
            node: 1
            for node, nodedata in x.nodes.items()}
        set_node_attributes(x, dummy, 'dummy')

        return x

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        families = []

        for graph, graph_attrs in graph_index.items():
            rna_id = graph.split(".")[0]
            # Selection of ribosomal DNA removed since I cannot access the server endpoint
            #if "node_" + self.target_var in graph_attrs and rna_id not in self.ribosomal_rnas:
            
            # Keep only RNAS with an assigned rfam family
            rfam = self.get_rfam(rna_id)
            if rfam:
                families.append(rfam)
                rnas_keep.append(rna_id)



        # Create mapping of Rfam families for OneHotEncoder
        self.mapping = {family: i for i, family in enumerate(set(families))}
        # Create dataset
        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             custom_encoders_targets= {self.target_var: OneHotEncoder(mapping=self.mapping)},
                             custom_encoders_features={self.input_var: BoolEncoder()},  
                             annotator=self._annotator,
                             rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep, 
                             )
        return dataset

    def get_ribosomal_rnas(self):
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "struct_keywords.pdbx_keywords",
                    "operator": "contains_phrase",
                    "value": "ribosome"
                }
            },
            "return_type": "entry",
            "request_options": {
                "return_all_hits": True
            }
        }
        response = requests.post(url, json=query)
        if response.status_code == 200:
            data = response.json()
            ribosomal_rnas = [result['identifier'] for result in data['result_set']]
            return ribosomal_rnas
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            print(response.text)
            return []