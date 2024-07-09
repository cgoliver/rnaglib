from rnaglib.data_loading import RNADataset
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter

from rnaglib.utils import load_index
import requests

class ProteinBindingDetection(RNAClassificationTask):
    target_var = "binding_protein"  # graph level attribute
    input_var = "nt_code"  # node level attribute

    def __init__(self, root, splitter=None, **kwargs):
        self.ribosomal_rnas = self.get_ribosomal_rnas()
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            rna_id = graph.split(".")[0]
            if "node_" + self.target_var in graph_attrs and rna_id not in self.ribosomal_rnas:
                rnas_keep.append(rna_id)

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep
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
