from rnaglib.data_loading import RNADataset, FeaturesComputer
from rnaglib.tasks import ResidueClassificationTask, RNAClassificationTask
from rnaglib.dataset_transforms import RandomSplitter

from rnaglib.utils import load_index
import requests


class BindingSiteDetection(ResidueClassificationTask):
    target_var = "binding_small-molecule"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            if "node_" + self.target_var in graph_attrs:
                rnas_keep.append(graph.split(".")[0])

        features_computer = FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)
        dataset = RNADataset.from_args(features_computer=features_computer,
                                       rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep,
                                       )

        return dataset


class ProteinBindingSiteDetection(ResidueClassificationTask):
    target_var = "binding_protein"
    input_var = "nt_code"

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


class ProteinBindingDetection(RNAClassificationTask):
    target_var = "proteins"  # graph level attribute
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


class BindingDetection(RNAClassificationTask):
    target_var = "ligands"  # graph level attribute
    input_var = "nt_code"  # node level attribute

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        dataset = RNADataset(rna_targets=[self.target_var],
                             rna_features=[self.input_var],
                             rna_filter=lambda x: x.graph['pdbid'][0].lower()
                             )
        return dataset


class ChemicalModification(ResidueClassificationTask):
    target_var = "is_modified"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    pass

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            if "node_" + self.target_var in graph_attrs:
                rnas_keep.append(graph.split(".")[0])

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep,
                             )

        return dataset
