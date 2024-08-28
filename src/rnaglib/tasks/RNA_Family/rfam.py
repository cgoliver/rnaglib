import networkx as nx
from networkx import set_node_attributes
from tqdm import tqdm
import requests

from rnaglib.data_loading import RNADataset, FeaturesComputer
from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter, get_ribosomal_rnas
from rnaglib.utils import load_index, BoolEncoder, OneHotEncoder


class RNAFamilyTask(RNAClassificationTask):
    input_var = "dummy"  # node level attribute
    target_var = 'rfam'  # graph level attribute

    def __init__(self, root, max_size: int = 200, splitter=None, **kwargs):
        self.max_size = max_size
        self.ribosomal_rnas = get_ribosomal_rnas()
        if 'debug' in kwargs:
            self.rnas_keep, self.families = self.compute_rfam_families(debug=kwargs['debug'])
        else:
            self.rnas_keep, self.families = self.compute_rfam_families()
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    def get_rfam(self, pdb_id):
        base_url = "https://www.ebi.ac.uk/pdbe/api/nucleic_mappings/rfam/"
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

    def compute_rfam_families(self, debug=False):
        graph_index = load_index()
        rnas_keep = []
        families = []

        todo = list(graph_index.items())
        if debug:
            todo = todo[:20]

        for graph, graph_attrs in tqdm(todo, desc="Querying Rfam"):
            rna_id = graph.split(".")[0]
            # Selection of ribosomal DNA removed since I cannot access the server endpoint
            # if "node_" + self.target_var in graph_attrs and rna_id not in self.ribosomal_rnas:

            # Keep only RNAS with an assigned rfam family
            rfam = self.get_rfam(rna_id)
            if rfam:
                families.append(rfam)
                rnas_keep.append(rna_id)
        return rnas_keep, families

    @property
    def output_mapping(self):
        # Create mapping of Rfam families for OneHotEncoder
        return {family: i for i, family in enumerate(sorted(list(set(self.families))))}

    def default_splitter(self):
        return RandomSplitter()

    def _rna_filter(self, g: nx.Graph):
        """ Remove RNAs that don't have a family annotation or are too large"""

        if g.graph['pdbid'][0].lower() not in self.rnas_keep:
            return False
        if len(g.nodes()) > self.max_size:
                return False
        return True

    def _nt_filter(self, x):
        chains = list(set(s[s.index('.'):s.rindex('.') + 1] for s in x.nodes))

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
                    batch_nodes = sorted_nodes[i:i + 50]
                    batch_subgraph = subgraph.subgraph(batch_nodes).copy()
                    yield batch_subgraph

    def _annotator(self, x):
        accession_number = self.get_rfam(x.graph['pdbid'][0])
        rfam = {node: accession_number for node, nodedata in x.nodes.items()}
        set_node_attributes(x, rfam, 'rfam')

        dummy = {node: 1 for node, nodedata in x.nodes.items()}
        set_node_attributes(x, dummy, 'dummy')
        return x

    def build_dataset(self, root):
        # Create dataset
        features_computer = FeaturesComputer(custom_encoders_features={self.input_var: BoolEncoder()},
                                             custom_encoders_targets={
                                                 self.target_var: OneHotEncoder(mapping=self.output_mapping)})
        dataset = RNADataset.from_database(features_computer=features_computer,
                                       annotator=self._annotator,
                                       rna_filter=self._rna_filter)
        return dataset
