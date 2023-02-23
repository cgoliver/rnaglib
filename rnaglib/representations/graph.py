import torch
import networkx as nx

from rnaglib.representations import Representation
from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.utils import fix_buggy_edges

class GraphRepresentation(Representation):
    """ Converts and RNA into a graph """
    def __init__(self,
                 clean_edges=True,
                 framework='nx',
                 edge_map=GRAPH_KEYS['edge_map'][TOOL],
                 etype_key='LW',
                 **kwargs):

        self.clean_edges = clean_edges

        self.etype_key = etype_key
        self.edge_map = edge_map

        super().__init__(framework=framework, frameworks=['nx', 'dgl', 'pyg'], **kwargs)
        pass

    def call(self, rna_dict, features_dict):
        print(f"COnveritng to {self.framework}")
        if self.clean_edges:
            base_graph = fix_buggy_edges(graph=rna_dict['rna'])
        else:
            base_graph = rna_dict['rna']

        if self.framework == 'nx':
            return self.to_nx(base_graph, features_dict)
        if self.framework == 'dgl':
            return self.to_dgl(base_graph, features_dict)
        if self.framework == 'pyg':
            return self.to_pyg(base_graph, features_dict)

    def name(self):
        return "graph"

    def to_nx(self, graph, features_dict):
        pass

    def to_dgl(self, graph, features_dict):
        import dgl
        # Get Edge Labels
        edge_type = {(u,v): self.edge_map[data[self.etype_key]] for u,v, data in graph.edges(data=True)}
        nx.set_edge_attributes(graph, name='edge_type', values=edge_type)
        nx.set_node_attributes(graph, name='features', values=features_dict['nt_features'])
        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=graph,
                                  edge_attrs=['edge_type'],
                                  node_attrs=['features'])


        return g_dgl

    def to_pyg(self, graph, features_dict):
        from torch_geometric.data import Data

        # for some reason from_networkx is not working so doing by hand
        # not super efficient at the moment
        node_map = {n:i for i,n in enumerate(sorted(graph.nodes()))}
        x = [features_dict['nt_features'][n] for n in sorted(graph.nodes())]
        edge_index = [[node_map[u], node_map[v]] for u, v in sorted(graph.edges())]
        edge_attrs = [self.edge_map[data[self.etype_key]] for u, v, data in sorted(graph.edges(data=True))]
        return Data(x=x, edge_attr=edge_attrs, edge_index=edge_index)

