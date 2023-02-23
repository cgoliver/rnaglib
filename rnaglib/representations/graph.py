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

        self.frameworks = ['dgl', 'pyg', 'nx']
        self.framework = framework

        self.clean_edges = clean_edges

        self.etype_key = etype_key
        self.edge_map = edge_map

        super().__init__(**kwargs)
        pass

    def call(self, rna_dict):
        if self.clean_edges:
            base_graph = fix_buggy_edges(graph=rna_dict['rna'])
        else:
            base_graph = rna_dict['rna']

        if self.framework == 'nx':
            graph = base_graph
            pass
        elif self.framework == 'dgl':
            graph = self.to_dgl(base_graph, rna_dict)

        rna_dict['graph'] = graph

    def to_dgl(self, graph, rna_dict):
        import dgl
        # Get Edge Labels
        edge_type = {(u,v): self.edge_map[data[self.etype_key]] for u,v, data in graph.edges(data=True)}
        nx.set_edge_attributes(graph, name='edge_type', values=edge_type)
        nx.set_node_attributes(graph, name='features', values=rna_dict['nt_features'])
        # Careful ! When doing this, the graph nodes get sorted.
        g_dgl = dgl.from_networkx(nx_graph=graph,
                                  edge_attrs=['edge_type'],
                                  node_attrs=['features'])


        return g_dgl

    def to_pyg(self, graph):
        pass

