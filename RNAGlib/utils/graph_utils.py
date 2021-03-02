import sys
import pickle
import os
import itertools

from tqdm import tqdm
import networkx as nx
import numpy as np

faces = ['W', 'S', 'H']
orientations = ['C', 'T']
valid_edges = ['B53'] + [orient + e1 + e2 for e1, e2 in itertools.product(faces, faces) for orient in orientations]

script_dir = os.path.dirname(os.path.realpath(__file__))

graph_dir = os.path.join("..", "data", "graphs", "rna_graphs_nr")
annot_dir = os.path.join("..", "data", "annotated", "all_rna_nr")

if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))


def graph_from_node(node_id,
                    annot_dir=os.path.join(script_dir, '../data/annotated/whole_v4/')):
    """
        Fetch graph from a node id.
    """
    graph_path = os.path.join(annot_dir, node_id[0].replace('.nx', '_annot.p'))
    return pickle.load(open(graph_path, 'rb'))['graph'].to_undirected()


def whole_graph_from_node(node_id, annot_dir=os.path.join(script_dir, graph_dir)):
    """
        Fetch whole graph from a node id.
    """
    if '_' in node_id[0]:
        graph_path = os.path.join(annot_dir, node_id[0].split('_')[0] + '.nx')
    else:
        graph_path = os.path.join(annot_dir, node_id[0])
    return nx.read_gpickle(graph_path)


def induced_edge_filter(G, roots, depth=1):
    """
        Remove edges in G introduced by the induced
        sugraph routine.
        Only keep edges which fall within a single
        node's neighbourhood.

        :param G: networkx subgraph
        :param roots: nodes to use for filtering
        :param depth: size of neighbourhood to take around each node.
        :returns clean_g: cleaned graph
    """
    # a depth of zero does not make sense for this operation as it would remove all edges
    if depth < 1:
        depth = 1
    neighbourhoods = []
    flat_neighbors = set()
    for root in roots:
        root_neighbors = bfs_expand(G, [root], depth=depth)
        neighbourhoods.append(root_neighbors)
        flat_neighbors = flat_neighbors.union(root_neighbors)

    flat_neighbors = list(flat_neighbors)
    subG = G.subgraph(flat_neighbors)
    subG = subG.copy()
    # G_new = G_new.subgraph(flat_neighbors)
    kill = []
    for (u, v) in subG.edges():
        for nei in neighbourhoods:
            if u in nei and v in nei:
                break
        else:
            kill.append((u, v))

    subG.remove_edges_from(kill)
    return subG


def fetch_graph(g_path):
    if g_path.endswith('.p'):
        graph = pickle.load(open(g_path, 'rb'))['graph']
    else:
        graph = nx.read_gpickle(g_path)
    return graph


def annots_from_node(annot_dir, node_id):
    """
        Fetch annots from a node id.
    """
    graph_path = os.path.join(annot_dir, node_id[0].replace('.nx', '_annot.p'))
    return pickle.load(open(graph_path, 'rb'))


def subgraph_clean(G, roots, depth):
    """
        Remove edges from G which aren't in the depth-hop
        neighbourhood of roots.
    """
    pass


def nc_clean_dir(graph_dir, dump_dir):
    """
        Copy graphs from graph_dir to dump_dir but copied graphs are
        trimmed according to `get_nc_nodes_index`.

        `graph_dir` should contain networkx pickles.
    """

    for g in tqdm(os.listdir(graph_dir)):
        G = nx.read_gpickle(os.path.join(graph_dir, g))
        keep_nodes = get_nc_nodes(G)
        print(f">>> kept {len(keep_nodes)} nodes of {len(G.nodes())}.")
        kill_nodes = set(G.nodes()) - keep_nodes
        G.remove_nodes_from(kill_nodes)
        dangle_trim(G)
        if len(G.nodes()) > 4:
            nx.write_gpickle(G, os.path.join(dump_dir, g))
    pass


def get_nc_nodes_index(graph, depth=3):
    """
        Returns indices of nodes in graph list which have a non canonical or
        looping base in their neighbourhood.

        :returns keep: set of nodes in loops or that have a NC.
    """

    keep = []
    for i, node in enumerate(sorted(graph.nodes())):
        if graph.degree(node) == 2:
            keep.append(i)
        elif has_NC_bfs(graph, node, depth=depth):
            keep.append(i)
        else:
            pass
    return keep


def get_nc_nodes(g, depth=4):
    """
        Returns indices of nodes in graph list which have a non canonical or
        looping base in their neighbourhood.

        :returns keep: set of nodes in loops or that have a NC.
    """

    keep = []
    for node in sorted(g.nodes()):
        if g.degree(node) == 2:
            keep.append(node)
        elif has_NC_bfs(g, node, depth=depth):
            keep.append(node)
        else:
            pass
    return set(keep)


def incident_nodes(G, nodes):
    """
        Returns set of nodes in $G \ nodes$ incident to nodes.
        `nodes` is a set
    """
    core = set(nodes)
    hits = set()
    for u, v in G.edges():
        if u in core and v not in core:
            hits.add(v)
        if u not in core and v in core:
            hits.add(u)
    return hits


def get_edge_map(graphs_dir):
    edge_labels = set()
    print("Collecting edge labels.")
    for g in tqdm(os.listdir(graphs_dir)):
        try:
            graph, _, _ = pickle.load(open(os.path.join(graphs_dir, g), 'rb'))
        except:
            print(f"failed on {g}")
            continue
        edges = {e_dict['label'] for _, _, e_dict in graph.edges(data=True)}
        edge_labels = edge_labels.union(edges)

    return {label: i for i, label in enumerate(sorted(edge_labels))}


def reindex_nodes_annot(graph_dir, dump=None):
    """
        Assign a unique id to each node in the graph.
    """
    graphs = []
    offset = 0
    for g in tqdm(sorted(os.listdir(graph_dir))):
        try:
            annot = pickle.load(open(os.path.join(graph_dir, g), 'rb'))
            G = annot['graph']
        except Exception as e:
            print(f"failed on {g}, {e}")
            continue
        G = nx.relabel.convert_node_labels_to_integers(G, first_label=offset, label_attribute='id')
        offset += len(G.nodes())
        if not dump is None:
            annot['graph'] = G
            pickle.dump(annot, open(os.path.join(dump, g), 'wb'))

        graphs.append(G)
    return graphs


def relabel_nodes_annot(graph_dir, dump=None):
    """
        Assign a unique id to each node in the graph.
        ID: (graph_id, original_id, sorted_index)
    """
    graphs = []
    for g in tqdm(sorted(os.listdir(graph_dir))):
        try:
            G = nx.read_gpickle(os.path.join(graph_dir, g))
        except Exception as e:
            print(f"failed on {g}, {e}")
            continue
        G = nx.relabel_nodes(G, {n: (g, n) for i, n in enumerate(sorted(G.nodes()))})
        if not dump is None:
            nx.write_gpickle(G, os.path.join(graph_dir, g))
        graphs.append(G)
    return graphs


def reindex_nodes_raw(graph_dir, dump=None):
    """
        Assign a unique id to each node in the graph.
    """
    graphs = []
    offset = 0
    for g in tqdm(sorted(os.listdir(graph_dir))):
        try:
            G = nx.read_gpickle(os.path.join(graph_dir, g))
        except Exception as e:
            print(f"failed on {g}, {e}")
            continue
        G = nx.relabel.convert_node_labels_to_integers(G, first_label=offset, label_attribute='id')
        offset += len(G.nodes())
        if not dump is None:
            nx.write_gpickle(G, os.path.join(dump, g))
        graphs.append(G)
    return graphs


def nx_to_dgl(graph, edge_map, embed_dim):
    """
        Networkx graph to DGL.
    """

    import torch
    import dgl

    graph, _, ring = pickle.load(open(graph, 'rb'))
    one_hot = {edge: edge_map[label] for edge, label in (nx.get_edge_attributes(graph, 'label')).items()}
    nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
    one_hot = {edge: torch.tensor(edge_map[label]) for edge, label in (nx.get_edge_attributes(graph, 'label')).items()}
    g_dgl = dgl.DGLGraph()
    g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
    n_nodes = len(g_dgl.nodes())
    g_dgl.ndata['h'] = torch.ones((n_nodes, embed_dim))

    return g_dgl


def dgl_to_nx(graph, edge_map):
    # find better way to do this..want to be able to launch without torch installed
    import torch
    import dgl
    g = dgl.to_networkx(graph, edge_attrs=['one_hot'])
    edge_map_r = {v: k for k, v in edge_map.items()}
    nx.set_edge_attributes(g, {(n1, n2): edge_map_r[d['one_hot'].item()] for n1, n2, d in g.edges(data=True)}, 'label')
    return g


def bfs_expand(G, initial_nodes, nc_block=False, depth=2):
    """
        Extend motif graph starting with motif_nodes.
        Returns list of nodes.
    """

    total_nodes = [list(initial_nodes)]
    for d in range(depth):
        depth_ring = []
        e_labels = set()
        for n in total_nodes[d]:
            for nei in G.neighbors(n):
                depth_ring.append(nei)
                e_labels.add(G[n][nei]['label'])
        if e_labels.issubset({'CWW', 'B53', ''}) and nc_block:
            break
        else:
            total_nodes.append(depth_ring)
        # total_nodes.append(depth_ring)
    return set(itertools.chain(*total_nodes))


def bfs(G, initial_node, depth=2):
    """
        Generator for bfs given graph and initial node.
        Yields nodes at next hop at each call.
    """

    total_nodes = [[initial_node]]
    visited = []
    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            visited.append(n)
            for nei in G.neighbors(n):
                if nei not in visited:
                    depth_ring.append(nei)
        total_nodes.append(depth_ring)
        yield depth_ring


def remove_self_loops(G):
    G.remove_edges_from([(n, n) for n in G.nodes()])


def remove_non_standard_edges(G):
    remove = []
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in valid_edges:
            remove.append((n1, n2))
    G.remove_edges_from(remove)


def to_orig_all(graph_dir, dump_dir):
    for g in tqdm(os.listdir(graph_dir)):
        try:
            G = nx.read_gpickle(os.path.join(graph_dir, g))
        except Exception as e:
            print(f">>> failed on {g} with exception {e}")
            continue
        H = to_orig(G)
        nx.write_gpickle(H, os.path.join(dump_dir, g))


def to_orig(G):
    H = nx.Graph()
    for n1, n2, d in G.edges(data=True):
        if d['label'] in valid_edges:
            assert d['label'] != 'B35'
            H.add_edge(n1, n2, label=d['label'])

    for attrib in ['mg', 'lig', 'lig_id', 'chemically_modified',
                   'pdb_pos', 'bgsu', 'carnaval', 'chain']:
        G_data = G.nodes(data=True)
        attrib_dict = {n: G_data[n][attrib] for n in H.nodes()}
        nx.set_node_attributes(H,
                               attrib_dict,
                               attrib)

    remove_self_loops(H)
    return H


def find_node(graph, chain, pos):
    for n, d in graph.nodes(data=True):
        if (n[0] == chain) and (d['nucleotide'].pdb_pos == str(pos)):
            return n
    return None


def has_NC(G):
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in ['CWW', 'B53']:
            # print(d['label'])
            return True
    return False


def has_NC_bfs(graph, node_id, depth=2):
    """
        Return True if node has NC in their neighbourhood.
    """

    subg = list(bfs_expand(graph, [node_id], depth=depth)) + [node_id]
    sG = graph.subgraph(subg).copy()
    return has_NC(sG)


def gap_fill(G, subG):
    """
        Get rid of all degree 1 nodes.
    """
    # while True:
    new_nodes = list(subG.nodes())
    # has_dang = False
    for n in subG.nodes():
        if subG.degree(n) == 1:
            new_nodes.append(subG.neighbors(n))
            # has_dang = True
    # if has_dang:
    # subG = G.subgraph(new_nodes).copy()
    # has_dang = False
    # else:
    # break
    return subG


def floaters(G):
    """
    Try to connect floating base pairs. (Single base pair not attached to backbone).
    Otherwise remove.
    """
    deg_ok = lambda H, u, v, d: (H.degree(u) == d) and (H.degree(v) == d)
    floaters = []
    for u, v in G.edges():
        if deg_ok(G, u, v, 1):
            floaters.append((u, v))

    G.remove_edges_from(floaters)

    return G


def dangle_trim(G, backbone_only=False):
    """
    Recursively remove dangling nodes from graph.
    """
    dangles = lambda G: [n for n in G.nodes() if G.degree(n) < 2]
    while dangles(G):
        G.remove_nodes_from(dangles(G))


def stack_trim(G):
    """
    Remove stacks from graph.
    """
    is_ww = lambda n, G: 'CWW' in [info['label'] for node, info in G[n].items()]
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        stacks = []
        for n in cur_G.nodes:
            if cur_G.degree(n) == 2 and is_ww(n, cur_G):
                # potential stack opening
                partner = None
                stacker = None
                for node, info in cur_G[n].items():
                    if info['label'] == 'B53':
                        stacker = node
                    elif info['label'] == 'CWW':
                        partner = node
                    else:
                        pass
                if cur_G.degree(partner) > 3:
                    continue
                partner_2 = None
                stacker_2 = None
                for node, info in cur_G[partner].items():
                    if info['label'] == 'B53':
                        stacker_2 = node
                    elif info['label'] == 'CWW':
                        partner_2 = node
                try:
                    if cur_G[stacker][stacker_2]['label'] == 'CWW':
                        stacks.append(n)
                        stacks.append(partner)
                except KeyError:
                    continue
        if len(stacks) == 0:
            break
        else:
            cur_G.remove_nodes_from(stacks)
            cur_G = cur_G.copy()
    return cur_G


def in_stem(G, u, v):
    non_bb = lambda G, n: len([info['label'] for node, info in G[n].items() if info['label'] != 'B53'])
    is_ww = lambda G, u, v: G[u][v]['label'] == 'CWW'
    if is_ww(G, u, v) and (non_bb(G, u) in (1, 2)) and (non_bb(G, v) in (1, 2)):
        return True
    return False


def symmetric_elabels(graph):
    """
        Make edge labels symmetric for a graph.
        Returns new graph.
    """
    H = graph.copy()
    new_e_labels = {}
    for n1, n2, d in graph.edges(data=True):
        old_label = d['label']
        if old_label not in ['B53', 'B35']:
            new_label = old_label[0] + "".join(sorted(old_label[1:]))
        else:
            new_label = 'B53'
        new_e_labels[(n1, n2)] = new_label
    nx.set_edge_attributes(H, new_e_labels, 'label')
    return H


def relabel_graphs(graph_dir, dump_path):
    """
        Take graphs in graph_dir and dump symmetrized in dump_path.
    """
    for g in tqdm(os.listdir(graph_dir)):
        G = nx.read_gpickle(os.path.join(graph_dir, g))
        G_new = symmetric_elabels(G)
        nx.write_gpickle(G_new, os.path.join(dump_path, g))
        pass
    pass


def weisfeiler_lehman_graph_hash(
        G,
        edge_attr=None,
        node_attr=None,
        iterations=3,
        digest_size=16
):
    """Return Weisfeiler Lehman (WL) graph hash.

    The function iteratively aggregates and hashes neighbourhoods of each node.
    After each node's neighbors are hashed to obtain updated node labels,
    a hashed histogram of resulting labels is returned as the final hash.

    Hashes are identical for isomorphic graphs and strong guarantees that
    non-isomorphic graphs will get different hashes. See [1] for details.

    Note: Similarity between hashes does not imply similarity between graphs.

    If no node or edge attributes are provided, the degree of each node
    is used as its initial label.
    Otherwise, node and/or edge labels are used to compute the hash.

    Parameters
    ----------
    G: graph
        The graph to be hashed.
        Can have node and/or edge attributes. Can also have no attributes.
    edge_attr: string
        The key in edge attribute dictionary to be used for hashing.
        If None, edge labels are ignored.
    node_attr: string
        The key in node attribute dictionary to be used for hashing.
        If None, and no edge_attr given, use
        degree of node as label.
    iterations: int
        Number of neighbor aggregations to perform.
        Should be larger for larger graphs.
    digest_size: int
        Size of blake2b hash digest to use for hashing node labels.

    Returns
    -------
    h : string
        Hexadecimal string corresponding to hash of the input graph.

    Examples
    --------
    Two graphs with edge attributes that are isomorphic, except for
    differences in the edge labels.

    >>> import networkx as nx
    >>> G1 = nx.Graph()
    >>> G1.add_edges_from([(1, 2, {'label': 'A'}),\
                           (2, 3, {'label': 'A'}),\
                           (3, 1, {'label': 'A'}),\
                           (1, 4, {'label': 'B'})])
    >>> G2 = nx.Graph()
    >>> G2.add_edges_from([(5,6, {'label': 'B'}),\
                           (6,7, {'label': 'A'}),\
                           (7,5, {'label': 'A'}),\
                           (7,8, {'label': 'A'})])

    Omitting the `edge_attr` option, results in identical hashes.

    >>> weisfeiler_lehman_graph_hash(G1)
    '0db442538bb6dc81d675bd94e6ebb7ca'
    >>> weisfeiler_lehman_graph_hash(G2)
    '0db442538bb6dc81d675bd94e6ebb7ca'

    With edge labels, the graphs are no longer assigned
    the same hash digest.

    >>> weisfeiler_lehman_graph_hash(G1, edge_attr='label')
    '408c18537e67d3e56eb7dc92c72cb79e'
    >>> weisfeiler_lehman_graph_hash(G2, edge_attr='label')
    'f9e9cb01c6d2f3b17f83ffeaa24e5986'

    References
    -------
    .. [1] Shervashidze, Nino, Pascal Schweitzer, Erik Jan Van Leeuwen,
       Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler Lehman
       Graph Kernels. Journal of Machine Learning Research. 2011.
       http://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf
    """

    from collections import Counter
    from hashlib import blake2b

    def neighborhood_aggregate(G, node, node_labels, edge_attr=None):
        """
            Compute new labels for given node by aggregating
            the labels of each node's neighbors.
        """
        label_list = [node_labels[node]]
        for nei in G.neighbors(node):
            prefix = "" if not edge_attr else G[node][nei][edge_attr]
            label_list.append(prefix + node_labels[nei])
        return ''.join(sorted(label_list))

    def weisfeiler_lehman_step(G, labels, edge_attr=None, node_attr=None):
        """
            Apply neighborhood aggregation to each node
            in the graph.
            Computes a dictionary with labels for each node.
        """
        new_labels = dict()
        for node in G.nodes():
            new_labels[node] = neighborhood_aggregate(G, node, labels,
                                                      edge_attr=edge_attr)
        return new_labels

    items = []
    node_labels = dict()
    # set initial node labels
    for node in G.nodes():
        if (not node_attr) and (not edge_attr):
            node_labels[node] = str(G.degree(node))
        elif node_attr:
            node_labels[node] = str(G.nodes[node][node_attr])
        else:
            node_labels[node] = ''

    for k in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels,
                                             edge_attr=edge_attr)
        counter = Counter()
        # count node labels
        for node, d in node_labels.items():
            h = blake2b(digest_size=digest_size)
            h.update(d.encode('ascii'))
            counter.update([h.hexdigest()])
        # sort the counter, extend total counts
        items.extend(sorted(counter.items(), key=lambda x: x[0]))

    # hash the final counter
    h = blake2b(digest_size=digest_size)
    h.update(str(tuple(items)).encode('ascii'))
    h = h.hexdigest()
    return h


if __name__ == "__main__":
    nc_clean_dir("../data/unchopped_v4_nr", "../data/unchopped_v4_nr_nc")
