"""Collection of functions operating on RNA graphs"""

import pickle
import os
from typing import Optional, Hashable, Dict, List, Tuple

from tqdm import tqdm
import networkx as nx
import numpy as np

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL

CANONICALS = GRAPH_KEYS["canonical"][TOOL]
VALID_EDGES = GRAPH_KEYS["edge_map"][TOOL].keys()


def multigraph_to_simple(g: nx.MultiDiGraph) -> nx.DiGraph:
    """Convert directed multi graph to simple directed graph.
    When multiple edges are found between two nodes, we keep backbone.
    """
    simple_g = nx.DiGraph()
    backbone_types = ["B53", "B35"]
    # first pass adds the backbones
    for u, v, data in g.edges(data=True):
        etype = data["LW"]
        if etype in backbone_types:
            simple_g.add_edge(u, v, **data)
        pass
    # second pass adds non-canonicals when no backbone exists
    basepairs = []
    for u, v, data in g.edges(data=True):
        etype = data["LW"]
        if etype not in backbone_types and not simple_g.has_edge(u, v):
            basepairs.append((u, v, data))

    simple_g.add_edges_from(basepairs)
    simple_g.graph = g.graph.copy()

    simple_g_nodes = set(simple_g.nodes())
    simple_g_node_attrs = {k: v for k, v in dict(g.nodes(data=True)).items() if k in simple_g_nodes}
    nx.set_node_attributes(simple_g, simple_g_node_attrs)
    return simple_g


def reorder_nodes(g: nx.DiGraph) -> nx.DiGraph:
    """
    Reorder nodes in graph according to default ``sorted()`` order.

    :param g: Pass a graph for node reordering.
    :type g: networkx.DiGraph

    :return h: (nx DiGraph)
    """

    reordered_graph = nx.DiGraph()
    reordered_graph.add_nodes_from(sorted(g.nodes.data()))
    reordered_graph.add_edges_from(g.edges.data())
    for key, value in g.graph.items():
        reordered_graph.graph[key] = value
    return reordered_graph


def induced_edge_filter(graph: nx.DiGraph, roots: List[Hashable], depth: Optional[int] = 1) -> nx.DiGraph:
    """
    Remove edges in graph introduced by the induced sugraph routine.
    Only keep edges which fall within a single node's neighbourhood.

    :param graph: networkx subgraph
    :param roots: nodes to use for filtering
    :param depth: size of neighbourhood to take around each node.

    :returns clean_g: cleaned graph
    """
    # a depth of zero does not make sense for this operation as it would remove
    # all edges
    if depth < 1:
        depth = 1
    neighbourhoods = []
    flat_neighbors = set()
    for root in roots:
        root_neighbors = bfs(graph, [root], depth=depth)
        neighbourhoods.append(root_neighbors)
        flat_neighbors = flat_neighbors.union(root_neighbors)

    flat_neighbors = list(flat_neighbors)
    subgraph = graph.subgraph(flat_neighbors)
    subgraph = subgraph.copy()
    # graph_new = graph_new.subgraph(flat_neighbors)
    kill = []
    for u, v in subgraph.edges():
        for nei in neighbourhoods:
            if u in nei and v in nei:
                break
        else:
            kill.append((u, v))

    subgraph.remove_edges_from(kill)
    return subgraph


def get_nc_nodes(graph: nx.DiGraph, depth: int = 4, return_index: bool = False) -> set:
    """
    Returns indices of nodes in graph which have a non-canonical or
    looping base in their neighbourhood.

    :param graph: a networkx graph
    :param depth: The depth up to which we consider nodes neighbors of a NC
    :param return_index: If True, return the index in the list instead.
    :return: set of nodes (or their index) in loops or that have a NC.
    """

    keep = []
    for i, node in enumerate(sorted(graph.nodes())):
        to_keep = i if return_index else node
        if graph.degree(node) == 2:
            keep.append(to_keep)
        elif has_NC_bfs(graph, node, depth=depth):
            keep.append(to_keep)
        else:
            pass
    return set(keep)


def nc_clean_dir(graph_dir, dump_dir):
    """
    Copy graphs from graph_dir to dump_dir but copied graphs are
        trimmed according to `get_nc_nodes_index`.

    :param graph_dir: A directory that should contain networkx pickles.
    :param dump_dir: The directory where to dump the trimmed graphs
    """

    for g in tqdm(os.listdir(graph_dir)):
        graph = nx.read_gpickle(os.path.join(graph_dir, g))
        keep_nodes = get_nc_nodes(graph)
        print(f">>> kept {len(keep_nodes)} nodes of {len(graph.nodes())}.")
        kill_nodes = set(graph.nodes()) - keep_nodes
        graph.remove_nodes_from(kill_nodes)
        dangle_trim(graph)
        if len(graph.nodes()) > 4:
            nx.write_gpickle(graph, os.path.join(dump_dir, g))


def incident_nodes(graph, nodes):
    """
    Returns set of nodes in $graph$ incident to input nodes.

    :param graph: A networkx graph
    :param nodes: set of nodes in graph

    :return: set of nodes around the input the set of nodes according to the connectivity of the graph
    """
    core = set(nodes)
    hits = set()
    for u, v in graph.edges():
        if u in core and v not in core:
            hits.add(v)
        if u not in core and v in core:
            hits.add(u)
    return hits


def nx_to_dgl(graph, edge_map, label="label"):
    """
    Networkx graph to DGL.
    """
    import dgl

    graph, _, ring = pickle.load(open(graph, "rb"))
    edge_type = {edge: edge_map[lab] for edge, lab in (nx.get_edge_attributes(graph, label)).items()}
    nx.set_edge_attributes(graph, name="edge_type", values=edge_type)
    g_dgl = dgl.DGLGraph()
    g_dgl.from_networkx(nx_graph=graph, edge_attrs=["edge_type"])
    return g_dgl


def dgl_to_nx(graph, edge_map, label="label"):
    import dgl

    g = dgl.to_networkx(graph, edge_attrs=["edge_type"])
    edge_map_r = {v: k for k, v in edge_map.items()}
    nx.set_edge_attributes(
        g,
        {(n1, n2): edge_map_r[d["edge_type"].item()] for n1, n2, d in g.edges(data=True)},
        label,
    )
    return g


def bfs_generator(graph, initial_node):
    """
    Generator version of bfs given graph and initial node.
    Yields nodes at next hop at each call.

    :param graph: Nx graph
    :param initial_node: single or iterable node
    :param depth:

    :return: The successive rings
    """
    if isinstance(initial_node, list) or isinstance(initial_node, set):
        previous_ring = [set(initial_node)]
    else:
        previous_ring = [set(initial_node)]
    visited = set()
    while len(visited) < len(graph):
        depth_ring = set()
        for n in previous_ring:
            visited.add(n)
            for nei in graph.neighbors(n):
                if nei not in visited:
                    depth_ring.add(nei)
        previous_ring = depth_ring
        yield list(depth_ring)


def bfs(graph, initial_nodes, nc_block=False, depth=2, label="label"):
    """
    BFS from seed nodes given graph and initial node.

    :param graph: Nx graph
    :param initial_nodes: single or iterable node
    :param depth: The number of hops to conduct from our roots

    :return: list of nodes
    """
    if isinstance(initial_nodes, list) or isinstance(initial_nodes, set):
        total_nodes = [set(initial_nodes)]
    else:
        total_nodes = [set(initial_nodes)]
    for d in range(depth):
        depth_ring = set()
        e_labels = set()
        for n in total_nodes[d]:
            for nei in graph.neighbors(n):
                depth_ring.add(nei)
                e_labels.add(graph[n][nei][label])
        if nc_block and e_labels.issubset({"CWW", "B53", ""}):
            break
        else:
            total_nodes.append(depth_ring)
    total_nodes = set().union(*total_nodes)
    return total_nodes


def extract_graphlet(graph, n, size=1, label="LW"):
    """
    Small util to extract a graphlet around a node

    :param graph: Nx graph
    :param n: a node in the graph
    :param size: The depth to consider

    :return: The graphlet as a copy
    """
    graphlet = graph.subgraph(bfs(graph, [n], depth=size, label=label)).copy()
    return graphlet


def remove_self_loops(graph):
    """
    Remove all self loops connexions by modifying in place

    :param graph: The graph to trim

    :return: None
    """
    graph.remove_edges_from([(n, n) for n in graph.nodes()])


def remove_non_standard_edges(graph, label="LW"):
    """
    Remove all edges whose label is not in the VALID EDGE variable

    :param graph: Nx Graph
    :param label: The name of the labels to check

    :return: the pruned graph, modifications are made in place
    """
    remove = []
    for n1, n2, d in graph.edges(data=True):
        if d[label] not in VALID_EDGES:
            remove.append((n1, n2))
    graph.remove_edges_from(remove)


def to_orig(graph, label="LW"):
    """
    Deprecated, used to include only the NC

    :param graph:
    :param label:

    :return:
    """
    H = nx.Graph()
    for n1, n2, d in graph.edges(data=True):
        if d[label] in VALID_EDGES:
            assert d[label] != "B35"
            H.add_edge(n1, n2, label=d[label])

    for attrib in [
        "mg",
        "lig",
        "lig_id",
        "chemically_modified",
        "pdb_pos",
        "bgsu",
        "carnaval",
        "chain",
    ]:
        graph_data = graph.nodes(data=True)
        attrib_dict = {n: graph_data[n][attrib] for n in H.nodes()}
        nx.set_node_attributes(H, attrib_dict, attrib)

    remove_self_loops(H)
    return H


def to_orig_all(graph_dir, dump_dir):
    """
    Deprecated

    :param graph_dir:
    :param dump_dir:

    :return:
    """
    for g in tqdm(os.listdir(graph_dir)):
        try:
            graph = nx.read_gpickle(os.path.join(graph_dir, g))
        except Exception as e:
            print(f">>> failed on {g} with exception {e}")
            continue
        H = to_orig(graph)
        nx.write_gpickle(H, os.path.join(dump_dir, g))


def find_node(graph, chain, pos):
    """
    Get a node from its PDB identification

    :param graph: Nx graph
    :param chain: The PDB chain
    :param pos: The PDB 'POS' field

    :return: The node if it was found, else None
    """
    for n, d in graph.nodes(data=True):
        if (n[0] == chain) and (d["nucleotide"].pdb_pos == str(pos)):
            return n
    return None


def has_NC(graph, label="LW"):
    """
    Does the input graph contain non canonical edges ?

    :param graph: Nx graph
    :param label: The label to use

    :return: Boolean
    """
    for n1, n2, d in graph.edges(data=True):
        if d[label] not in CANONICALS:
            return True
    return False


def has_NC_bfs(graph, node_id, depth=2):
    """
        Return True if node has NC in their neighbourhood.

    :param graph: Nx graph
    :param node_id: The nodes from which to start our search
    :param depth: The number of hops to conduct from our roots

    :return: Boolean
    """

    subg = list(bfs(graph, node_id, depth=depth))
    sG = graph.subgraph(subg).copy()
    return has_NC(sG)


def floaters(graph):
    """
    Try to connect floating base pairs. (Single base pair not attached
    to backbone).
    Otherwise remove.

    :param graph: Nx graph

    :return: trimmed graph
    """
    deg_ok = lambda H, u, v, d: (H.degree(u) == d) and (H.degree(v) == d)
    floaters = []
    for u, v in graph.edges():
        if deg_ok(graph, u, v, 1):
            floaters.append((u, v))

    graph.remove_edges_from(floaters)

    return graph


def dangle_trim(graph):
    """
    Recursively remove dangling nodes from graph, with in place modification

    :param graph: Nx graph

    :return: trimmed graph
    """
    dangles = lambda graph: [n for n in graph.nodes() if graph.degree(n) < 2]
    while dangles(graph):
        graph.remove_nodes_from(dangles(graph))
    return graph


def stack_trim(graph):
    """
    Remove stacks from graph.

    :param graph: Nx graph

    :return: trimmed graph
    """
    is_ww = lambda e, graph: "CWW" in [info["LW"] for node, info in graph[e].items()]
    degree = lambda i, graph, nodelist: np.sum(nx.to_numpy_matrix(graph, nodelist=nodelist)[i])
    cur_graph = graph.copy()
    while True:
        stacks = []
        for n in cur_graph.nodes:
            if cur_graph.degree(n) == 2 and is_ww(n, cur_graph):
                # potential stack opening
                partner = None
                stacker = None
                for node, info in cur_graph[n].items():
                    if info["label"] == "B53":
                        stacker = node
                    elif info["label"] == "CWW":
                        partner = node
                    else:
                        pass
                if cur_graph.degree(partner) > 3:
                    continue
                partner_2 = None
                stacker_2 = None
                for node, info in cur_graph[partner].items():
                    if info["label"] == "B53":
                        stacker_2 = node
                    elif info["label"] == "CWW":
                        partner_2 = node
                try:
                    if cur_graph[stacker][stacker_2]["label"] == "CWW":
                        stacks.append(n)
                        stacks.append(partner)
                except KeyError:
                    continue
        if len(stacks) == 0:
            break
        else:
            cur_graph.remove_nodes_from(stacks)
            cur_graph = cur_graph.copy()
    return cur_graph


def in_stem(graph, u, v):
    """
    Find if two nodes are part of a stem and engage in NC interactions

    :param graph: Nx graph
    :param u: one graph node
    :param v: one graph node

    :return: Boolean
    """
    non_bb = lambda graph, e: len([info["LW"] for node, info in graph[e].items() if info["LW"] not in CANONICALS])
    is_ww = lambda graph, u, v: graph[u][v]["LW"] not in {"CWW", "cWW"}
    if is_ww(graph, u, v) and (non_bb(graph, u) in (1, 2)) and (non_bb(graph, v) in (1, 2)):
        return True
    return False


def gap_fill(original_graph, graph_to_expand):
    """
    If we subgraphed, get rid of all degree 1 nodes by completing them with one more hop

    :param original_graph: nx graph
    :param graph_to_expand: nx graph that needs to be expanded to fix dangles

    :return: the expanded graph
    """
    # while True:
    new_nodes = list(graph_to_expand.nodes())
    for n in graph_to_expand.nodes():
        if graph_to_expand.degree(n) == 1:
            new_nodes.append(graph_to_expand.neighbors(n))
    res_graph = original_graph.subgraph(new_nodes).copy()
    return res_graph


def symmetric_elabels(graph):
    """
    Make edge labels symmetric for a graph.

    :param graph: Nx graph

    :return: Same graph but edges are now symmetric and calling undirected is straightforward.
    """
    H = graph.copy()
    new_e_labels = {}
    for n1, n2, d in graph.edges(data=True):
        old_label = d["label"]
        if old_label not in ["B53", "B35"]:
            new_label = old_label[0] + "".join(sorted(old_label[1:]))
        else:
            new_label = "B53"
        new_e_labels[(n1, n2)] = new_label
    nx.set_edge_attributes(H, new_e_labels, "label")
    return H


def relabel_graphs(graph_dir, dump_path):
    """
    Take graphs in graph_dir and dump symmetrized in dump_path.
    """
    for g in tqdm(os.listdir(graph_dir)):
        graph = nx.read_gpickle(os.path.join(graph_dir, g))
        graph_new = symmetric_elabels(graph)
        nx.write_gpickle(graph_new, os.path.join(dump_path, g))
        pass
    pass


def weisfeiler_lehman_graph_hash(graph, edge_attr=None, node_attr=None, iterations=3, digest_size=16):
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
    graph: graph
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

    def neighborhood_aggregate(graph, node, node_labels, edge_attr=None):
        """
        Compute new labels for given node by aggregating
        the labels of each node's neighbors.
        """
        label_list = [node_labels[node]]
        for nei in graph.neighbors(node):
            prefix = "" if not edge_attr else graph[node][nei][edge_attr]
            label_list.append(prefix + node_labels[nei])
        return "".join(sorted(label_list))

    def weisfeiler_lehman_step(graph, labels, edge_attr=None, node_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = dict()
        for node in graph.nodes():
            new_labels[node] = neighborhood_aggregate(graph, node, labels, edge_attr=edge_attr)
        return new_labels

    items = []
    node_labels = dict()
    # set initial node labels
    for node in graph.nodes():
        if (not node_attr) and (not edge_attr):
            node_labels[node] = str(graph.degree(node))
        elif node_attr:
            node_labels[node] = str(graph.nodes[node][node_attr])
        else:
            node_labels[node] = ""

    for k in range(iterations):
        node_labels = weisfeiler_lehman_step(graph, node_labels, edge_attr=edge_attr)
        counter = Counter()
        # count node labels
        for node, d in node_labels.items():
            h = blake2b(digest_size=digest_size)
            h.update(d.encode("ascii"))
            counter.update([h.hexdigest()])
        # sort the counter, extend total counts
        items.extend(sorted(counter.items(), key=lambda x: x[0]))

    # hash the final counter
    h = blake2b(digest_size=digest_size)
    h.update(str(tuple(items)).encode("ascii"))
    h = h.hexdigest()
    return h


def fix_buggy_edges(graph, label="LW", strategy="remove", edge_map=GRAPH_KEYS["edge_map"][TOOL]):
    """
    Sometimes some edges have weird names such as t.W representing a fuzziness.
    We just remove those as they don't deliver a good information

    :param graph:
    :param strategy: How to deal with it : for now just remove them.
    In the future maybe add an edge type in the edge map ?
    :return:
    """
    if strategy == "remove":
        # Filter weird edges for now
        to_remove = list()
        for start_node, end_node, nodedata in graph.edges(data=True):
            if nodedata[label] not in edge_map:
                to_remove.append((start_node, end_node))
        for start_node, end_node in to_remove:
            graph.remove_edge(start_node, end_node)
    else:
        raise ValueError(f"The edge fixing strategy : {strategy} was not implemented yet")
    return graph


def get_sequences(graph: nx.Graph,
                  gap_tolerance=2,
                  longest_only=True,
                  min_size_return=5,
                  verbose=True) -> Tuple[Dict[str, Tuple[str, List[str]]]]:
    """Extract ordered sequences from each chain of the RNA.
    Returns a dictionary mapping <pdbid.chain>: (sequence, list of node IDs)

    .. warning::
        Currently does not handle missing residues. If a residue is missing it is simply skipped.

    :param graph: an nx.Graph of an RNA.

    """

    sequences = {}
    chains = set([n.split(".")[1] for n in graph.nodes()])
    seqs = {c: [] for c in chains}
    for nt, d in graph.nodes(data=True):
        pdbid, ch, pos = nt.split(".")
        nuc = d["nt_code"].upper()
        if nuc not in ["A", "U", "C", "G"]:
            nuc = "N"
        seqs[ch].append((nuc, int(pos)))

    for ch, seq in seqs.items():
        sorted_seq = sorted(seq, key=lambda x: x[1])
        sorted_ids = [f"{pdbid}.{ch}.{pos}" for _, pos in sorted_seq]

        # check if sequence is discontinuous and keep track of all its consecutive segments
        previous = 0
        consecutives = []
        for i in range(len(sorted_ids) - 1):
            fivep = int(sorted_ids[i].split(".")[2])
            threep = int(sorted_ids[i + 1].split(".")[2])
            if threep != fivep + 1:
                if verbose:
                    print(f"WARNING: chain discontinuous.")
                gap = threep - fivep - 1
                if gap >= gap_tolerance:
                    consecutives.append((previous, i + 1))
                    previous = i + 1
        consecutives.append((previous, len(sorted_ids)))

        # Simply return the longest
        if longest_only:
            longest = sorted(consecutives, key=lambda x: x[1] - x[0])[-1]
            consecutives = [longest]
        # If we return more than one, only keep ones larger than a threshold, using 5 is nice for CD-Hit usage
        else:
            consecutives = [x for x in consecutives if x[1] - x[0] > min_size_return]

        # Finally, return all such chunks, named with their start/end residues
        for i, (start, end) in enumerate(consecutives):
            sorted_seq_chunk = "".join([s for s, _ in sorted_seq[start:end]])
            sorted_ids_chunk = sorted_ids[start:end]
            if len(consecutives) == 1:
                chunk_name = f"{pdbid}.{ch}"
            else:
                start_id = sorted_ids_chunk[0].split('.')[-1]
                end_id = sorted_ids_chunk[-1].split('.')[-1]
                chunk_name = f"{pdbid}.{ch}.{start_id}.{end_id}"
            sequences[chunk_name] = sorted_seq_chunk, sorted_ids_chunk
    return sequences
