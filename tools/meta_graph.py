"""
    Functions for iterative motif building.  """

import sys
import os
import time
from collections import Counter, defaultdict
import pickle
import itertools
import doctest

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from tqdm import tqdm
import multiset as ms

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from tools.graph_utils import graph_from_node, whole_graph_from_node, has_NC_bfs
from tools.learning_utils import inference_on_graph_run
from tools.learning_utils import inference_on_list
from tools.graph_utils import bfs_expand, graph_from_node, fetch_graph
from tools.clustering import *
from tools.rna_ged_nx import ged


import seaborn as sns
#from tools.drawing import rna_draw

def def_set():
    return defaultdict(set)

class MGraph:
    """
    Data we have :
    - graphs with nodes like (graph_id, node_id)
    - embeddings are an embedding array Z (number of nodes, dimensions) and a node_map {node_id : index in array}

    Meta graph : a nx graph with :
        -    MNodes are clusters that pass filters. They contain a set of node_ids they contain
        -    MEdges are a set of all edges in the original graphs in the form (smaller_id, bigger_id)

    The node data is called 'node_ids'
    The edge data is called 'edge_set'

    self.run : to produce the embeddings
    self.graph_dir = '../data/annotated/whole_v3'

    # Nodes parameters
    n_components : number of clusters considered
    min_count : min number of nodes to keep a cluster
    max_var : max variance authorized inside a cluster

    # Edges parameters
    min_edge : min connectivity to keep two clusters connected

    # Clustering
    node_map: {(graph, nodeid) : id }
    reversed_node_map : reversed.
    id_to_score : { id : max_proba } the proba with which a node belongs to a cluster.
                  We could also use embeddings distances
    gm : a gaussian mixture object that is already trained to perform inference on.


    """

    def split_by_graph(self, nodesets):
        graphsets = defaultdict(list)
        for s in nodesets:
            node_id = self.reversed_node_map[list(s)[0]]
            graph_id = node_id[0]
            graphsets[graph_id].append(s)
        return graphsets


    def prune(self):
        """
            Remove Mnodes or individual nodes
            from meta graph based on distance/probability criteria.
        """
        kill_nodes = []
        var_fails = 0
        count_fails = 0
        keep = 0
        for node, d in self.graph.nodes(data=True):
            sp = self.spread[node]
            if sp > self.max_var or np.isnan(sp):
                kill_nodes.append(node)
                var_fails += 1
            elif len(d['node_ids']) < self.min_count:
                kill_nodes.append(node)
                count_fails += 1
            else:
                keep += 1
                continue

        self.graph.remove_nodes_from(kill_nodes)
        print(f"Killed {len(kill_nodes)} mnodes. Kept {keep}")
        print(f"Variance fails {var_fails}, Count fails {count_fails}")
        pass

    def retrieve(self, motif):
        """
        Start with a motif representative : a list of nodes that make motif.
        Build the query graph :
         -Create embeddings for the motif nodes, they need the whole graph. then do clustering and put query nodes
         in the appropriate cluster. Then add the edges that make up the connectivity of the query motif
        - Then add all nodes in a cluster that is part of the query graph in a big motif_instance set
        - Then Follow the query graph edges and connect these instances

        We maintain both a dict motif_instance { frozenset_of_ids : score}
        and a dict motifs_instances_grouped { pdb_id : set of frozensets } for a more efficient looping :
        When exploring a new edge in the query meta graph, we loop through edges that make this edge and every time
        we can only look at the frozensets in motifs_instances_grouped[current_pdb]
        :param motif:
        :return: {frozenset of node ids : score}
        """
        original_graph = whole_graph_from_node(motif[0])
        # sort the query edges based on meta edge identity to get speedup
        query_nodes, query_edges = self.build_query_graph(original_graph, motif)
        query_edges = sorted(list(query_edges), key=lambda x: (x[2], x[3]))

        # print(query_nodes)
        # print(query_edges)
        # print("query nodes ", query_nodes)

        # get unchopped version of node IDs
        # motif = [(pdbid.split("_")[0] + ".nx", n) for (pdbid, n) in motif]

        def node_to_pdbid(node_index):
            """
                Return PDB which contains motif instance frozenset.
            """
            pdbid, _ = self.reversed_node_map[list(node_index)[0]]
            return pdbid[:4]

        # get all nodes in the data that might be involved ie 1 motifs that could be a part of the motif
        motifs_instances_int = set()
        for node, clust_id in query_nodes:
            mnode = self.graph.nodes[clust_id]
            node_candidates = mnode['node_ids']
            motifs_instances_int = motifs_instances_int.union(node_candidates)

        # map pdbid -> set(frozensets)
        motifs_instances_grouped = defaultdict(set)
        # Turn the ids of seeds into frozensets that represent current motif nodes that get expanded
        motifs_instances = dict()
        for i, int_id in enumerate(motifs_instances_int):
            motif_nodes = frozenset([int_id])
            motifs_instances[motif_nodes] = float(self.id_to_score[int_id])
            motifs_instances_grouped[node_to_pdbid(motif_nodes)].add(motif_nodes)

        # Iterate through edges and merge
        for edge in query_edges:
            # Try to access the corresponding meta-edge and get the list of all candidate edges
            _, _, start_clust, end_clust, _ = edge
            try:
                medge_set = self.graph.edges[start_clust, end_clust]['edge_set']
            except KeyError:
                print('skipped an edge')
                continue
            # A node that was expanded should be removed after this round
            # as its merged version is strictly more promising
            visited_ones = set()
            new_ones = dict()
            # print()
            # print('=== NEW EDGE === ')
            # print('M of cardinal', len(motifs_instances))
            # print()

            for start_node, end_node, distance in medge_set:
                # Adding the nodes in the two directions can result in introducing two sets with one
                # being the subset of the other. We manually check that using a temp dict
                temp_new_ones = dict()
                current_pdb = node_to_pdbid(frozenset([start_node]))
                # start = time.perf_counter()
                for current_motif in motifs_instances_grouped[current_pdb]:
                    # for current_motif in motifs_instances:
                    # Only take expanding motifs
                    if start_node not in current_motif and end_node not in current_motif:
                        continue
                    if start_node in current_motif and end_node in current_motif:
                        continue

                    # If one of the end of the edge is in the current motif, expand it
                    # Then we remove it from the list as it would only yield inferior scores
                    visited_ones.add(current_motif)
                    score = motifs_instances[current_motif]
                    extended_motif = set(current_motif)
                    if start_node in current_motif:
                        extended_motif.add(end_node)
                        new_score = float(self.id_to_score[end_node])
                    else:
                        extended_motif.add(start_node)
                        new_score = float(self.id_to_score[start_node])

                    extended_motif = frozenset(extended_motif)
                    temp_new_ones[extended_motif] = new_score + score
                    # new_ones[extended_motif] = new_score + score
                # print(f">>> time1 {time.perf_counter() - start}")
                # start = time.perf_counter()

                # # Now we remove the doublons results (when the edge added both times resulted in an inferior result)
                # This has negligeable runtime compared to iteration
                to_remove = set()
                for sa, sb in itertools.combinations(temp_new_ones.keys(), 2):
                    if sa.issubset(sb):
                        to_remove.add(sa)
                    if sb.issubset(sa):
                        to_remove.add(sb)
                for subset in to_remove:
                    del temp_new_ones[subset]
                new_ones.update(temp_new_ones)
                # print(f">>> time2 {time.perf_counter() - start}")

            # Now we remove all doublons
            # to_remove = set()
            # for sa, sb in itertools.combinations(new_ones.keys(), 2):
            #     if sa.issubset(sb):
            #         to_remove.add(sa)
            #     if sb.issubset(sa):
            #         to_remove.add(sb)
            # for subset in to_remove:
            #     # if 1232 in subset:
            #     #     print('subset', subset)
            #     #     print()
            #     del new_ones[subset]

            # map(motifs_instances.pop, visited_ones)
            for visited in visited_ones:
                motifs_instances.pop(visited)
                motifs_instances_grouped[node_to_pdbid(visited)].remove(visited)
            motifs_instances.update(new_ones)
            for new_one in new_ones:
                motifs_instances_grouped[node_to_pdbid(new_one)].add(new_one)
        return motifs_instances

    def retrieve_2(self, motif):
        """
        Start with a motif representative : a list of nodes that make motif.
        Build the query graph :
         -Create embeddings for the motif nodes, they need the whole graph. then do clustering and put query nodes
         in the appropriate cluster. Then add the edges that make up the connectivity of the query motif
        - Then add all nodes in a cluster that is part of the query graph in a big motif_instance set
        - Then Follow the query graph edges and connect these instances

        We maintain both a dict motif_instance { frozenset_of_ids : score}
        and a dict motifs_instances_grouped { pdb_id : set of frozensets } for a more efficient looping :
        When exploring a new edge in the query meta graph, we loop through edges that make this edge and every time
        we can only look at the frozensets in motifs_instances_grouped[current_pdb]
        :param motif:
        :return: {frozenset of node ids : score}
        """
        original_graph = whole_graph_from_node(motif[0])
        query_nodes, query_edges = self.build_query_graph(original_graph, motif)

        # Sort the query edges based on meta edge identity to get speedup
        # Try other sorting : the fastest is that one where we start with
        # populated edges that thus don't have to go trough a large M
        # query_edges = sorted(list(query_edges), key=lambda x: (x[2], x[3]))
        clusts_populations = {clust_id: len(self.graph.nodes[clust_id]['node_ids']) for node, clust_id in query_nodes}
        query_edges = sorted(list(query_edges),
                             key=lambda x: (-sum((clusts_populations[x[2]], clusts_populations[x[3]])), x[2]))

        def node_to_pdbid(node_index):
            """
                Return PDB which contains motif instance frozenset.
            """
            pdbid, _ = self.reversed_node_map[list(node_index)[0]]
            return pdbid[:4]

        def add_mnode(clust_id, mg, motifs_instances, motifs_instances_grouped):
            # get all nodes in the data that might be involved ie 1 motifs that could be a part of the motif
            mnode = mg.graph.nodes[clust_id]
            node_candidates = mnode['node_ids']

            # map pdbid -> set(frozensets)
            # Turn the ids of seeds into frozensets that represent current motif nodes that get expanded
            for int_id in node_candidates:
                motif_nodes = frozenset([int_id])
                motifs_instances[motif_nodes] = float(mg.id_to_score[int_id])
                motifs_instances_grouped[node_to_pdbid(motif_nodes)].add(motif_nodes)

        motifs_instances = dict()
        motifs_instances_grouped = defaultdict(set)
        visited_clusts = set()

        for edge in query_edges:
            # Try to access the corresponding meta-edge and get the list of all candidate edges
            _, _, start_clust, end_clust, _ = edge
            try:
                medge_set = self.graph.edges[start_clust, end_clust]['edge_set']
            except KeyError:
                continue

            if start_clust not in visited_clusts:
                add_mnode(start_clust,
                          mg=self,
                          motifs_instances=motifs_instances,
                          motifs_instances_grouped=motifs_instances_grouped)
                visited_clusts.add(start_clust)

            if end_clust not in visited_clusts:
                add_mnode(end_clust,
                          mg=self,
                          motifs_instances=motifs_instances,
                          motifs_instances_grouped=motifs_instances_grouped)
                visited_clusts.add(start_clust)

            # A node that was expanded should be removed after this round
            # as its merged version is strictly more promising
            visited_ones = set()
            new_ones = dict()
            # print()
            # print('=== NEW EDGE === ')
            # print('M of cardinal', len(motifs_instances))
            # print()

            for start_node, end_node, distance in medge_set:
                # Adding the nodes in the two directions can result in introducing two sets with one
                # being the subset of the other. We manually check that using a temp dict
                temp_new_ones = dict()
                current_pdb = node_to_pdbid(frozenset([start_node]))
                # start = time.perf_counter()
                for current_motif in motifs_instances_grouped[current_pdb]:
                    # for current_motif in motifs_instances:
                    # Only take expanding motifs
                    if start_node not in current_motif and end_node not in current_motif:
                        continue
                    if start_node in current_motif and end_node in current_motif:
                        continue

                    # If one of the end of the edge is in the current motif, expand it
                    # Then we remove it from the list as it would only yield inferior scores
                    visited_ones.add(current_motif)
                    score = motifs_instances[current_motif]
                    extended_motif = set(current_motif)
                    if start_node in current_motif:
                        extended_motif.add(end_node)
                        new_score = float(self.id_to_score[end_node])
                    else:
                        extended_motif.add(start_node)
                        new_score = float(self.id_to_score[start_node])

                    extended_motif = frozenset(extended_motif)
                    temp_new_ones[extended_motif] = new_score + score
                    # new_ones[extended_motif] = new_score + score

                # print(f">>> time1 {time.perf_counter() - start}")
                # start = time.perf_counter()

                # # Now we remove the doublons results (when the edge added both times resulted in an inferior result)
                # This has negligeable runtime compared to iteration
                to_remove = set()
                for sa, sb in itertools.combinations(temp_new_ones.keys(), 2):
                    if sa.issubset(sb):
                        to_remove.add(sa)
                    if sb.issubset(sa):
                        to_remove.add(sb)
                for subset in to_remove:
                    del temp_new_ones[subset]
                new_ones.update(temp_new_ones)
                # print(f">>> time2 {time.perf_counter() - start}")

            # map(motifs_instances.pop, visited_ones)
            for visited in visited_ones:
                motifs_instances.pop(visited)
                motifs_instances_grouped[node_to_pdbid(visited)].remove(visited)
            motifs_instances.update(new_ones)
            for new_one in new_ones:
                motifs_instances_grouped[node_to_pdbid(new_one)].add(new_one)
        return motifs_instances


    def statistics(self):
        """
        Computes statistics over the mnodes and medges occupancy
        :return:
        """
        node_counts = list()
        edge_counts = list()

        for node, node_ids in self.graph.nodes(data='node_ids'):
            # print(node_set)
            node_counts.append(len(node_ids))

        for start, end, edge_set in self.graph.edges(data='edge_set'):
            edge_counts.append(len(edge_set))

        return node_counts, edge_counts

class MGraphNC(MGraph):
    """
    This one is only to use the non canonical
    """

    def __init__(self,
                 run,
                 graph_dir='../data/annotated/whole_v4',
                 n_components=8,
                 min_count=50,
                 max_var=0.1,
                 min_edge=50,
                 clust_algo='k_means',
                 aggregate=-1,
                 optimize=True,
                 max_graphs=None,
                 nc_only=True):
        # General
        self.run = run
        self.graph_dir = graph_dir
        # Nodes parameters
        self.n_components = n_components
        self.min_count = min_count
        self.max_var = max_var

        # Edges parameters
        self.min_edge = min_edge

        # BUILD MNODES
        model_output = inference_on_list(self.run,
                                         self.graph_dir,
                                         os.listdir(self.graph_dir),
                                         max_graphs=max_graphs,
                                         nc_only=nc_only
                                         )

        self.node_map = model_output['node_to_zind']
        self.reversed_node_map = {value: key for key, value in self.node_map.items()}

        Z = model_output['Z']
        # self.Z = Z

        # Extract the non canonical edges ids and compute distances between them
        nc_nodes = set()
        nc_edges = set()
        for graph_name in os.listdir(self.graph_dir)[:max_graphs]:
            graph_path = os.path.join(self.graph_dir, graph_name)
            g = pickle.load(open(graph_path, 'rb'))['graph'].to_undirected()
            local_nodes = set()
            for source, target, label in g.edges(data='label'):
                if label not in ['CWW', 'B53']:
                    local_nodes.add((source, self.node_map[source]))
                    local_nodes.add((target, self.node_map[target]))
            for source, sid in local_nodes:
                nc_nodes.add(sid)
            for (source, sid), (target, tid) in itertools.combinations(local_nodes, 2):
                try:
                    distance = len(nx.shortest_path(g, source, target))
                    # TODO : Find better cutoff
                    if distance < 7:
                        nc_edges.add((sid, tid, distance))
                except nx.NetworkXNoPath:
                    pass

        list_ids = sorted(list(nc_nodes))
        extracted_embeddings = Z[list_ids]

        clust_info = cluster(extracted_embeddings,
                             algo=clust_algo,
                             optimize=optimize,
                             n_clusters=n_components)

        distance = True

        self.cluster_model = clust_info['model']
        self.n_components = clust_info['n_components']
        self.components = clust_info['components']
        self.labels = clust_info['labels']
        self.centers = clust_info['centers']
        if distance:
            dists = cdist(Z, self.centers)
            scores = np.take_along_axis(dists, self.labels[:, None], axis=1)
        else:
            probas = clust_info['scores']
            scores = np.take_along_axis(probas, self.labels[:, None], axis=1)
        self.id_to_score = {ind: scores[ind]
                            for ind, _ in self.reversed_node_map.items()}
        self.spread = clust_info['spread']

        self.graph = nx.Graph()

        # don't keep clusters that are too sparse or not populated enough
        # keep_clusts = cluster_filter(clusts, cov, self.min_count, self.max_var)
        # keep_clusts = set(keep_clusts)
        # print(f">>> keeping {len(self.clusts)} clusters")
        for id_clust in self.components:
            self.graph.add_node(id_clust, node_ids=set())

        # Here there needs to be a modification to avoid putting wrong nodes
        for index, clust in enumerate(self.labels):
            if clust in list(set(self.labels)):
                nc_id = list_ids[index]
                self.graph.nodes[clust]['node_ids'].add(nc_id)

        id_to_clust = dict(zip(list_ids, self.labels))

        # BUILD MEDGES
        for sid, tid, distance in nc_edges:
            # Filter out the nodes that link a cluster that got removed
            start_clust, end_clust = id_to_clust[sid], id_to_clust[tid]
            if start_clust in self.clusts and end_clust in self.clusts:
                if not self.graph.has_edge(start_clust, end_clust):
                    self.graph.add_edge(start_clust, end_clust, edge_set=set())
                self.graph.edges[(start_clust, end_clust)]['edge_set'].add((sid, tid, distance))

        # Filtering and hashing
        to_remove = list()
        for start, end, edge_set in self.graph.edges(data='edge_set'):
            # remove from adjacency
            if len(edge_set) < self.min_edge:
                to_remove.append((start, end))
        for start, end in to_remove:
            self.graph.remove_edge(start, end)

    def build_query_graph(self, original_graph, motif):
        """
        From an rna graph and the motif nodes flagged as such,
        return two sets with nodes that have cluster ids
        and edges between those clusters but cannot be a graph because of the several edges that can happen in a motif
        :param original_graph: a nx graph of the full chunk
        :param motif: a list of nodes flagged as motifs
        :return:
        """

        # BUILD MNODES
        Z, motif_node_map = inference_on_graph_run(self.run,
                                                   graph=original_graph,
                                                   verbose=False)

        local_reversed_node_map = {value: key for key, value in motif_node_map.items()}
        # self.Z = Z

        nx_motif = original_graph.subgraph(motif)
        predictions = self.cluster_model.predict(Z)
        motif_clust_map = {node: predictions[motif_node_map[node]] for node in nx_motif}

        query_nodes = set()
        # query_graph = nx.Graph()
        # We have to add the nodes that pass the selection criterion and are in kept clusters
        for source, target, label in nx_motif.edges(data='label'):
            if label not in ['CWW', 'B53']:
                source_clust = motif_clust_map[source]
                if source_clust in self.graph.nodes():
                    query_nodes.add((source, source_clust))
                    # query_graph.add_node()

                target_clust = motif_clust_map[target]
                if target_clust in self.graph.nodes():
                    query_nodes.add((target, target_clust))

        query_edges = set()
        for (source, sid), (target, tid) in itertools.combinations(query_nodes, 2):
            try:
                distance = len(nx.shortest_path(nx_motif, source, target))
                # TODO : Find better cutoff
                if distance < 7:
                    query_edges.add((source, target, sid, tid, distance))
            except nx.NetworkXNoPath:
                pass
        return query_nodes, query_edges


class MGraphAll(MGraph):
    def __init__(self,
                 run,
                 graph_dir='../data/graphs/native',
                 n_components=8,
                 min_count=50,
                 max_var=0.1,
                 min_edge=50,
                 clust_algo='k_means',
                 optimize=True,
                 max_graphs=None,
                 nc_only=False,
                 bb_only=False):

        # General
        self.run = run
        self.graph_dir = graph_dir

        # Nodes parameters
        self.n_components = n_components
        self.min_count = min_count
        self.max_var = max_var
        self.clust_algo = clust_algo

        # Edges parameters
        self.bb_only= bb_only
        self.min_edge = min_edge

        # BUILD MNODES
        model_output = inference_on_list(self.run,
                                         self.graph_dir,
                                         os.listdir(self.graph_dir),
                                         max_graphs=max_graphs,
                                         nc_only=nc_only
                                         )

        Z = model_output['Z']
        self.node_map = model_output['node_to_zind']
        self.reversed_node_map = model_output['ind_to_node']

        print(len(Z))

        clust_info = cluster(Z,
                             algo=clust_algo,
                             optimize=optimize,
                             n_clusters=n_components)

        if self.clust_algo == 'gmm':
            distance = True
            self.cluster_model = clust_info['model']
            self.labels = clust_info['labels']
            if distance:
                centers = clust_info['centers']
                dists = cdist(Z, centers)
                scores = np.take_along_axis(dists, self.labels[:, None], axis=1)
                scores = np.exp(-scores)
            else:
                probas = clust_info['scores']
                scores = np.take_along_axis(probas, self.labels[:, None], axis=1)
        elif self.clust_algo == 'som':
            self.cluster_model = clust_info['model']
            self.labels = clust_info['labels']
            scores = np.exp(-clust_info['errors'])
        elif self.clust_algo == 'k_means':
            self.cluster_model = clust_info['model']
            self.labels = clust_info['labels']
            centers = clust_info['centers']
            dists = cdist(Z, centers)
            scores = np.take_along_axis(dists, self.labels[:, None], axis=1)
            scores = np.exp(-scores)
        else:
            raise NotImplementedError

        self.spread = clust_info['spread']

        self.components = np.unique(self.labels)
        self.id_to_score = {ind: scores[ind]
                            for ind, _ in self.reversed_node_map.items()}
        print("Clustered")

        self.graph = nx.Graph()

        # don't keep clusters that are too sparse or not populated enough
        # keep_clusts = cluster_filter(clusts, cov, self.min_count, self.max_var)
        # keep_clusts = set(keep_clusts)

        for id_clust in self.components:
            self.graph.add_node(id_clust, node_ids=set())

        for index, clust in enumerate(self.labels):
            self.graph.nodes[clust]['node_ids'].add(index)

        # BUILD MEDGES
        for graph_name in os.listdir(self.graph_dir)[:max_graphs]:
            graph_path = os.path.join(self.graph_dir, graph_name)
            g = fetch_graph(graph_path)
            g = g.to_undirected()
            for start_node, end_node in g.edges():
                # Get edges id
                if start_node not in self.node_map:
                    continue
                if end_node not in self.node_map:
                    continue
                if self.bb_only and g[start_node][end_node]['label'] != 'B53':
                    continue

                start_id, end_id = self.node_map[start_node], self.node_map[end_node]
                start_clust, end_clust = self.labels[start_id], self.labels[end_id]

                # Filter edges between discarded clusters
                # if start_clust not in keep_clusts or end_clust not in keep_clusts:
                # continue

                # Reorder and either create MEdge or complete it
                # if start_node > end_node:
                #     start_node, end_node = end_node, start_node
                # if start_clust > end_clust:
                #     start_clust, end_clust = end_clust, start_clust

                if not self.graph.has_edge(start_clust, end_clust):
                    self.graph.add_edge(start_clust, end_clust, edge_set=set())

                # self.graph.edges[(start_clust, end_clust)]['edge_set'].add((start_node, end_node, 1))
                self.graph.edges[(start_clust, end_clust)]['edge_set'].add((start_id, end_id, 1))

        # Filtering and hashing
        to_remove = list()
        for start, end, edge_set in self.graph.edges(data='edge_set'):
            # remove from adjacency
            if len(edge_set) < self.min_edge:
                to_remove.append((start, end))
        for start, end in to_remove:
            self.graph.remove_edge(start, end)

    def build_query_graph(self, original_graph, motif):
        """
        From an rna graph and the motif nodes flagged as such,
        return two sets with nodes that have cluster ids
        and edges between those clusters but cannot be a graph because of the several edges that can happen in a motif
        :param original_graph: a nx graph of the full chunk
        :param motif: a list of nodes flagged as motifs
        :return:
        """

        # BUILD MNODES
        Z, motif_node_map = inference_on_graph_run(self.run, graph=original_graph, verbose=False)

        # local_reversed_node_map = {value: key for key, value in motif_node_map.items()}
        # self.Z = Z

        nx_motif = original_graph.subgraph(motif)
        predictions = self.cluster_model.predict(Z)
        motif_clust_map = {node: predictions[motif_node_map[node]] for node in nx_motif}

        query_nodes = set()
        # query_graph = nx.Graph()
        # We have to add the nodes that are in kept clusters
        for motif_node in nx_motif.nodes():
            node_clust = motif_clust_map[motif_node]
            if node_clust in self.graph.nodes():
                query_nodes.add((motif_node, node_clust))

        query_edges = set()
        for start_node, end_node in nx_motif.edges():
            # Get edges id with a random 'start node' identifier to enable duplicates of clust to clust edge
            start_clust, end_clust = motif_clust_map[start_node], motif_clust_map[end_node]
            start_node, end_node = motif_node_map[start_node], motif_node_map[end_node]

            # Filter edges between discarded clusters
            if start_clust in self.graph.nodes() and end_clust in self.graph.nodes():
                query_edges.add((start_node, end_node, start_clust, end_clust, 1))

        return query_nodes, query_edges


def cluster_filter(clusts, cov, min_count, max_var):
    """
        Filters out nodes that don't meet criteria.
    """
    ct = Counter(clusts)
    print(ct)
    keep_clusts = []
    cov_fails, count_fails = 0, 0
    print(min_count)
    for i, var in enumerate(cov):
        var_ok, count_ok = [True] * 2
        if var > max_var:
            cov_fails += 1
            var_ok = False
        if ct[i] < min_count:
            count_fails += 1
            count_ok = False
        if var_ok and count_ok:
            keep_clusts.append(i)
    print(f">>> Covariance fails {cov_fails}, count fails {count_fails}")
    return keep_clusts


def get_embeddings_inference(run,
                             annot_path='../data/annotated/whole_v4',
                             max_graphs=300,
                             nc_only=False):
    """
        Build embedding matrix and graph list for clustering.
        Filters out nodes that don't have non-canonicals in neighbourhood.

    """
    from tools.learning_utils import inference_on_list
    annot_list = os.listdir(annot_path)[:max_graphs]
    keep_node_ids = []
    keep_inds = []
    # Get predictions
    model_output = inference_on_list(run, graph_list=annot_list, graphs_path=annot_path)
    Z = model_output['Z']
    node_to_ind = model_output['node_to_zind']
    node_ids = model_output['node_id_list']

    if not nc_only:
        return Z, node_to_ind
    for i, node in enumerate(node_ids):
        G = whole_graph_from_node(node)
        if has_NC_bfs(G, node, depth=1):
            keep_node_ids.append(node)
            keep_inds.append(i)

    print(f">>> got {len(Z)} nodes")
    return Z[keep_inds], keep_node_ids


if __name__ == "__main__":
    """
        RNA motifs
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="default_name",
                        help="The name of the pickled meta graph.")
    parser.add_argument('--run', type=str, default="1hop_weight",
                        help="The RGCN model to use.")
    parser.add_argument('--clust_algo', type=str, default="k_means",
                        help="The clustering algo to use to build the meta_graph : possible choices are"
                             "Gaussian Mixture Model, K-Means or Self-Organizing Maps")
    parser.add_argument("-N", "--n_components", type=int,
                        help="components in the clustering",
                        default=200)
    parser.add_argument('--prune', default=False, action='store_true',
                        help="To make the meta graph sparser, remove infrequent edges")
    parser.add_argument('--backbone', default=False, action='store_true',
                        help="If True, only connect via backbone.")
    parser.add_argument("--nc", default=False, action='store_true',
                        help="To use only nc"),
    args, _ = parser.parse_known_args()

    start = time.perf_counter()
    mgg = MGraphAll(run=args.run,
                    clust_algo=args.clust_algo,
                    n_components=args.n_components,
                    optimize=False,
                    min_count=100,
                    max_var=0.1,
                    min_edge=100,
                    max_graphs=None,
                    graph_dir='../data/unchopped_v4_nr',
                    nc_only=args.nc,
                    bb_only=args.backbone
                    )
    print(f"Built Meta Graph in {time.perf_counter() - start} s")

    if args.prune:
        mgg.prune()

    pickle.dump(mgg, open(args.name + '.p', 'wb'))
