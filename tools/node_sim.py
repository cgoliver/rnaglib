"""
Functions for comparing node similarity.
"""
import os, sys
import pickle
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pickle
import itertools

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

# from tools.graph_utils import bfs, bfs_expand
# from learning.timed_learning import train_model
# from tools.rna_ged import ged
# from tools.drawing import rna_draw_pair
from tools.graphlet_hash import *

# GLOBAL VARIABLES

iso_matrix = pickle.load(open(os.path.join(script_dir, '../data/iso_mat.p'), 'rb'))
# iso_matrix_ged = iso_matrix
iso_matrix = iso_matrix[1:, 1:]

# edge_map = {'B53': 0, 'CHH': 1, 'CHS': 2, 'CHW': 3, 'CSH': 4, 'CSS': 5, 'CSW': 6, 'CWH': 7, 'CWS': 8, 'CWW': 9,
#             'THH': 10, 'THS': 11, 'THW': 12, 'TSH': 13, 'TSS': 14, 'TSW': 15, 'TWH': 16, 'TWS': 17, 'TWW': 18}
EDGE_MAP = {'B53': 0, 'CHH': 1, 'CHS': 2, 'CHW': 3, 'CSH': 2, 'CSS': 4, 'CSW': 5, 'CWH': 3, 'CWS': 5, 'CWW': 6,
            'THH': 7, 'THS': 8, 'THW': 9, 'TSH': 8, 'TSS': 10, 'TSW': 11, 'TWH': 9, 'TWS': 11, 'TWW': 12}

IDF = {'TSS': 1.3508944643423815, 'TWW': 2.2521850545837103, 'CWW': 0.7302387734487946, 'B53': 0.6931471805599453,
       'CSS': 1.3562625353981017, 'TSH': 1.0617196804844624, 'THS': 1.0617196804844624, 'CSH': 1.6543492684466312,
       'CHS': 1.6543492684466312, 'THW': 1.3619066730630602, 'TWH': 1.3619066730630602, 'THH': 2.3624726636947186,
       'CWH': 2.220046456989285, 'CHW': 2.220046456989285, 'TSW': 2.3588208814802263, 'TWS': 2.3588208814802263,
       'CWS': 2.0236918714028707, 'CHH': 4.627784875752877, 'CSW': 2.0236918714028707}

indel_vector = [1 if e == 'B53' else 2 if e == 'CWW' else 3 for e in sorted(EDGE_MAP.keys())]


def simfunc_from_hparams(hparams):
    """

    :param hparams:
    :return:
    """
    node_simfunc = SimFunctionNode(method=hparams.get('argparse', 'sim_function'),
                                   depth=hparams.get('argparse', 'kernel_depth'),
                                   idf=hparams.get('argparse', 'idf'),
                                   hash_init=hparams.get('argparse', 'annotated_data'),
                                   decay=hparams.get('argparse', 'decay'),
                                   normalization=hparams.get('argparse', 'normalization'),
                                   edge_map=hparams.get('edges', 'edge_map'),
                                   )
    return node_simfunc


class SimFunctionNode():
    """
    Factory object to factor out the method choices from the function calls
    """

    def __init__(self,
                 method,
                 depth,
                 decay=0.5,
                 idf=False,
                 normalization=None,
                 hash_init='whole_v3',
                 edge_map=EDGE_MAP):

        POSSIBLE_METHODS = ['R_1', 'R_iso', 'R_graphlets', 'R_ged', 'hungarian', 'graphlet']
        assert method in POSSIBLE_METHODS

        self.method = method
        self.depth = depth
        self.decay = decay
        self.normalization = normalization

        self.edge_map = edge_map

        if self.method in ['R_ged', 'R_graphlets', 'graphlet']:
            self.GED_table = defaultdict(dict)
            init_path = os.path.join(script_dir, '..', 'data', 'hashing', hash_init + '.p')
            print(f">>> loading hash table from {init_path}")
            self.hasher, self.hash_table = \
                pickle.load(open(init_path, 'rb'))

        if idf:
            global IDF
            self.idf = IDF
        else:
            self.idf = None

        if self.method in ['R_1', 'R_iso', 'R_graphlets']:
            # depth+1 and -1 term at the end account for skipping the 0th hop in the rings.
            # we increase the size of the geometric sum by 1 and subtract 1 to remove the 0th term.
            self.norm_factor = ((1 - self.decay ** (self.depth + 1)) / (1 - self.decay)) - 1
        else:
            self.norm_factor = 1.0

    def compare(self, rings1, rings2, debug=False):
        """
            Compares first K rings at each level.
            Takes two ring lists: [[None], [first ring], [second ring],...]
            or
            Dealing with 0th hop for 'edge' annotation is ambiguous so we added a None first hop,
                when we say depth=3, we want rings[1:4], hence range(1, depth+1)
            Need to take this into account for normalization (see class constructor)
        """
        # if self.depth < 1 or self.depth > len(rings1):
        # raise ValueError("depth must be 1 <= depth <= number_of_rings ")

        if self.method == 'graphlet':
            return self.graphlet(rings1, rings2)

        if self.method == 'hungarian':
            return self.hungarian(rings1, rings2)

        res = 0

        if self.method == 'R_graphlets':
            for k in range(0, self.depth):
                value = self.R_graphlets(rings1[k], rings2[k])
                res += self.decay ** (k + 1) * value
        else:
            for k in range(1, self.depth + 1):
                if self.method == 'R_1':
                    value = self.R_1(rings1[k], rings2[k])
                else:
                    value = self.R_iso(rings1[k], rings2[k])
                res += self.decay ** k * value
        return res / self.norm_factor

    def normalize(self, unnormalized, length):
        """
        We want our normalization to be more lenient to longer matches
        :param unnormalized: a score in [0, length]
        :param length: the best possible matching score of the sequences we are given
        :return: a score in [0,1]
        """

        # print(f'mine, {(unnormalized / length) ** power}, usual {(unnormalized / length)}')
        if self.normalization == 'sqrt':
            power = (1 / (np.sqrt(length) / 5))
            return (unnormalized / length) ** power
        elif self.normalization == 'log':
            power = (1 / (np.log(length) + 1))
            return (unnormalized / length) ** power
        return unnormalized / length

    def get_length(self, ring1, ring2, graphlets=False):
        """
        This is meant to return an adapted 'length' based on the sum of IDF terms if it is used
        :param ring1:
        :param ring2:
        :return:
        """
        if self.idf is None or graphlets:
            return max(len(ring1), len(ring2))
        else:
            # This is what you obtain when matching all the nodes to themselves
            return max(sum([self.idf[node] ** 2 for node in ring1]), sum([self.idf[node] ** 2 for node in ring2]))

    @staticmethod
    def delta_indices_sim(i, j, distance=False):
        """
        We need a scoring related to matching different nodes

        Returns a positive score in [0,1]
        :param i:
        :param j:
        :return:
        """
        if distance:
            return 1 - np.exp(-abs(i - j))
        return np.exp(-abs(i - j))

    def get_cost_nodes(self, node_i, node_j, bb=False, pos=False):
        """
        Compare two nodes and returns a cost.

        Returns a positive number that has to be negated for minimization

        :param iso_matrix:
        :param bb : Check if what is being compared is backbone (no isostericity then)
        :param pos : Check if this is used within a ring (no indices then)
        :return:
        """
        global iso_matrix
        if bb:
            res = SimFunctionNode.delta_indices_sim(node_i[1], node_j[1])
            if self.idf is not None:
                return res * self.idf[node_i[0]] * self.idf[node_j[0]]
            else:
                return res

        # If we only have node type information
        elif pos is False:
            res = iso_matrix[self.edge_map[node_i] - 1, self.edge_map[node_j] - 1]
            if self.idf is not None:
                return res * self.idf[node_i] * self.idf[node_j]
            else:
                return res

        else:
            res = SimFunctionNode.delta_indices_sim(node_i[1], node_j[1]) + iso_matrix[
                self.edge_map[node_i[0]] - 1, self.edge_map[node_j[0]] - 1]
            if self.idf is not None:
                return res * self.idf[node_i[0]] * self.idf[node_j[0]]
            else:
                return res

    def R_1(self, ring1, ring2):
        """
        Compute R function over lists of features:
        first attempt : count intersect and normalise by the number (Jacard?)
        :param ring1: list of features
        :param ring2: ''
        :return:
        """
        feat_1 = Counter(ring1)
        feat_2 = Counter(ring2)

        # sometimes ring is empty which throws division by zero error
        if len(feat_1) == 0 and len(feat_2) == 0:
            return 1

        if self.idf:
            num = 0
            den = 0
            minmax = lambda x: (min(x), max(x))
            for e, w in self.idf.items():
                mi, ma = minmax((feat_1[e], feat_2[e]))
                num += w * mi
                den += w * ma
            return num / den

        else:
            diff = feat_1 & feat_2
            hist = feat_1 | feat_2
            return sum(diff.values()) / sum(hist.values())

    def R_iso(self, list1, list2):
        """
        Compute R function over lists of features:
        first attempt : count intersect and normalise by the number (Jacard?)
        :param list1: list of features
        :param list2: ''
        :return:
        """
        feat_1 = Counter(list1)
        feat_2 = Counter(list2)

        # === First deal with BB ===
        def R_1_like_bb(count1, count2):
            if count1 == 0 == count2:
                return 1
            loc_min, loc_max = min(count1, count2), max(count1, count2)

            return (loc_min / loc_max) ** 1.5

        def exp_dist(count1, count2):
            """Exponentially weighted but does not depend on lengths"""
            diff_bb = abs(- feat_2['B53'])
            return np.exp(-diff_bb)

        # sim_bb = exp_dist(feat_1['B53'], feat_2['B53'])
        sim_bb = R_1_like_bb(feat_1['B53'], feat_2['B53'])

        # === Then deal with NC ===

        # On average the edge rings only have 0.68 edges that are not BB
        # Therefore bruteforcing is acceptable
        nc_list1 = [i for i in list1 if i != 'B53']
        nc_list2 = [i for i in list2 if i != 'B53']

        def compare_smooth(ring1, ring2):
            """
            Compare two lists of non backbone
            :param ring1:
            :param ring2:
            :return:
            """
            if len(ring1) == 0 and len(ring2) == 0:
                return 1
            if len(ring1) == 0 or len(ring2) == 0:
                return 0

            # This makes loading go from 0.17 to 2.6
            # cost = np.array(self.get_cost_matrix(ring1, ring2))
            cost = np.array([[self.get_cost_nodes(node_i, node_j) for node_j in ring2] for node_i in ring1])
            cost = -cost
            row_ind, col_ind = linear_sum_assignment(cost)
            unnormalized = - np.array(cost[row_ind, col_ind]).sum()

            length = self.get_length(ring1, ring2)
            return self.normalize(unnormalized, length)

        def compare_brute(ring1, ring2):
            """
            Bruteforce the hungarian problem since it is pretty sparse. Test all permutation assignment of the longest list
            :param ring1:
            :param ring2:
            :return:
            """
            if len(ring1) == 0 and len(ring2) == 0:
                return 1
            if len(ring1) == 0 or len(ring2) == 0:
                return 0

            # Get the longest first to permute it to get the exact solution
            if len(ring2) > len(ring1):
                ring1, ring2 = ring2, ring1
            perms = set(itertools.permutations(ring1))
            all_costs = [sum([self.get_cost_nodes(perm[i], node_j) for i, node_j in enumerate(ring2)]) for
                         perm in perms]
            unnormalized = max(all_costs)

            length = self.get_length(ring1, ring2)
            return self.normalize(unnormalized, length)

        # This was computed empirically, for small rings, brute is faster, but for longer one, we should get smart
        if len(nc_list1) < 6 and len(nc_list2) < 6:
            sim_non_bb = compare_brute(nc_list1, nc_list2)

        else:
            sim_non_bb = compare_smooth(nc_list1, nc_list2)
        # They do return the same thing :)
        # print(sim_non_bb - sim_non_bb_nobrute)
        # print(time_used, time_used1)
        # global time_res_brute, time_res_smooth
        # time_res_brute[len(nc_list1), len(nc_list2)] += time_brute
        # time_res_smooth[len(nc_list1), len(nc_list2)] += time_smooth

        return (sim_non_bb + sim_bb) / 2

    def R_graphlets(self, list1, list2):

        """
        Compute R function over lists of features:
        first attempt : count intersect and normalise by the number (Jacard?)
        :param list1: list of features
        :param list2: ''
        :return:
        """

        def compare_smooth(ring1, ring2):
            """
            Compare two lists of non backbone
            :param ring1:
            :param ring2:
            :return:
            """
            if len(ring1) == 0 and len(ring2) == 0:
                return 1
            if len(ring1) == 0 or len(ring2) == 0:
                return 0

            # Try the similarity version the bonus we get is that we use the normalization also used for riso
            # And therefore we avoid a double exponential
            cost = - np.array(
                [[self.graphlet_cost_nodes(node_i, node_j, similarity=True) for node_j in ring2] for node_i in ring1])
            row_ind, col_ind = linear_sum_assignment(cost)
            unnormalized = - np.array(cost[row_ind, col_ind]).sum()

            length = self.get_length(ring1, ring2, graphlets=True)
            return self.normalize(unnormalized, length)

        def compare_brute(ring1, ring2):
            """
            Bruteforce the hungarian problem since it is pretty sparse. Test all permutation assignment of the longest list
            :param ring1:
            :param ring2:
            :return:
            """
            if len(ring1) == 0 and len(ring2) == 0:
                return 1
            if len(ring1) == 0 or len(ring2) == 0:
                return 0

            # Try the similarity version the bonus we get is that we use the normalization also used for riso
            # And therefore we avoid a double exponential

            if len(ring2) > len(ring1):
                ring1, ring2 = ring2, ring1
            perms = set(itertools.permutations(ring1))
            all_costs = [
                sum([self.graphlet_cost_nodes(perm[i], node_j, similarity=True) for i, node_j in enumerate(ring2)]) for
                perm in perms]
            unnormalized = max(all_costs)

            length = self.get_length(ring1, ring2, graphlets=True)
            return self.normalize(unnormalized, length)

        # This was computed empirically, for small rings, brute is faster, but for longer one, we should get smart
        if len(list1) < 6 and len(list2) < 6:
            sim_non_bb = compare_brute(list1, list2)
        else:
            sim_non_bb = compare_smooth(list1, list2)
        # They do return the same thing :)
        # print(sim_non_bb - sim_non_bb_nobrute)
        # print(time_used, time_used1)
        # global time_res_brute, time_res_smooth
        # time_res_brute[len(nc_list1), len(nc_list2)] += time_brute
        # time_res_smooth[len(nc_list1), len(nc_list2)] += time_smooth

        return sim_non_bb

    def hungarian(self, rings1, rings2):
        """
        Formulate the kernel as an assignment problem
        :param rings1: list of lists
        :param rings2:
        :return:
        """

        def rings_to_lists(rings, depth):
            can, noncan = [], []
            for k in range(1, depth + 1):
                for value in rings[k]:
                    if value == 'B53':
                        can.append((value, k))
                    else:
                        noncan.append((value, k))
            return can, noncan

        def compare_lists(ring1, ring2, bb=False, pos=False):
            if len(ring1) == 0 and len(ring2) == 0:
                return 1
            if len(ring1) == 0 or len(ring2) == 0:
                return 0
            cm = [[self.get_cost_nodes(node_i, node_j, bb=bb, pos=pos) for node_j in ring2] for node_i in ring1]
            # Dont forget the minus for minimization
            cost = -np.array(cm)
            row_ind, col_ind = linear_sum_assignment(cost)
            unnormalized = - np.array(cost[row_ind, col_ind]).sum()
            # If the cost also includes distance information, we need to divide by two
            factor_two = 2 if bb is False and len(ring1[0]) == 2 else 1
            length = self.get_length([node[0] for node in ring1], [node[0] for node in ring2])
            return self.normalize(unnormalized / factor_two, length)

        can1, noncan1 = rings_to_lists(rings1, depth=self.depth)
        can2, noncan2 = rings_to_lists(rings2, depth=self.depth)

        cost_can = compare_lists(can1, can2, bb=True)
        cost_noncan = compare_lists(noncan1, noncan2, bb=False, pos=True)

        return (cost_can + cost_noncan) / 2

    def graphlet(self, rings1, rings2):
        """
            Compare graphlet rings using GED and memoizing.

            1. Get list of graphlets and distances in subgraph: [(sG_hash_1, d_1), ..].
            2. Build graphlet distance matrix DM with GED
            3. Match the subgraphs with Hungarian, using DM for cost.
            4. Return kernel value
        """

        def rings_to_lists_g(rings, depth):
            ringlist = []
            for k in range(depth):
                for value in rings[k]:
                    ringlist.append((value, k))
            return ringlist

        ringlist1 = rings_to_lists_g(rings1, depth=self.depth)
        ringlist2 = rings_to_lists_g(rings2, depth=self.depth)

        cost = - np.array(
            [[self.graphlet_cost_nodes(node_i, node_j, pos=True, similarity=True) for node_j in ringlist2]
             for node_i in ringlist1])
        row_ind, col_ind = linear_sum_assignment(cost)
        unnormalized = - np.array(cost[row_ind, col_ind]).sum()

        # Because we have pos
        unnormalized /= 2

        length = self.get_length(ringlist1, ringlist2, graphlets=True)
        return self.normalize(unnormalized, length)

        '''
        cost_matrix = np.array(
            [[self.graphlet_cost_nodes(node_i, node_j, pos=True) for node_j in ringlist2] for node_i in ringlist1])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # This is a distance score, we turn in into a similarity with exp(-x)
        cost_raw = cost_matrix[row_ind, col_ind].sum()
        normed = np.exp(-cost_raw)
        return normed
        '''

    def graphlet_cost_nodes(self, node1, node2, pos=False, similarity=False):
        """
        Returns a node distance between nodes represented as graphlets
        :param node1:
        :param node2:
        :param pos: if pos is true, nodes are expected to be (graphlet, distance from root)
        :return:
        """
        if pos:
            g_1, p_1 = node1
            g_2, p_2 = node2
            ged = GED_hashtable_hashed(g_1, g_2, self.GED_table, self.hash_table, normed=True, similarity=similarity)
            delta = SimFunctionNode.delta_indices_sim(p_1, p_2, distance=not similarity)
            return ged + delta
        else:
            return GED_hashtable_hashed(node1, node2, self.GED_table, self.hash_table, normed=True,
                                        similarity=similarity)


def graph_edge_freqs(graphs, stop=0):
    """
        Get IDF for each edge label over whole dataset.

        {'CWW': 110, 'TWW': 23}
    """
    graph_counts = Counter()
    # get document frequencies
    num_docs = 0
    for graph in graphs:
        labels = {e['label'] for _, _, e in graph.edges(data=True)}
        graph_counts.update(labels)
        num_docs += 1
        if num_docs > 100 and stop:
            break
    return {k: np.log(num_docs / graph_counts[k] + 1) for k in graph_counts}


def pdist_list(rings, node_sim):
    """
    Defines the block creation using a list of rings at the graph level (should also ultimately include trees)
    Creates a SIMILARITY matrix.
    :param rings: a list of rings, dictionnaries {node : (nodelist, edgelist)}
    :param node_sim: the pairwise node comparison function
    :return:
    """
    rings_values = [list(ring.values()) for ring in rings]
    nodes = list(itertools.chain.from_iterable(rings_values))
    assert node_sim.compare(nodes[0][1], nodes[0][1]) == 1, "Identical rings giving non 1 similarity."

    sims = [node_sim.compare(n1[1], n2[1])
            for i, (n1, n2) in enumerate(itertools.combinations(nodes, 2))]

    return sims


def k_block_list(rings, node_sim):
    """
    Defines the block creation using a list of rings at the graph level (should also ultimately include trees)
    Creates a SIMILARITY matrix.
    :param rings: a list of rings, dictionnaries {node : (nodelist, edgelist)}
    :param node_sim: the pairwise node comparison function
    :return:
    """

    rings_values = [list(ring.values()) for ring in rings]
    nodes = list(itertools.chain.from_iterable(rings_values))
    block = np.zeros((len(nodes), len(nodes)))
    b = node_sim.compare(nodes[0], nodes[0])
    assert node_sim.compare(nodes[0], nodes[0]) > 0.99, "Identical rings giving non 1 similarity."
    sims = [node_sim.compare(n1, n2)
            for i, (n1, n2) in enumerate(itertools.combinations(nodes, 2))]
    block[np.triu_indices(len(nodes), 1)] = sims
    block += block.T

    block += np.eye(len(nodes))

    return block


def simfunc_time(simfuncs, graph_path, batches=1, batch_size=5,
                 names=None):
    """
        Do time benchmark on a simfunc.
    """
    from random import shuffle
    from time import perf_counter

    import pandas as pd

    rows = []
    graphlist = os.listdir(graph_path)
    for ind, simfunc in enumerate(simfuncs):
        print(f">>> DOING KERNEL {simfunc.method}")
        level = 'graphlet' if simfunc.method == 'graphlet' else 'edge'
        batch_times = []
        for b in range(batches):
            shuffle(graphlist)
            loading_times = []
            ringlist = []
            print(f">>> batch {b}")
            for i in range(batch_size):
                start = perf_counter()
                G = pickle.load(open(os.path.join(graph_path, graphlist[i]), 'rb'))
                loading_times.append(perf_counter() - start)
                graph = G['graph']
                for node in graph.nodes():
                    ringlist.append(G['rings'][level][node])

            print(f">>> tot batch loading, {sum(loading_times)}")
            print(f">>> avg time per loading, {np.mean(loading_times)}")
            print(f">>> max loading, {max(loading_times)}")
            print(f">>> min loading, {min(loading_times)}")

            times = []
            for i, r1 in enumerate(ringlist):
                for j, r2 in enumerate(ringlist[i:]):
                    start = perf_counter()
                    k = simfunc.compare(r1, r2)
                    t = perf_counter() - start
                    times.append(t)
            print(f">>> batch size {batch_size}")
            print(f">>> total batch time {sum(times)}")
            print(f">>> avg time per comparison, {np.mean(times)}")
            print(f">>> max comparison, {max(times)}")
            print(f">>> min comparison, {min(times)}")
            # batch_times.append((sum(times) + sum(loading_times)) / len(times))
            batch_time = sum(times) + sum(loading_times)
            rows.append({'batch_time': batch_time,
                         'kernel': simfunc.method,
                         'comparisons': len(times),
                         'batch_num': b
                         })
            batch_times.append(batch_time)
        if not names is None:
            label = names[ind]
        else:
            label = simfunc.method
        plt.plot(batch_times, label=label)
        plt.xlabel("Batch")
        plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("../figs/time_2.pdf", format="pdf")
    # plt.show()

    df = pd.DataFrame.from_dict(rows)
    df.to_csv('kernel_times_2.csv')
    pass


if __name__ == "__main__":
    pass
    # k_block_all("../data/chunks_nx_annot", "../data/test_sim")
    # a = [[None], ['CWW', 'B53', 'B53'], ['B53', 'B53', 'B53'], ['B53', 'B53'], ['B53'], ['CWW', 'B53']]
    # ring1 = [[None], ['B53', 'B53', 'CSS'], ['B53', 'B53'], ['THW', 'B53'], ['B53', 'B53', 'B35']]
    # simfunc = SimFunctionNode(method='R_1', idf=False, depth=3, decay=0.8, normalization='sqrt')
    # k = simfunc.compare(ring1, ring1)
    # print(k)

    # value = hungarian(ring1, ring1, 3)
    # print(value)
    graph_path = os.path.join("..", "data", "annotated", "whole_v4")
    graphs = os.listdir(graph_path)
    data1 = pickle.load(open(os.path.join(graph_path, graphs[0]), 'rb'))
    data2 = pickle.load(open(os.path.join(graph_path, graphs[1]), 'rb'))
    G, rings1 = data1['graph'], data1['rings']['graphlet']
    H, rings2 = data2['graph'], data2['rings']['graphlet']
    # G, rings1 = data1['graph'], data1['rings']['edge']
    # H, rings2 = data2['graph'], data2['rings']['edge']
    # simfunc_r1 = SimFunctionNode('R_1', 2)
    # simfunc_hung = SimFunctionNode('hungarian', 2, hash_init='whole_v3')
    # simfunc_iso = SimFunctionNode('R_iso', 2, hash_init='whole_v3', idf=True)
    # simfunc_graphlet = SimFunctionNode('graphlet', 2, hash_init='whole_v3')
    simfunc_graphlet = SimFunctionNode('R_graphlets', 2, hash_init='whole_v4')

    for node1, ring1 in rings1.items():
        for node2, ring2 in rings2.items():
            a = simfunc_graphlet.compare(ring1, ring2)
            b = simfunc_graphlet.compare(ring1, ring1)
            # a = simfunc_r1.compare(ring1, ring2)
            # b = simfunc_r1.compare(ring1, ring1)
            # print(a)

    # ring1 = list(rings1.values())[0]
    # print(simfunc_iso.compare(ring1, ring1))

    # simfunc_time([simfunc_graphlet], graph_path, batches=100)
    # simfunc_time([simfunc_r1, simfunc_graphlet, simfunc_iso, simfunc_hung], graph_path, batches=10)
