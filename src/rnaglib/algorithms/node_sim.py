"""
Functions for comparing node similarity.
"""

import os

from collections import defaultdict, Counter
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import linear_sum_assignment

from rnaglib.config.graph_keys import GRAPH_KEYS, TOOL
from rnaglib.config.build_iso_mat import iso_mat as iso_matrix
from .graphlet_hash import *

script_dir = os.path.dirname(os.path.abspath(__file__))


class SimFunctionNode:

    def __init__(
        self,
        method,
        depth,
        decay=0.5,
        idf=False,
        normalization=None,
        hash_init_path=os.path.join(script_dir, "..", "data", "hashing", "NR_chops_hash.p"),
    ):
        """
        Factory object to compute all node similarities.
        These methods take as input an annotated pair of nodes
        and compare them.

        These methods are detailed in the supplemental of the paper, but include five methods.
        These methods frequently
        rely on the hungarian algorithm, an algorithm that finds optimal matches according to a cost function.

        Three of them compare the edges :

        - R_1 compares the histograms of each ring, possibly with an idf weighting (to emphasize differences
          in rare edges)
        - R_iso compares each ring with the best matching based on the isostericity values
        - hungarian compares the whole annotation, with the rings being
        differentiated with an additional 'depth' field.

        Then all the nodes are compared based on isostericity and this depth field.

        Two of them compare the graphlets. The underlying idea is that just
        comparing lists of edges does not
        constraint the graph structure, while the assembling of graphlet does it more
        (exceptions can be found but
        for most graphs, knowing all graphlets at each depth enables recreating the graph) :

        - R_graphlets works like R_iso except that the isostericity is replaced by the GED
        - graphlet works like the hungarian except that the isostericity is replaced by the GED

        :param method: a string that identifies which of these method to use
        :param depth: The depth to use in the annotations rings
        :param decay: When using rings comparison function, the weight decay of importance based on the depth (the
        closest rings weigh more as they are central to the comparison)
        :param idf: Whether to use IDF weighting on the frequency of labels.
        :param normalization: We experiment with three normalization scheme,
                              the basal one is just a division of the
        score by the maximum value, 'sqrt' denotes using the square root of the ratio as a power of the raw value and
        'log' uses the log. The underlying idea is that we want to put more emphasis on the long matches than on just
        matching a few nodes
        :param hash_init_path: For the graphlets comparisons, we need to supply a hashing path to be able to store the
        values of ged and reuse them based on the hash.
        """

        POSSIBLE_METHODS = {"R_1", "R_iso", "hungarian", "R_graphlets", "graphlet"}
        assert method in POSSIBLE_METHODS

        self.method = method
        self.depth = depth
        self.decay = decay
        self.normalization = normalization

        self.hash_init_path = hash_init_path

        edge_map = GRAPH_KEYS["edge_map"][TOOL]
        self.edge_map = edge_map

        # Placeholders for the hashing information. We defer its creation into the call of the object for parallel use.
        self.hasher = None
        self.GED_table = defaultdict(dict)
        self.hash_table = {}

        if idf:
            self.idf = GRAPH_KEYS["idf"][TOOL]
        else:
            self.idf = None

        if self.method in ["R_1", "R_iso", "R_graphlets"]:
            # depth+1 and -1 term at the end account for skipping the 0th hop in the rings.
            # we increase the size of the geometric sum by 1 and subtract 1 to remove the 0th term.
            self.norm_factor = ((1 - self.decay ** (self.depth + 1)) / (1 - self.decay)) - 1
        else:
            self.norm_factor = 1.0

    def add_hashtable(self, hash_init_path):
        """

        :param hash_init_path: A string with the full path to a pickled hashtable
        :return: None, modify self.
        """
        print(f">>> loading hash table from {hash_init_path}")
        self.hasher, self.hash_table = pickle.load(open(hash_init_path, "rb"))

    def compare(self, rings1, rings2):
        """
        Compares two nodes represented as their rings.

        The edge list for the first hop (centered around a node) is None, so it gets skipped, when we say depth=3,
        we want rings[1:4], hence range(1, depth+1) Need to take this into account for normalization

        (see class constructor)

         :param rings1: A list of rings at each depth. Rings contain a list of node, edge or graphlet information at a
         given distance from a central node.
         :param rings2: Same as above for another node.
         :return: Normalized comparison score between the nodes
        """
        # We only load the hashing table when we make a first computation, a lazy optimization
        if self.method in ["R_ged", "R_graphlets", "graphlet"] and self.hasher is None:
            self.add_hashtable(hash_init_path=self.hash_init_path)

        if self.method == "graphlet":
            return self.graphlet(rings1, rings2)

        if self.method == "hungarian":
            return self.hungarian(rings1, rings2)

        res = 0
        if self.method == "R_graphlets":
            for k in range(0, self.depth):
                value = self.R_graphlets(rings1[k], rings2[k])
                res += self.decay ** (k + 1) * value
        else:
            for k in range(1, self.depth + 1):
                if self.method == "R_1":
                    value = self.R_1(rings1[k], rings2[k])
                else:
                    value = self.R_iso(rings1[k], rings2[k])
                res += self.decay**k * value
        return res / self.norm_factor

    def normalize(self, unnormalized, max_score):
        """
        We want our normalization to be more lenient to longer matches

        :param unnormalized: a score in [0, max_score]
        :param max_score: the best possible matching score of the sequences we are given
        :return: a score in [0,1]
        """
        if self.normalization == "sqrt":
            power = 1 / (np.sqrt(max_score) / 5)
            return (unnormalized / max_score) ** power
        elif self.normalization == "log":
            power = 1 / (np.log(max_score) + 1)
            return (unnormalized / max_score) ** power
        return unnormalized / max_score

    def get_length(self, ring1, ring2, graphlets=False):
        """
        This is meant to return an adapted 'length' that represents the optimal score obtained when matching all the
        elements in the two rings at hands

         :param rings1: A list of rings at each depth. Rings contain a list of node, edge or graphlet information at a
         given distance from a central node.
         :param rings2: Same as above for another node.
         :param graphlets: Whether we use graphlets instead of edges. Then no isostericity can be used to compute length
        :return: a float that represents the score of a perfect match
        """
        if self.idf is None or graphlets:
            return max(len(ring1), len(ring2))
        else:
            # This is what you obtain when matching all the nodes to themselves
            return max(
                sum([self.idf[node] ** 2 for node in ring1]),
                sum([self.idf[node] ** 2 for node in ring2]),
            )

    @staticmethod
    def delta_indices_sim(i, j, distance=False):
        """
        We need a scoring related to matching different depth nodes.
        Returns a positive score in [0,1]

        :param i: pos of the first node
        :param j: pos of the second node
        :return: A normalized value of their distance (exp(abs(i-j))
        """
        if distance:
            return 1 - np.exp(-abs(i - j))
        return np.exp(-abs(i - j))

    def get_cost_nodes(self, node_i, node_j, bb=False, pos=False):
        """
        Compare two nodes and returns a cost.

        Returns a positive number that has to be negated for minimization

        :param node_i : This either just contains a label to be compared with isostericity, or a tuple that also
        includes distance from the root node
        :param node_j : Same as above
        :param bb : Check if what is being compared is backbone (no isostericity then)
        :param pos: if pos is true, nodes are expected to be (edge label, distance from root) else just a edge label.
        pos is True when used from a comparison between nodes from different rings
        :return: the cost of matching those two nodes
        """
        global iso_matrix

        score = 0
        # If we have distance info, extract it and compute a first component of the score.
        if bb or pos:
            node_i_type, node_i_depth = node_i
            node_j_type, node_j_depth = node_j
            res_distance = SimFunctionNode.delta_indices_sim(node_i_depth, node_j_depth)
            score += res_distance
        else:
            node_i_type = node_i
            node_j_type = node_j

        # If we are not bb, also use isostericity.
        if not bb:
            res_isostericity = iso_matrix[self.edge_map[node_i_type], self.edge_map[node_j_type]]
            score += res_isostericity

        if self.idf is not None:
            return score * self.idf[node_i[0]] * self.idf[node_j[0]]
        else:
            return score

    def R_1(self, ring1, ring2):
        """
        Compute R_1 function over lists of features by counting intersect and normalise by the number

        :param ring1: list of features
        :param ring2: Same as above for other node
        :return: Score
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
        Compute R_iso function over lists of features by matching each ring with
        the hungarian algorithm on the iso values

        We do a separate computation for backbone.

        :param list1: list of features
        :param list2: ''
        :return: Score
        """
        feat_1 = Counter(list1)
        feat_2 = Counter(list2)

        # === First deal with BB ===
        def R_1_like_bb(count1, count2):
            if count1 == 0 == count2:
                return 1
            loc_min, loc_max = min(count1, count2), max(count1, count2)

            return (loc_min / loc_max) ** 1.5

        sim_bb_53 = R_1_like_bb(feat_1["B53"], feat_2["B53"])
        sim_bb_35 = R_1_like_bb(feat_1["B35"], feat_2["B35"])
        sim_bb = (sim_bb_53 + sim_bb_35) / 2

        # === Then deal with NC ===

        # On average the edge rings only have 0.68 edges that are not BB
        # Therefore bruteforcing is acceptable
        nc_list1 = [i for i in list1 if i not in ["B53", "B35"]]
        nc_list2 = [i for i in list2 if i not in ["B53", "B35"]]

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
            unnormalized = -np.array(cost[row_ind, col_ind]).sum()

            length = self.get_length(ring1, ring2)
            return self.normalize(unnormalized, length)

        def compare_brute(ring1, ring2):
            """
            Bruteforce the hungarian problem since it is pretty sparse.
            Test all permutation assignment of the longest list

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
            all_costs = [
                sum([self.get_cost_nodes(perm[i], node_j) for i, node_j in enumerate(ring2)]) for perm in perms
            ]
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

    def hungarian(self, rings1, rings2):
        """
        Compute hungarian function over lists of features by adding a depth field into each ring (based on its index
        in rings). Then we try to match all nodes together, to deal with bulges for instances.

        We do a separate computation for backbone.

        :param list1: list of features
        :param list2: ''
        :return: Score
        """

        def rings_to_lists(rings, depth):
            can, noncan = [], []
            for k in range(1, depth + 1):
                for value in rings[k]:
                    if value in ["B53", "B35"]:
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
            unnormalized = -np.array(cost[row_ind, col_ind]).sum()
            # If the cost also includes distance information, we need to divide by two
            factor_two = 2 if bb is False and len(ring1[0]) == 2 else 1
            length = self.get_length([node[0] for node in ring1], [node[0] for node in ring2])
            return self.normalize(unnormalized / factor_two, length)

        can1, noncan1 = rings_to_lists(rings1, depth=self.depth)
        can2, noncan2 = rings_to_lists(rings2, depth=self.depth)

        cost_can = compare_lists(can1, can2, bb=True)
        cost_noncan = compare_lists(noncan1, noncan2, bb=False, pos=True)

        return (cost_can + cost_noncan) / 2

    def graphlet_cost_nodes(self, node_i, node_j, pos=False, similarity=False):
        """
        Returns a node distance between nodes represented as graphlets
                Compare two nodes and returns a cost.

        Returns a positive number that has to be negated for minimization

        :param node_i : This either just contains a label to be compared with isostericity, or a tuple that also
        includes distance from the root node
        :param node_j : Same as above
        :param pos: if pos is true, nodes are expected to be (graphlet, distance from root) else just a graphlet.
        pos is True when used from a comparison between nodes from different rings
        :return: the cost of matching those two nodes
        """
        if pos:
            g_1, p_1 = node_i
            g_2, p_2 = node_j
            ged = get_ged_hashtable(
                g_1,
                g_2,
                self.GED_table,
                self.hash_table,
                normed=True,
                similarity=similarity,
            )
            delta = SimFunctionNode.delta_indices_sim(p_1, p_2, distance=not similarity)
            return ged + delta
        else:
            return get_ged_hashtable(
                node_i,
                node_j,
                self.GED_table,
                self.hash_table,
                normed=True,
                similarity=similarity,
            )

    def R_graphlets(self, ring1, ring2):
        """
        Compute R_graphlets function over lists of features.

        :param ring1: list of list of graphlets
        :param ring2: Same as above for other node
        :return: Score
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
            cost = -np.array(
                [[self.graphlet_cost_nodes(node_i, node_j, similarity=True) for node_j in ring2] for node_i in ring1]
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            unnormalized = -np.array(cost[row_ind, col_ind]).sum()

            length = self.get_length(ring1, ring2, graphlets=True)
            return self.normalize(unnormalized, length)

        def compare_brute(ring1, ring2):
            """
            Bruteforce the hungarian problem since it is pretty sparse.
            Test all permutation assignment of the longest list

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
                sum([self.graphlet_cost_nodes(perm[i], node_j, similarity=True) for i, node_j in enumerate(ring2)])
                for perm in perms
            ]
            unnormalized = max(all_costs)

            length = self.get_length(ring1, ring2, graphlets=True)
            return self.normalize(unnormalized, length)

        # This was computed empirically, for small rings, brute is faster, but for longer one, we should get smart
        if len(ring1) < 6 and len(ring2) < 6:
            sim_non_bb = compare_brute(ring1, ring2)
        else:
            sim_non_bb = compare_smooth(ring1, ring2)
        # They do return the same thing :)
        # print(sim_non_bb - sim_non_bb_nobrute)
        # print(time_used, time_used1)
        # global time_res_brute, time_res_smooth
        # time_res_brute[len(nc_list1), len(nc_list2)] += time_brute
        # time_res_smooth[len(nc_list1), len(nc_list2)] += time_smooth

        return sim_non_bb

    def graphlet(self, rings1, rings2):
        """
        This function performs an operation similar to the hungarian algorithm using ged between graphlets instead of
        isostericity.

        We also add a distance to root node attribute to each graphlet and then match them optimally

        :param ring1: list of list of graphlets
        :param ring2: Same as above for other node
        :return: Score
        """

        def rings_to_lists_g(rings, depth):
            ringlist = []
            for k in range(depth):
                for value in rings[k]:
                    ringlist.append((value, k))
            return ringlist

        ringlist1 = rings_to_lists_g(rings1, depth=self.depth)
        ringlist2 = rings_to_lists_g(rings2, depth=self.depth)

        cost = -np.array(
            [
                [self.graphlet_cost_nodes(node_i, node_j, pos=True, similarity=True) for node_j in ringlist2]
                for node_i in ringlist1
            ]
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        unnormalized = -np.array(cost[row_ind, col_ind]).sum()

        # Because we have pos
        unnormalized /= 2

        length = self.get_length(ringlist1, ringlist2, graphlets=True)
        return self.normalize(unnormalized, length)


def graph_edge_freqs(graphs, stop=False):
    """
    Get IDF for each edge label over whole dataset.
    First get a total frequency dictionnary :{'CWW': 110, 'TWW': 23}
    Then compute IDF and returns the value.

    :param graphs: The graphs over which to compute the frequencies, a list of nx graphs
    :param stop: Set to True for just doing it on a subset
    :return: A dict with the idf values.
    """
    graph_counts = Counter()
    # get document frequencies
    num_docs = 0
    for graph in graphs:
        labels = {e["label"] for _, _, e in graph.edges(data=True)}
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
    :return: the upper triangle of a similarity matrix, in the form of a list
    """
    rings_values = [list(ring.values()) for ring in rings]
    nodes = list(itertools.chain.from_iterable(rings_values))
    assert node_sim.compare(nodes[0][1], nodes[0][1]) == 1, "Identical rings giving non 1 similarity."

    sims = [node_sim.compare(n1[1], n2[1]) for i, (n1, n2) in enumerate(itertools.combinations(nodes, 2))]

    return sims


def k_block_list(rings, node_sim):
    """
    Defines the block creation using a list of rings at the graph level (should also ultimately include trees)
    Creates a SIMILARITY matrix.

    :param rings: a list of rings, dictionnaries {node : (nodelist, edgelist)}
    :param node_sim: the pairwise node comparison function
    :return: A whole similarity matrix in the form of a numpy array that follows the order of rings
    """

    node_rings = [ring_values for node, ring_values in rings]

    # rings_values = [list(ring.values()) for ring in rings]
    # node_rings = list(itertools.chain.from_iterable(rings_values))
    block = np.zeros((len(node_rings), len(node_rings)))
    assert node_sim.compare(node_rings[0], node_rings[0]) > 0.99, "Identical rings giving non 1 similarity."
    sims = [node_sim.compare(n1, n2) for i, (n1, n2) in enumerate(itertools.combinations(node_rings, 2))]
    block[np.triu_indices(len(node_rings), 1)] = sims
    block += block.T
    block += np.eye(len(node_rings))
    return block


def simfunc_time(simfuncs, graph_path, batches=1, batch_size=5, names=None):
    """
    Do time benchmark on a list of simfunc.

    :param simfuncs:
    :param graph_path:
    :param batches:
    :param batch_size:
    :param names:
    :return:
    """
    from random import shuffle
    from time import perf_counter

    import pandas as pd

    rows = []
    graphlist = os.listdir(graph_path)
    for ind, simfunc in enumerate(simfuncs):
        print(f">>> DOING KERNEL {simfunc.method}")
        level = "graphlet" if simfunc.method == "graphlet" else "edge"
        batch_times = []
        for b in range(batches):
            shuffle(graphlist)
            loading_times = []
            ringlist = []
            print(f">>> batch {b}")
            for i in range(batch_size):
                start = perf_counter()
                G = pickle.load(open(os.path.join(graph_path, graphlist[i]), "rb"))
                loading_times.append(perf_counter() - start)
                graph = G["graph"]
                for node in graph.nodes():
                    ringlist.append(G["rings"][level][node])

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
            rows.append(
                {
                    "batch_time": batch_time,
                    "kernel": simfunc.method,
                    "comparisons": len(times),
                    "batch_num": b,
                }
            )
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
    df.to_csv("kernel_times_2.csv")
    pass


if __name__ == "__main__":
    pass

    hash_graphlets_to_use = "thursday"
    graph_path = os.path.join("..", "data", "annotated", hash_graphlets_to_use)
    graphs = os.listdir(graph_path)
    data1 = pickle.load(open(os.path.join(graph_path, graphs[0]), "rb"))
    data2 = pickle.load(open(os.path.join(graph_path, graphs[1]), "rb"))
    G, rings1 = data1["graph"], data1["rings"]["graphlet"]
    H, rings2 = data2["graph"], data2["rings"]["graphlet"]
    # G, rings1 = data1['graph'], data1['rings']['edge']
    # H, rings2 = data2['graph'], data2['rings']['edge']
    simfunc_r1 = SimFunctionNode("R_1", 2)
    simfunc_hung = SimFunctionNode("hungarian", 2)
    simfunc_iso = SimFunctionNode("R_iso", 2, idf=True)
    simfunc_r_graphlet = SimFunctionNode("R_graphlets", 2, hash_init=hash_graphlets_to_use)
    simfunc_graphlet = SimFunctionNode("graphlet", 2, hash_init=hash_graphlets_to_use)
    simfunc = simfunc_r_graphlet
    for node1, ring1 in rings1.items():
        for node2, ring2 in rings2.items():
            a = simfunc.compare(ring1, ring2)
            b = simfunc.compare(ring1, ring1)
            print(a)
            print(b)

    # ring1 = list(rings1.values())[0]
    # print(simfunc_iso.compare(ring1, ring1))

    # simfunc_time([simfunc_graphlet], graph_path, batches=100)
    # simfunc_time([simfunc_r1, simfunc_graphlet, simfunc_iso, simfunc_hung], graph_path, batches=10)
