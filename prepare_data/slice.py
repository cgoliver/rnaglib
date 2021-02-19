import networkx as nx
import argparse
import os
import csv
import sys
from numpy import random

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from tools.graph_utils import bfs_expand, dangle_trim

class CustomError(Exception):
    pass

def get_largest_graph(graphs):

    largest = graphs[0]
    for g in graphs:
        if len(list(g.nodes)) > len(list(largest.nodes)):
            largest = g

    return largest

def get_pbid(g):

    return list(g.nodes)[0][0]

def print_node_data(g):

    print('graph: ', get_pbid(g))
    print('g.nodes:')
    for node in g.nodes:
        print(node, g.nodes[node])

def get_num_nodes(g):

    return len(list(g.nodes))

def get_label_key(g, node):

            pbid = get_pbid(g)
            pdb_pos = g.nodes[node]['pdb_pos']
            chain = g.nodes[node]['chain']

            return (pbid, chain, int(pdb_pos))

def connect_components(g, g_native, depth=2, trim_dangles=True):
    """
    Check if the graph contains disconnected components, connect them if it can be done
    with just a few edges. else return components as their own graphs
    Only coded for depth = 2.
    for bigger depths use recursive calls
    for a depth of just one do not use a nested neighbour search
    :param g: networkx graph
    :param g_native: native graph before interface was chopped
    :return graphs: list of connected graphs
    """
    graphs = []

    depth = int(depth/2)
    # if depth % 2 != 0:
        # inner_depth = int(depth/2)
        # outer_depth = int(depth/2) + 1
    # else:
        # inner_depth, outer_depth = depth/2

    pbid = get_pbid(g)

    if get_num_nodes(g) == 0:
        print("ERROR graph is null:", pbid, '\n\n')
        return [g]

    if nx.is_connected(g):
        return [g]
    else:

        connected_components = list(nx.connected_components(g))
        num_initial_components = len(connected_components)
        i = 0
        while(len(connected_components) > 1):

            # Remove a random component and gets its neighbours
            connected_nodes = component_outer = connected_components.pop()
            expanded_outer = bfs_expand(g_native, connected_nodes, depth=depth)

            # Search if neighbours overlap with other components neighbors
            for component_inner in connected_components:
                if component_inner == component_outer: continue
                expanded_inner = bfs_expand(g_native, component_inner, depth=depth)

                # If the neighbours of components intersect connect them
                neighbours_intersection = expanded_inner.intersection(expanded_outer)
                if len(neighbours_intersection) > 0:
                    connected_nodes = connected_nodes.union(neighbours_intersection)
                    connected_nodes = connected_nodes.union(component_inner)

            # Remove components that got merged
            for component_inner in connected_components:
                if component_inner.issubset(connected_nodes):
                    connected_components.remove(component_inner)

            # Add the new merge connected component
            connected_components.append(connected_nodes)

            # Create exit for inifinite loop (components cannot be connected)
            i += 1
            if i > (num_initial_components*10):
                break

    for component in connected_components:
        h = g_native.subgraph(list(component)).copy()
        if trim_dangles: dangle_trim(h)
        if len(list(h.nodes)) != 0:
            graphs.append(h)

    if graphs == []:
        raise CustomError


    return graphs

def balance_complement(interface, complement):
    """
    Count the number of nodes in the interface
    Do BFS_expand on a random node in the complement until number of nodes is equal
    :param interface: nx graph
    :param complement: nx graph
    :return balanced_complement: nx_graph the same size as the interface
    """

    num_nodes = get_num_nodes(interface)
    print(get_pbid(interface))
    print('Interface nodes: ', num_nodes)
    print('Complement nodes: ', get_num_nodes(complement))
    if num_nodes >= get_num_nodes(complement):
        print('No changes to complement, complement is smaller than interface')
        return complement, True



    random_node_idx = int(random.rand()*len(list(complement.nodes)))

    balanced_complement_nodes = [list(complement.nodes)[random_node_idx]]

    while(len(balanced_complement_nodes) < num_nodes):
        balanced_complement_nodes = bfs_expand(complement, balanced_complement_nodes, depth=1)

    balanced_complement = complement.subgraph(balanced_complement_nodes).copy()

    size_difference = abs( get_num_nodes(balanced_complement) - get_num_nodes(interface) )
    if size_difference > 10:
        print("\n WARNING Graph: ", get_pbid(interface),
                "has size_difference: ", size_difference, '\n\n')

    print('After Balancing, Complement nodes: ', get_num_nodes(balanced_complement))

    return balanced_complement, False

def add_labels(g, labels=None):

    if labels:
        for node in g.nodes:
            label_key = get_label_key(g, node)
            try:
                for key, value in labels[label_key].items():
                        g.nodes[node][key] = value
            except KeyError:
                # Nodes extended by connect components will 
                # have the same annotations as their neighbors
                no_valid_neighbor = True
                for neighbor in g.neighbors(node):
                    label_key = get_label_key(g, neighbor)
                    try:
                        for key, value in labels[label_key].items():
                                g.nodes[node][key] = value
                        no_valid_neighbor = False
                        break
                    except KeyError:
                           continue

                if no_valid_neighbor:
                    print('Warning: No valid neighbor with a set of attributes found for',
                            get_pbid(g))
    else:
        for node in g.nodes:
            for key in ['rna', 'protein', 'ligand', 'ion']:
                g.nodes[node][key] = None

def slice_graph(g, subset_info, quiet=False):
    """
    :param g:       graph
    :param subset_info:  list of 3-tuples containing info on the nodes to subset
                        format = [(pbid.nx, chain, pdb_pos), ...]

    :return g_subgraph: subgraph of nodes in subset
    :return g_prime:    complement of subgraph
    """
    g_prime = g.copy()

    if len(subset_info) == 0:
        print('ERROR: subset_info is empty for', get_pbid(g))
        raise CustomError

    subset = []
    for node in g.nodes:
        for _, chain, pdb_pos in subset_info:
            if int(g.nodes[node]['pdb_pos']) == pdb_pos and g.nodes[node]['chain'] == chain:
                subset.append(node)

    if len(subset) > 0:
        g_subgraph = g_prime.subgraph(subset).copy()
        g_prime.remove_nodes_from([n for n in g_prime if n in set(subset)])
    else:
        if not quiet:
            print('ERROR subset does not overlap with nodes in the graph', get_pbid(g))
            print_node_data(g)
            print('\nsubset_info:', subset_info, '\n')
        g_subgraph = None
        raise CustomError

    if get_num_nodes(g_subgraph) == 0:
        if not quiet:
            print('ERROR subset does not overlap any nodes in graph', get_pbid(g))
            print('\ng:', g.nodes)
            print('\n\nsubset: ', subset)

    return g_subgraph, g_prime

def slice_all(native_dir, output_dir, subset, quiet=False):
    """
    Slice all graphs in native_dir and writes them in the output_dir
    Connects all the sliced graphs organized them into distinct connected components
    (one connected graph for file, for graphs with more than one connected component
    more than one file is outputted)
    Complement Graphs are balanced to the same size as interface graphs
    (only if they are larger)

    :param native_dir: directory containing native PDB graphs
    :param output_dir: output directory for new interface and complement graphs
    :param subset: dictionary -> list -> dictionary -> dictionary
                i.e.
                subset[xxxx.nx] = {(xxxx.nx, A, 0) :
                                        {
                                            'rna': True,
                                            'protein': None,
                                            'ligand': None,
                                            'ion': None
                                        }
                                }
    (See parse_subset_fromcsv for more info)
    :return:
    """

    failed_slices = []

    for graph_file in os.listdir(native_dir):
        # print(f'Slicing RNA graph: \t {graph_file}')
        g = nx.read_gpickle(os.path.join(native_dir, graph_file))
        g_native = g.copy()
        pbid = list(g.nodes)[0][0]

        # Slice graph it has an interface in the subset
        if pbid in subset.keys():
            sub = subset[pbid].keys()
            try:
                g_subgraph, g_complement = slice_graph(g, sub)
            except CustomError:
                failed_slices.append((g, sub))
                continue

            if len(list(g_complement.nodes)) == 0:
                g_complement = None

            # Connect the components into a set of graphs
            try:
                interface_graphs = connect_components(g_subgraph, g_native)
            except CustomError:
            # skip if the interface is just dangles
                if not quiet: print('Only dangles found for graph: ', get_pbid(g_subgraph))
                continue
            if g_complement:
                try:
                    complement_graphs = connect_components(g_complement, g_native)
                    num_comps = len(complement_graphs)
                    largest_complement = get_largest_graph(complement_graphs)
                except CustomError:
                    if not quiet: print('Warning complement graph is only dangles',
                                        'Complement removed',
                                        get_pbid(g_complement))
                    g_complement = None

            # Balance and write output
            pbid_name = graph_file[:4]
            for i, h in enumerate(interface_graphs):

                # Balance complement
                if g_complement:
                    balanced_comp, too_small = balance_complement(h, largest_complement)
                    add_labels(balanced_comp)

                    # Add all complement graphs > 5 if complement is too small
                    if too_small:
                        j = 1
                        for comp in complement_graphs:
                            if get_num_nodes(comp) > 5:
                                add_labels(comp)
                                name = (pbid_name + '_' + str(i) +'_'+'C'+'_'+ str(j) +'.nx')
                                nx.write_gpickle(comp, os.path.join(output_dir, name))
                                j+=1
                    else:
                        name = (pbid_name + '_' + str(i) +'_'+'C'+'_'+ '0' +'.nx')
                        nx.write_gpickle(balanced_comp, os.path.join(output_dir, name ))

                # Add labels to the node attributes
                add_labels(h, subset[pbid])

                # Write output
                nx.write_gpickle(h, os.path.join(output_dir,
                                                (pbid_name + '_' + str(i) + '.nx') ))

    # Error Logging
    if len(failed_slices) > 0:
        with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
            if not quiet: print('some graph slicing failed, see log.txt for more details')
            f.write('These graphs failed to slice because the subset nodes did not\n')
            f.write('intersect the nodes in the graph\n')
            f.write('The graph is likely missing elements from the pdb original\n')
            f.write('i.e. dimer chains\n\n')
            for graph, subset in failed_slices:
                # print graphs that failed to slice
                if not quiet: print(graph)
                f.write('nodes in graph: ')
                f.write(get_pbid(graph))
                f.write('\n\n')
                # write node data to log
                for node in g.nodes:
                    f.write(str(node))
                    f.write('\t')
                    f.write(str(g.nodes[node]))
                    f.write('\n')
                f.write('\n\n subset:')
                for line in subset:
                    f.write(str(line))
                    f.write('\n')
                f.write('\n\n')


def parse_subset_fromcsv(subset_file, interaction_type):
    """
    Initialize subset from subset csv input argument
    :param subset_file:
    :param interaction_type: list of strings of interaction type
                            options = {protein, rna, ligand, ion, all}
    :return subset: dictionary of pbid.nx mapping to list of node data
                    each with attribute labels
    """
    subset = {}
    print('Building interface subset dictionary...')
    with open(subset_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = True
        for pbid, _, chain, curr_type, binding_molecule, position in reader:
            if header:
                header = False
                continue
            if curr_type not in interaction_type and 'all' not in interaction_type: continue
            pbid = pbid + '.nx'
            node = ( pbid, chain, int(position) )
            if pbid in subset.keys():
                if node not in subset[pbid].keys():
                    subset[pbid][node] = {
                                            'rna': None,
                                            'protein': None,
                                            'ligand': None,
                                            'ion': None
                                            }
            else:
                subset[pbid] = {node :  {
                                            'rna': None,
                                            'protein': None,
                                            'ligand': None,
                                            'ion': None
                                        }
                                }

            subset[pbid][node][curr_type] = binding_molecule

    return subset


