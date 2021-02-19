import argparse
import os
import sys
import networkx as nx
from numpy import random

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..'))

from prepare_data.slice import *
from prepare_data.interfaces import *

from tools.graph_utils import bfs_expand, dangle_trim

def get_pbid(g):

    return list(g.nodes)[0][0]

def get_num_nodes(g):

    return len(list(g.nodes))

def print_component_info(g, i):


    pbid = list(g.nodes)[0][0]
    # print(f'\n\nedges: {list(g.nodes)[0][0]} \n')
    # for e in g.edges:
    # print(e)


    if nx.is_connected(g):
        # print(pbid, i, ': connected')
        pass
    else:
        print("\n\n WARNING \n",
                pbid, i, ': not connected \t components:', nx.number_connected_components(g))

    # for n, nbrs in g.adj.items():
        # for nbr, eattr in nbrs.items():
            # print(eattr)

   # Remove nodes that have no other interactions other than backbone

def print_num_nodes(native, interface, complement):

    pbid = get_pbid(native)
    print(pbid, '\t', get_num_nodes(native), '\t', get_num_nodes(interface),
            '\t', get_num_nodes(complement))
    if get_num_nodes(interface) + get_num_nodes(complement) != get_num_nodes(native):
        print('SLICING ERROR')

def print_num_nodes_all(native_dir, interface_dir, complement_dir):

    i = 0
    for graph_file in os.listdir(native_dir):
        # Loop control
        # if i == 30: break
        # i += 1
        if '.nx' not in graph_file: continue

        # read input and compute function
        try:
            interface = nx.read_gpickle(os.path.join(interface_dir, graph_file))
        except FileNotFoundError:
            print('\nWARNING, interface graph not found for: ',
                    graph_file, '\n')
            break
        try:
            complement = nx.read_gpickle(os.path.join(complement_dir, graph_file))
        except FileNotFoundError:
            print('\nWARNING, complement graph not found for: ',
                    graph_file, '\n\n')
            continue
        try:
            native = nx.read_gpickle(os.path.join(native_dir, graph_file))
        except FileNotFoundError:
            print('\nWARNING, native graph not found for: ',
                    graph_file, '\n\n')
            continue
        print_num_nodes(native, interface, complement)

    return

def balance_complement_all(interface_dir, complement_dir, output_dir):

    i = 0
    for graph_file in os.listdir(interface_dir):
        # Loop control
        # if i == 30: break
        # i += 1
        if '.nx' not in graph_file: continue

        # read input and compute function
        try:
            interface = nx.read_gpickle(os.path.join(interface_dir, graph_file))
            complement = nx.read_gpickle(os.path.join(complement_dir, graph_file))
            print('Balancing', graph_file, '...')
        except FileNotFoundError:
            print('\nWARNING, complement graph not found for: ', graph_file, '\n\n')
            continue
        balanced_complement = balance_complement(interface, complement)

        # Write output
        nx.write_gpickle(balanced_complement, os.path.join(output_dir, graph_file))


def connect_all(input_dir, native_dir, output_dir):
    """
    runs connect_components on all graphs in input_dir and outputs
    resulting connected graphs to output_dir
    """
    i = 0
    for graph_file in os.listdir(input_dir):
        # Loop control
        # if i == 30: break
        # i += 1
        if '.nx' not in graph_file: continue

        # read input and compute function
        g = nx.read_gpickle(os.path.join(input_dir, graph_file))
        g_native = nx.read_gpickle(os.path.join(native_dir, graph_file))
        connected_graphs = connect_components(g, g_native)

        # Write output
        pbid = graph_file[:4]
        for i, h in enumerate(connected_graphs):
            nx.write_gpickle(h, os.path.join(output_dir, (pbid + '_' + str(i) + '.nx') ))
            print_component_info(h, i)

def connect_and_balance_all(interface_dir, native_dir, complement_dir, output_dir,
                            quiet=False):
    """
    UNFINISHED: STILL NEED TO WRITE DESCRIPTION
    """
    # Make a directory inside output for the complements
    try:
        os.mkdir(os.path.join(output_dir, 'complement'))
    except FileExistsError:
        print('complement directory already exists! make sure you are not overwriting')
    comp_dir = os.path.join(output_dir, 'complement')

    i = 0
    for graph_file in os.listdir(interface_dir):
        #Loop control
        if i == 30: break
        i += 1
        if '.nx' not in graph_file: continue

        if not quiet: print("Connecting and Balancing graph", graph_file )
        # read interface, complement and native graphs
        g = nx.read_gpickle(os.path.join(interface_dir, graph_file))
        g_native = nx.read_gpickle(os.path.join(native_dir, graph_file))
        g_complement = nx.read_gpickle(os.path.join(complement_dir, graph_file))

        # Connect the components into a set of graphs
        interface_graphs = connect_components(g, g_native)
        complement_graphs = connect_components(g_complement, g_native, trim_dangles=False)
        num_comps = len(complement_graphs)

        # Balance and write output
        pbid = graph_file[:4]
        for i, h in enumerate(interface_graphs):
            balanced_comp = balance_complement(h, complement_graphs[i%num_comps])
            nx.write_gpickle(h, os.path.join(output_dir, (pbid + '_' + str(i) + '.nx') ))
            nx.write_gpickle(balanced_comp, os.path.join(comp_dir,
                                                        (pbid + '_' + str(i) + '.nx') ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--interface_dir',
                        help= 'Directory containing interface graphs')
    parser.add_argument('-C', '--complement_dir',
                        help= 'Directory containing complement interface graphs')
    parser.add_argument('-N', '--native_dir',
                        help='directory containing native unchopped graphs',
                        default = os.path.join(scriptdir, '..', 'data', 'graphs', 'native'))
    parser.add_argument('-O', '--output',
                        help='output directory to store cleaned up graphs')
    args = parser.parse_args()

    #connect_all(args.interface_dir, args.native_dir, args.output)

    #balance_complement_all(args.interface_dir, args.complement_dir, args.output)
    #print_num_nodes_all(args.native_dir, args.interface_dir, args.complement_dir)
    connect_and_balance_all(args.interface_dir, args.native_dir, args.complement_dir,
                            args.output)

if __name__ == '__main__':
    main()
