"""
Draw RNA graphs for all files in a directory
"""


from drawing import *
import argparse
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', help='input directory')
    parser.add_argument('-i', help='use to draw one graph, graph file')
    parser.add_argument('o', help='output_directory')
    args = parser.parse_args()
    output_dir = args.o

    if args.I:
        print(output_dir)
        for graph_file in os.listdir(args.I):
            if '.nx' not in graph_file: continue
            g = nx.read_gpickle(os.path.join(args.I, graph_file))
            if len(list(g.nodes)) < 1: continue
            savefile = os.path.join(output_dir, graph_file[:-3]) + '.pdf'
            rna_draw(g, save=savefile)
            continue
    if args.i:
        g = nx.read_gpickle(args.i)
        savefile = os.path.join(output_dir, args.i[-9:-3]) + '.pdf'
        rna_draw(g, save=savefile)




if __name__ == '__main__':
    main()
