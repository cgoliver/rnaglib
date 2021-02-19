import networkx as nx
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-I')
    args = parser.parse_args()

    for file in os.listdir(args.I):
        path = os.path.join(args.I, file)
        g = nx.read_gpickle(path)
        non_cano = 0
        n = len(list(g.nodes))
        # print(file)
        for _, _, data in g.edges(data=True):
            if data['label'] not in ['B53', 'CWW']:
                non_cano += 1
        if 2 <= non_cano <= 4 and n < 10:
            print('2 to 4 non_canos and n<10: ', file)

if __name__ == '__main__':
    main()
