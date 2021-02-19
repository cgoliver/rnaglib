import xml.etree.cElementTree as ET
import argparse
import networkx as nx
from Bio.PDB import *

def get_coords(pdb):
    """
    {residue_id: (x, y, z)}
    """
    coords = {}
    for residue in pdb.get_residues():
        if residue.resname.strip() not in ['A', 'U', 'C', 'G']:
            #print(residue.resname)
            continue
        for atom in residue:
            if atom.id == 'P':
                key = (residue.get_parent().id, residue.id[1])
                coords[(residue.get_parent().id, residue.id[1])] = tuple(atom.coord)
                break
    return coords

def pdb_to_markers(pdb, graph, output):

    parser = MMCIFParser()
    structure = parser.get_structure('', pdb)

    root = ET.Element("marker_set", name="marker set 1")

    gr = nx.read_gpickle(graph)

    res_coords = get_coords(structure)
    # print(res_coords)

    res_to_id = lambda r: f"{r[0]}_{r[1]}"

    int_to_node = {}

    for i,node in enumerate(gr.nodes(data=True)):
        pdb_pos = node[1]['pdb_pos']
        chain = node[1]['chain']
        node = (chain, int(pdb_pos))

        try:
            x,y,z = res_coords[node]
        except KeyError:
            print(f"missing node coords {node}")
            continue

        int_to_node[node] = str(i)
        x,y,z = map(str, (x,y,z))
        ET.SubElement(root, "marker", id=str(i), x=x, y=y, z=z, radius="1")

    done_edges = set()
    for n1, n2, data in gr.edges(data=True):
        try:
            p1 = gr.nodes[n1]['pdb_pos']
            p2 = gr.nodes[n2]['pdb_pos']
            c1 = gr.nodes[n1]['chain']
            c2 = gr.nodes[n2]['chain']

            n1 = (c1, int(p1))
            n2 = (c1, int(p2))

            id_1 = int_to_node[n1]
            id_2 = int_to_node[n2]
        except KeyError:
            # print(n1, n2)
            continue
        else:
            if (id_1, id_2) not in done_edges and (id_2, id_1) not in done_edges:
                e_label = data['label']
                r,g,b = (255, 255, 255)
                if e_label not in ['B53', 'CWW']:
                    r,g,b = (255, 0, 0)
                if e_label == 'CWW':
                    r,g,b = (0, 255, 0)
                r,g,b = tuple(map(str, (r,g,b)))
                ET.SubElement(root, "link", id1=id_1, id2=id_2, radius="0.1", note=e_label,
                                r=r, g=g, b=b)
                done_edges.add((id_1, id_2))
                done_edges.add((id_2, id_1))

    tree = ET.ElementTree(root)
    tree.write(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb',
                        help='pdb file in cif format')
    parser.add_argument('graph',
                        help='networkx gpickle')
    parser.add_argument('output')
    args = parser.parse_args()

    pdb_to_markers(args.pdb, args.graph, args.output)



if __name__ == "__main__":
    main()
