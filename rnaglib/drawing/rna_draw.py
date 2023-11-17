import os
import sys

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from rnaglib.utils.graph_io import load_json
from rnaglib.drawing.rna_layout import circular_layout
from distutils.spawn import find_executable

use_tex = False
if find_executable('latex'):
    use_tex = True
else:
    print("No LaTex installation was found, using a fallback drawing system.")

if use_tex:
    matplotlib.rcParams['text.usetex'] = True
    params = {'text.latex.preamble': r'\usepackage{fdsymbol}\usepackage{xspace}'}
    plt.rc('font', family='serif')
    plt.rcParams.update(params)
    labels = {
        'cW': r"$\medblackcircle$\xspace",
        'cS': r"$\medblacktriangleright$\xspace",
        'cH': r"$\medblacksquare$\xspace",
        'tW': r"$\medcircle$\xspace",
        'tS': r"$\medtriangleright$\xspace",
        'tH': r"$\medsquare$\xspace"
    }
else:
    labels = {
        'cW': r"$\oplus\ $",
        'cS': r"$\blacktriangleright\ $",
        'cH': r"$\blacksquare\ $",
        'tW': r"$\bigcirc\ $",
        'tS': r"$\triangleright\ $",
        'tH': r"$\boxdot\ $",
    }

make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2 else labels[s[:2]]


NT_COLORS = {'A': 'blue', 'U': 'green', 'C': 'red', 'G': 'yellow', 'a': 'blue', 'u': 'green', 'c': 'red', 'g': 'yellow'}

def process_axis(axis,
                 g,
                 subtitle=None,
                 highlight_edges=None,
                 node_color=None,
                 node_labels=None,
                 node_ids=False,
                 layout='spring',
                 label='LW'):
    """
     Draw a graph  on a given axis.

    :param axis: matplotlib axis to draw on
    :param g: networkx graph to draw
    :param subtitle: string to use as a subtitle on this axis
    :param highlight_edges: A list of edges to highlight on the drawing
    :param node_color:
    :param node_labels:
    :param node_ids:
    :param label:
    :return:
    """
    if layout == 'spring':
        pos = nx.spring_layout(g)
    else:
        pos = circular_layout(g)

    if not node_color is None:
        nodes = nx.draw_networkx_nodes(g, pos, node_size=50, node_color=node_color, linewidths=1, ax=axis)
    else:
        nt_color = []
        for node, d in g.nodes(data=True):
            try:
                #nt_color.append(NT_COLORS[d['nt_code']])
                nt_color.append(NT_COLORS[d['nt_code']])
            except:
                nt_color.append('grey')

        nodes = nx.draw_networkx_nodes(g,
                                       pos,
                                       node_size=50,
                                       node_color=nt_color,
                                       linewidths=1,
                                       ax=axis)

    if node_ids:
        node_labels = {n: str(n).replace("_", "-") for n in g.nodes()}
        nx.draw_networkx_labels(g, pos, node_labels, font_color='black', ax=axis)
    if not node_labels is None:
        nx.draw_networkx_labels(g, pos, node_labels, font_color='black', ax=axis)

    nodes.set_edgecolor('black')
    edge_labels = {}
    for n1, n2, d in g.edges(data=True):
        try:
            symbol = make_label(d[label])
            edge_labels[(n1, n2)] = symbol
        except:
            if d[label] == 'B53' or d[label] == 'B35':
                edge_labels[(n1, n2)] = ''
            else:
                edge_labels[(n1, n2)] = r"{0}".format(d[label])
            continue

    non_bb_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) if d[label][0] != 'B']
    # bb_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) if d[label][0] == 'B']
    bb_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) if d[label] == 'B53']

    nx.draw_networkx_edges(g, pos, edge_color="red", edgelist=non_bb_edges, ax=axis)
    # nx.draw_networkx_edges(g, pos, edge_color="red", connectionstyle="arc3,rad=0.1", edgelist=non_bb_edges, ax=axis)
    nx.draw_networkx_edges(g, pos, edgelist=bb_edges, width=1, ax=axis)

    if not highlight_edges is None:
        nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=axis)

    nx.draw_networkx_edge_labels(g,
                                 pos,
                                 font_size=10,
                                 edge_labels=edge_labels,
                                 ax=axis)
    axis.set_axis_off()
    if not subtitle is None:
        axis.set_title(subtitle)


def rna_draw(g, title="",
             node_ids=False,
             highlight_edges=None,
             node_labels=None,
             node_colors=None,
             num_clusters=None,
             pos=None,
             pos_offset=(0, 0),
             scale=1,
             ax=None,
             show=False,
             alpha=1,
             save=False,
             node_size=250,
             fontsize=12,
             format='pdf',
             seed=None,
             layout='circular'):
    """
    Draw an RNA with the edge labels used by Leontis Westhof

    :param nx_g:
    :param title:
    :param highlight_edges:
    :param node_colors:
    :param num_clusters:
    :return:
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    pos = circular_layout(g)

    process_axis(ax,
                 g,
                 subtitle=title,
                 highlight_edges=highlight_edges,
                 node_color=node_colors,
                 node_labels=node_labels,
                 layout=layout)

    if save:
        plt.savefig(save, format=format)
        plt.clf()
    if show:
        plt.show()
    return ax
    # plt.clf()


def rna_draw_pair(graphs, subtitles=None, highlight_edges=None, node_colors=None,
                  save=None, show=False, node_ids=False):
    """
    Plot a line of plots of graphs along with a value for each graph. Useful for graph comparison vizualisation

    :param graphs: iterable nx graphs
    :param estimated_value: iterable of values of comparison (optional)
    :param iihighlight_edges:
    :param node_colors: iterable of node colors
    :return:
    """
    fig, ax = plt.subplots(1, len(graphs), num=1)
    for i, g in enumerate(graphs):
        subtitle, node_color = (None, None)
        if not subtitles is None:
            subtitle = subtitles[i]
        if not node_colors is None:
            node_color = node_colors[i]

        process_axis(ax[i],
                     g,
                     subtitle=subtitle,
                     highlight_edges=highlight_edges,
                     node_color=node_color)
    plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(save, format='pdf')
    if show:
        plt.show()


def rna_draw_grid(graphs,
                  subtitles=None,
                  highlight_edges=None,
                  node_colors=None,
                  row_labels=None,
                  save=None,
                  show=False,
                  format='png',
                  grid_shape=None):
    """
    Plot a line of plots of graphs along with a value for each graph. Useful for graph comparison vizualisation

    :param graphs: list of lists containing nx graphs all lists must have the same dimension along axis 1. To skip a cell, add a None instead of graph.
    :param estimated_value: iterable of values of comparison (optional)
    :param highlight_edges:
    :param node_colors: iterable of node colors
    :return:
    """
    if grid_shape is None:
        assert len(set(map(len, graphs))) == 1, "All rows must have the same number of entries."
    if not subtitles is None:
        assert len(set(map(len, subtitles))) == 1, "All rows must have the same number of entries."

        N = len(graphs)
        M = len(graphs[0])
        fig, ax = plt.subplots(N, M, num=1)
        for i, gs in enumerate(graphs):
            for j, g in enumerate(gs):
                process_axis(ax,
                             g,
                             subtitle=subtitles[i],
                             highlight_edges=highlight_edges,
                             node_color=node_colors)
    else:
        m, n = grid_shape
        assert m * n == len(graphs)
        fig, axes = plt.subplots(nrows=m, ncols=n)
        for i in range(len(graphs)):
            k, l = i // n, i % n
            process_axis(axes[k, l],
                         graphs[i],
                         subtitle='',
                         highlight_edges=highlight_edges,
                         node_color='grey')

    if not row_labels is None:
        for a, row in zip(ax[:, 0], row_labels):
            a.set_ylabel(row, rotation=0)

    plt.axis('off')
    if save:
        plt.savefig(save, format=save_format)
    if show:
        plt.show()


if __name__ == "__main__":
    G = load_json("data/examples/4nlf.json")
    rna_draw(G, show=True)
    # for f in os.listdir("data/all_graphs_chops"):
    #     G = nx.read_gpickle(os.path.join("data/all_graphs_chops", f))
    #     print("HI")
    #     print(f)
    #     rna_draw(G, show=True, format="pdf")
    pass
