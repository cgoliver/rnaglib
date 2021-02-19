import os, sys
import pickle
import argparse


import networkx as nx
import numpy as np
import matplotlib

# matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.realpath(__file__))


from rna_layout import circular_layout

params = {'text.latex.preamble': r'\usepackage{fdsymbol}\usepackage{xspace}'}
plt.rc('font', family='serif')
plt.rcParams.update(params)

labels = {
    'CW': r"$\medblackcircle$\xspace",
    'CS': r"$\medblacktriangleright$\xspace",
    'CH': r"$\medblacksquare$\xspace",
    'TW': r"$\medcircle$\xspace",
    'TS': r"$\medtriangleright$\xspace",
    'TH': r"$\medsquare$\xspace"
}

make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2 \
    else labels[s[:2]]


def process_axis(axis,
                 g,
                 subtitle=None,
                 highlight_edges=None,
                 node_color=None,
                 node_labels=None,
                 node_ids=False):
    pos = nx.spring_layout(g)

    if not node_color is None:
        nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color=node_color, linewidths=2, ax=axis)
    else:
        nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color='grey', linewidths=2, ax=axis)

    if node_ids:
        node_labels = {n: str(n).replace("_", "-") for n in nx_g.nodes()}
        nx.draw_networkx_labels(g, pos, node_labels, font_color='black', ax=axis)
    if not node_labels is None:
        nx.draw_networkx_labels(g, pos, node_labels, font_color='black', ax=axis)

    nodes.set_edgecolor('black')
    edge_labels = {}
    for n1, n2, d in g.edges(data=True):
        try:
            symbol = make_label(d['label'])
            edge_labels[(n1, n2)] = symbol
        except:
            if d['label'] == 'B53':
                edge_labels[(n1, n2)] = ''
            else:
                edge_labels[(n1, n2)] = r"{0}".format(d['label'])
            continue

    non_bb_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) if d['label'] == 'CWW']
    non_cw_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) \
                                            if d['label'] not in ['CWW', 'B53']]
    bb_edges = [(n1, n2) for n1, n2, d in g.edges(data=True) if d['label'] == 'B53']

    nx.draw_networkx_edges(g, pos, edgelist=non_bb_edges, width=2, edge_color='green', ax=axis)
    nx.draw_networkx_edges(g, pos, edgelist=non_cw_edges, width=2, edge_color='red', ax=axis)
    nx.draw_networkx_edges(g, pos, edgelist=bb_edges, width=2, ax=axis)

    if not highlight_edges is None:
        nx.draw_networkx_edges(g, pos,
                edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=axis)

    nx.draw_networkx_edge_labels(g, pos, font_size=16,
                                 edge_labels=edge_labels, ax=axis)
    axis.set_axis_off()
    if not subtitle is None:
        axis.set_title(subtitle)


def rna_draw(g, title="", node_ids=False,
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
             seed=None):
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
    # if pos is None:
    # pos = nx.spring_layout(nx_g, seed=seed)
    # new_pos = {}
    # for n, p in pos.items():
    #     new_pos[n] = scale * pos[n] + np.array(pos_offset)

    # pos = new_pos
    #
    # if node_colors is None:
    #     nodes = nx.draw_networkx_nodes(nx_g, pos,
    #                                    node_size=node_size,
    #                                    node_color='white',
    #                                    linewidths=2,
    #                                    ax=ax,
    #                                    alpha=alpha)
    # else:
    #     nodes = nx.draw_networkx_nodes(nx_g, pos,
    #                                    node_size=node_size,
    #                                    node_color=node_colors,
    #                                    linewidths=2,
    #                                    ax=ax,
    #                                    alpha=alpha)
    #
    # nodes.set_edgecolor('black')
    # if node_ids:
    #     node_labels = {n: str(n).replace("_", "-") for n in nx_g.nodes()}
    #     nx.draw_networkx_labels(nx_g, pos, node_labels, font_size=fontsize, font_color='black', ax=ax)
    # if not node_labels is None:
    #     nx.draw_networkx_labels(nx_g, pos, node_labels, font_size=fontsize, font_color='black', ax=ax)
    #
    # plt.title(r"{0}".format(title))
    # edge_labels = {}
    # for n1, n2, d in nx_g.edges(data=True):
    #     try:
    #         symbol = make_label(d['label'])
    #         edge_labels[(n1, n2)] = symbol
    #     except:
    #         if d['label'] in ['B53', 'pool']:
    #             edge_labels[(n1, n2)] = ''
    #         else:
    #             edge_labels[(n1, n2)] = r"{0}".format(d['label'])
    #         continue
    #
    # non_bb_edges = [(n1, n2) for n1, n2, d in nx_g.edges(data=True) if d['label'] not in ['B53', 'pool']]
    # bb_edges = [(n1, n2) for n1, n2, d in nx_g.edges(data=True) if d['label'] == 'B53']
    # pool_edges = [(n1, n2) for n1, n2, d in nx_g.edges(data=True) if d['label'] == 'pool']
    #
    # nx.draw_networkx_edges(nx_g, pos, edgelist=non_bb_edges, ax=ax)
    # nx.draw_networkx_edges(nx_g, pos, edgelist=bb_edges, width=2, ax=ax)
    # nx.draw_networkx_edges(nx_g, pos, edgelist=pool_edges, width=1, edge_color='grey', style='dashed', ax=ax)
    #
    # if not highlight_edges is None:
    #     nx.draw_networkx_edges(nx_g, pos,
    #                            edgelist=highlight_edges,
    #                            edge_color='y',
    #                            width=8,
    #                            alpha=0.3,
    #                            ax=ax)
    #
    # nx.draw_networkx_edge_labels(nx_g, pos, font_size=fontsize,
    #                              edge_labels=edge_labels, ax=ax)
    # ax.axis('off')

    process_axis(ax,
                 g,
                 subtitle=title,
                 highlight_edges=highlight_edges,
                 node_color=node_colors,
                 node_labels=node_labels)

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
    :param highlight_edges:
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


def rna_draw_grid(graphs, subtitles=None, highlight_edges=None, node_colors=None, row_labels=None,
                  save=None, show=False, grid_shape=None):
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
                process_axis(ax, g, subtitle=subtitles[i], highlight_edges=highlight_edges, node_color=node_colors)
    else:
        m, n = grid_shape
        assert m * n == len(graphs)
        fig, axes = plt.subplots(nrows=m, ncols=n)
        for i in range(len(graphs)):
            k, l = i // n, i % n
            process_axis(axes[k, l], graphs[i], subtitle=subtitles[i], highlight_edges=highlight_edges,
                         node_color=node_colors[i])

    if not row_labels is None:
        for a, row in zip(ax[:, 0], row_labels):
            a.set_ylabel(row, rotation=0)

    plt.axis('off')
    if save:
        plt.savefig(save, format='pdf')
    if show:
        plt.show()


def generic_draw_pair(graphs, title="", highlight_edges=None, node_colors=None, num_clusters=None):
    fig, ax = plt.subplots(1, len(graphs), num=1)
    for i, g in enumerate(graphs):
        pos = circular_layout(g)

        if not node_colors is None:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color=node_colors[i], linewidths=2, ax=ax[i])
        else:
            nodes = nx.draw_networkx_nodes(g, pos, node_size=150, node_color='grey', linewidths=2, ax=ax[i])

        nodes.set_edgecolor('black')

        # plt.title(r"{0}".format(title))
        edge_labels = {}
        for n1, n2, d in g.edges(data=True):
            edge_labels[(n1, n2)] = str(d['label'])

        if not highlight_edges is None:
            nx.draw_networkx_edges(g, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=ax[i])

        nx.draw_networkx_edge_labels(g, pos, font_size=16,
                                     edge_labels=edge_labels, ax=ax[i])
        ax[i].set_axis_off()

    plt.axis('off')
    plt.title(f"distance {title}")
    plt.show()


def generic_draw(graph, title="", highlight_edges=None, node_colors=None):
    fig, ax = plt.subplots(1, 2, num=1)
    pos = nx.spring_layout(graph)

    if not node_colors is None:
        nodes = nx.draw_networkx_nodes(graph, pos, node_size=150, cmap=plt.cm.Blues, node_color=node_colors,
                                       linewidths=2, ax=ax[0])
    else:
        nodes = nx.draw_networkx_nodes(graph, pos, node_size=150, node_color='grey', linewidths=2, ax=ax[0])

    nodes.set_edgecolor('black')

    # plt.title(r"{0}".format(title))
    edge_labels = {}
    for n1, n2, d in graph.edges(data=True):
        edge_labels[(n1, n2)] = str(d['label'])

    if not highlight_edges is None:
        nx.draw_networkx_edges(graph, pos, edgelist=highlight_edges, edge_color='y', width=8, alpha=0.5, ax=ax[0])

    nx.draw_networkx_edges(graph, pos, ax=ax[0])
    nx.draw_networkx_edge_labels(graph, pos, font_size=16,
                                 edge_labels=edge_labels, ax=ax[0])
    ax[0].set_axis_off()

    plt.axis('off')
    plt.title(f"motif {title}")
    plt.show()


def draw_hierarchical(graphs, node_colors=None, delta_y=1):
    """
        Draw hierarchical graph. Each subplot is one level of graph.
        links are edges between consecutive hierarchical levels.
    """
    # put all the graphs in one big one
    G = nx.Graph()
    if not node_colors is None:
        colors = []
        for n in node_colors:
            for c in n:
                colors.append(c)
        node_colors = colors

    for i, g in enumerate(graphs):
        g = nx.relabel_nodes(g, {n: (*n, i) for n in g.nodes()}, copy=True)
        if i == 0:
            pos = nx.spring_layout(g)
        G = nx.compose(G, g)

    import seaborn as sns
    pal = sns.color_palette("hls", 3)
    done_children = set()
    # add links using the 'children' attribute, from bottom graph to top one
    for i in range(1, len(graphs)):
        for n, d in graphs[i].nodes(data=True):
            pos[(n[0], n[1], i)] = pos[(n[0], n[1], i - 1)] + np.array([0, delta_y])
            for c in d['children']:
                if (c[0], c[1]) not in done_children or (c[0], c[1]) in graphs[i - 1].nodes():
                    G.add_edge((*n, i), (*c, i - 1), label='pool')
                    done_children.add((c[0], c[1]))
    rna_draw(G, pos=pos, node_colors=node_colors)
    # nx.draw_networkx(G, pos, node_size=20, edge_color=edge_colors, node_color=node_colors)
    plt.show()

    pass


def graph_align(g1, g2, ops, title="", save=None, rna=False):
    """
    Draw aligned graphs.
    """

    f = nx.compose(g1, g2)
    path = reversed(list(ops.path_iter()))
    cost = ops.cost
    # for o in path:
    edit_edges = []
    edge_list = []
    color_list = []
    for i, o in enumerate(path):
        e1, e2 = o.op
        if e1 == 'NILL' and e2 == 'NILL':
            continue
        f.add_edge(e1, e2, label='')
        edit_edges.append((e1, e2))

    c = plt.get_cmap('Paired')
    pos = nx.spring_layout(f)
    # pos = rna_layout.circular_layout(f)
    # nx.draw_networkx(f, pos, color='red')
    config = {'node_size': 500, 'alpha': 1, 'font_size': 25}
    nx.draw(f, pos, nodelist=['NILL'], color='red', **config)
    nx.draw_networkx_edges(f, pos,
                           edgelist=edit_edges, edge_color='grey', width=3)
    nx.draw_networkx_nodes(f, pos, nodelist=g1.nodes, node_color='blue', **config)
    nx.draw_networkx_edges(f, pos, edgelist=g1.edges)
    nx.draw_networkx_nodes(f, pos, nodelist=g2.nodes, node_color='green', **config)
    nx.draw_networkx_edges(f, pos, edgelist=g2.edges)

    make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2 \
        else labels[s[:2]]
    edge_labels = {}
    for e in f.edges.data():
        label = e[2]['label']
        if label == 'B53':
            label = ''
        edge_labels[(e[0], e[1])] = label

    if rna:
        import matplotlib
        # matplotlib.rcParams['text.usetex'] = False
        matplotlib.rcParams['text.usetex'] = True
        params = {'text.latex.preamble': [r'\usepackage{fdsymbol}\usepackage{xspace}']}
        plt.rcParams.update(params)

        labels = {
            'CW': r"$\medblackcircle$\xspace",
            'CS': r"$\medblacktriangleright$\xspace",
            'CH': r"$\medblacksquare$\xspace",
            'TW': r"$\medcircle$\xspace",
            'TS': r"$\medtriangleright$\xspace",
            'TH': r"$\medsquare$\xspace"
        }
        make_label = lambda s: labels[s[:2]] + labels[s[0::2]] if len(set(s[1:])) == 2 \
            else labels[s[:2]]
        edge_labels = {(e1, e2):
                           make_label(d['label']) if d['label'] not in ['B53', '']
                           else '' for e1, e2, d in f.edges.data()}
        nx.draw_networkx_edge_labels(f, pos, edge_labels=edge_labels)
    else:
        nx.draw_networkx_edge_labels(f, pos, edge_labels=edge_labels, font_size=config['font_size'])

    plt.title(f"GED: {cost} " + title)
    if save:
        plt.savefig(save, format="pdf")
    plt.show()

    pass



if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, ".."))
    main()
