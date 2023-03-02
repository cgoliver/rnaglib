import random
from collections import Counter

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rnaglib.utils import available_pdbids
from rnaglib.utils import graph_from_pdbid

sns.set_theme(style="whitegrid")

graphs = [graph_from_pdbid(p) for p in available_pdbids()]

pockets = []
for i,G in enumerate(graphs):
        try:
            pocket = [n for n, data in G.nodes(data=True) if data['binding_small-molecule'] is not None]
            # sample same number of random nucleotides 
            non_pocket = random.sample(list(G.nodes()), k=len(pocket))
        except KeyError as e:
            continue
        if pocket:
            pockets.append((pocket, non_pocket, G))
        else:
            # no pocket found
            pass

bps, sses = [], []

for pocket, non_pocket, G in pockets:
    for nt in pocket:
        # add edge type of all base pairs in pocket
        bps.extend([{'bp_type': data['LW'],
                     'is_pocket': True} for _,data in G[nt].items()])
        # sse key is format '<sse type>_<id>'
        node_data = G.nodes[nt]
        if node_data['sse']['sse'] is None:
            continue
        sses.append({'sse_type': node_data['sse']['sse'].split("_")[0],
                     'is_pocket': True})

    # do the same for non-pocket
    for nt in non_pocket:
        # add edge type of all base pairs in pocket
        bps.extend([{'bp_type': data['LW'],
                     'is_pocket': False} for _,data in G[nt].items()])
        # sse key is format '<sse type>_<id>'
        node_data = G.nodes[nt]
        if node_data['sse']['sse'] is None:
            continue
        print(node_data['sse'])
        sses.append({'sse_type': node_data['sse']['sse'].split("_")[0],
                     'is_pocket':False})




# for convenience convert to dataframe
bp_df = pd.DataFrame(bps)
sse_df = pd.DataFrame(sses)

# remove backbone edges
bp_df = bp_df.loc[~bp_df['bp_type'].isin(['B35', 'B53'])]

sns.histplot(y='bp_type', hue='is_pocket', multiple='dodge', stat='proportion', data=bp_df)
sns.despine(left=True, bottom=True)
plt.savefig("bp.png")
plt.clf()

sns.histplot(y='sse_type', hue='is_pocket', multiple='dodge', stat='proportion', data=sse_df)
sns.despine(left=True, bottom=True)
plt.savefig("sse.png")
plt.clf()
