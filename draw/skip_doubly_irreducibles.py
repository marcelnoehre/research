from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

file = '43'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
lat = ConceptLattice.from_context(cxt)
G     = nx.transitive_reduction(lat.to_networkx())
hasse = G.copy()   # original structure for DI detection and chain walking

# doubly irreducible: exactly one predecessor and one successor in the Hasse diagram
di_nodes = {n for n in hasse.nodes()
            if hasse.in_degree(n) == 1 and hasse.out_degree(n) == 1}

# for every maximal chain of DI nodes, add skip edges for every contiguous sub-chain
visited = set()
for start in di_nodes:
    if start in visited:
        continue
    pred = next(iter(hasse.predecessors(start)))
    if pred in di_nodes:
        continue                          # not the start of a maximal chain
    chain = []
    cur = start
    while cur in di_nodes:
        chain.append(cur)
        visited.add(cur)
        cur = next(iter(hasse.successors(cur)))
    # nodes = [node-before-chain] + chain + [node-after-chain]
    nodes = [pred] + chain + [cur]
    for i in range(1, len(nodes) - 1):
        for j in range(i, len(nodes) - 1):
            G.add_edge(nodes[i - 1], nodes[j + 1])

# --- layout -----------------------------------------------------------------
# heights via DP on the original Hasse diagram (length of longest path to bot)
heights = {}
for n in reversed(list(nx.topological_sort(hasse))):
    succs = list(hasse.successors(n))
    heights[n] = 0 if not succs else 1 + max(heights[s] for s in succs)

# place DI nodes at midpoint between their Hasse neighbours
for n in di_nodes:
    above = next(iter(hasse.predecessors(n)))
    below = next(iter(hasse.successors(n)))
    heights[n] = heights[below] + (heights[above] - heights[below]) / 2

# x: spread nodes evenly within each height level
by_height = defaultdict(list)
for n, h in heights.items():
    by_height[h].append(n)

pos = {}
for h, ns in by_height.items():
    ns = sorted(ns)
    for i, n in enumerate(ns):
        pos[n] = (i - (len(ns) - 1) / 2, h)

# --- draw -------------------------------------------------------------------
hasse_edges = list(hasse.edges())
skip_edges  = [(u, v) for u, v in G.edges() if not hasse.has_edge(u, v)]

fig, ax = plt.subplots(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_size=200, node_color='steelblue', ax=ax)
nx.draw_networkx_labels(G, pos, font_size=7, font_color='white', ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=hasse_edges,
                       edge_color='black', arrows=True, arrowsize=10, ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=skip_edges,
                       edge_color='tomato', style='dashed',
                       arrows=True, arrowsize=10, ax=ax)
ax.legend(handles=[mpatches.Patch(color='black',  label='Hasse'),
                   mpatches.Patch(color='tomato', label='DI skip')])
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{file}_sublattices.pdf', bbox_inches='tight')
plt.show()

