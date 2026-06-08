from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx

i = '58'
print(f'### {i} ###')
parser = Parser()
cxt = parser.decode_cxt(f'../../data/{i}.cxt')
lat = ConceptLattice.from_context(cxt)
G = nx.transitive_reduction(lat.to_networkx())

top = 0
bot = len(G.nodes)-1

def longest_path_length(G, source, target):
    try:
        return max(len(p) - 1 for p in nx.all_simple_paths(G, source, target))
    except (nx.NetworkXNoPath, ValueError):
        return 0

max_chain = longest_path_length(G, top, bot)
heights = {}

for node in G.nodes:
    heights[node] = float(longest_path_length(G, node, bot))

for node in G.nodes:
    preds = list(G.predecessors(node))
    succs = list(G.successors(node))
    
    is_doubly_irreducible = (len(preds) == 1 and len(succs) == 1)
    
    if is_doubly_irreducible:
        above = preds[0]
        below = succs[0]
        gap = heights[above] - heights[below]
        heights[node] = heights[below] + gap / 2

for node in G.nodes:
    print(f"Node {node:3d} | height: {heights[node]}")