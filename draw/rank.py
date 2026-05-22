from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx

file = '43'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{file}.cxt')
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

def get_predecessors_min_height(node, heights):
    preds = list(G.predecessors(node))
    return min(heights[p] for p in preds) if preds else 0

# First pass: assign integer heights from from_bot
heights = {}
for node in G.nodes:
    depth_from_top = longest_path_length(G, top, node)
    depth_from_bot = longest_path_length(G, node, bot)
    heights[node] = float(depth_from_bot)

for node in G.nodes:
    preds = list(G.predecessors(node))   # nodes above
    succs = list(G.successors(node))     # nodes below
    if not preds or not succs:
        continue
    
    max_succ_height = max(heights[s] for s in succs)
    min_pred_height = min(heights[p] for p in preds)
    # subdividing: sits between two levels that are 2 apart in from_bot
    if (heights[node] - max_succ_height == 0) and (min_pred_height - heights[node] == 2):
        heights[node] += 0.5

for node in G.nodes:
    print(f"Node {node:3d} | from_bot: {longest_path_length(G, node, bot):2d} | height: {heights[node]}")