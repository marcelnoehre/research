from data import Parser
from fcapy.lattice import ConceptLattice
import networkx as nx
from grandiso import find_motifs
import networkx as nx

# data
file = 'forum_romanum'
cxt = Parser().decode_cxt(f'./data/{file}.cxt')
lattice = ConceptLattice.from_context(cxt)
G = lattice.to_networkx()

# Build the B3 motif as a DiGraph (Hasse diagram of Boolean lattice on 3 atoms)
B3 = nx.DiGraph()
B3.add_edges_from([
    ('bot', 'a'), ('bot', 'b'), ('bot', 'c'),
    ('a', 'ab'), ('a', 'ac'),
    ('b', 'ab'), ('b', 'bc'),
    ('c', 'ac'), ('c', 'bc'),
    ('ab', 'top'), ('ac', 'top'), ('bc', 'top'),
])

tr = nx.transitive_reduction(G)
tr_smoothed = tr.copy()

# Find all nodes that act purely as a "subdivision" (1 in-degree, 1 out-degree)
nodes_to_remove = [
    node for node in tr_smoothed.nodes() 
    if tr_smoothed.in_degree(node) == 1 and tr_smoothed.out_degree(node) == 1
]

for node in nodes_to_remove:
    # Get the single predecessor and single successor
    pred = list(tr_smoothed.predecessors(node))[0]
    succ = list(tr_smoothed.successors(node))[0]
    
    # Connect the two outer nodes directly
    tr_smoothed.add_edge(pred, succ)
    
    # Remove the middle node
    tr_smoothed.remove_node(node)

tr = tr_smoothed

# Find all B3 sublattices
matches = find_motifs(B3, tr)

# Just deduplicate for atom/coatom permutations
seen = set()
unique_B3s = []
for m in matches:
    key = frozenset(m.values())
    if key not in seen:
        seen.add(key)
        unique_B3s.append(m)

print(f"Found {len(unique_B3s)} unique B3 sublattice(s).")
