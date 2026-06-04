import networkx as nx
from fcapy.lattice import ConceptLattice
from data import Parser

# length 6 cycle = cube

# 1. Load Data & Build the Clean Hasse skeleton
file = '30'
cxt = Parser().decode_cxt(f'./data/{file}.cxt')
lattice = ConceptLattice.from_context(cxt)
G_hasse = nx.transitive_reduction(lattice.to_networkx())

# Create your core drawing graph
G_core = G_hasse.copy()

# 3. Switch to Undirected Graph to find visual loops
G_undirected = G_core.to_undirected()

# 4. Extract the Minimal Cycle Basis
# This finds every fundamental geometric "window" or hole in your layout
cycles = nx.cycle_basis(G_undirected)

print(f"=== Drawing Layout Analysis for {file} ===")
print(f"Total Nodes to draw (excluding bounds): {G_core.number_of_nodes()}")
print(f"Total Visual Loops Detected: {len(cycles)}")

# 5. Classify the Loops into Building Blocks
squares = 0
diamonds_or_larger = 0

for i, cycle in enumerate(cycles):
    cycle_len = len(cycle)
    if cycle_len == 4:
        squares += 1
    elif cycle_len > 4:
        diamonds_or_larger += 1
        print(f"\nFound large structural loop #{diamonds_or_larger} (Length {cycle_len}):")
        print(f"Nodes in this block: {cycle}")

print("\n=== Building Block Breakdown ===")
print(f"Independent 4-cycles (Squares/B2 blocks): {squares}")
print(f"Larger complex multi-axis loops: {diamonds_or_larger}")