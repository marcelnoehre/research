import networkx as nx

from data.parser import Parser
from fca.lattice import Lattice

from intersection import find_intersections, build_planar_graph
from utils import normalize_positions, normalize_intersections
from plotting import plot_lattice

################################################################################ 
# Data
################################################################################
FILE = 'living_beings_and_water_original'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{FILE}.cxt')
print('Formal Context')
print(cxt.print_data())

with open(f'../data/{FILE}.pos', 'r') as f:
    coords = {
        c: tuple(map(float, line.split()[:2]))
        for c, line in enumerate(f) if line.strip()
    }

lattice = Lattice(cxt)
edges_list = list(lattice.cover_relations())

################################################################################
# Detect intersections and build planar graph
################################################################################
intersections = find_intersections(edges_list, coords)
print(f'{len(intersections)} intersections found')
for i, j, pt in intersections:
    print(f'  Edge {i} {edges_list[i]} x Edge {j} {edges_list[j]} @ {pt.coords[0]}')

G = build_planar_graph(edges_list, coords, intersections)

################################################################################
# Normalize coordinates and set physical scale
################################################################################
normalize_positions(G, target_height=10.0)
intersection_points = normalize_intersections(G, intersections)

# First Betti number (number of independent cycles): β₁ = E - V + C
betti_1 = G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)
print(f'β₁ = {betti_1} independent cycles')

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="lattice.pdf",
    intersections=intersection_points,
    show_vertex_ids=True,
    show_intersections=True
)