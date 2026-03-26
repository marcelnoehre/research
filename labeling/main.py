import networkx as nx

from data.parser import Parser
from fca.lattice import Lattice

from intersection import find_intersections, build_planar_graph
from utils import normalize_positions, normalize_intersections
from topology import betti_1, extract_faces
from plotting import plot_lattice

################################################################################ 
# Data
################################################################################
FILE = 'living_beings_and_water_original'
parser = Parser()
cxt = parser.decode_cxt(f'../data/{FILE}.cxt')
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
scale = normalize_positions(G, target_height=10.0)
intersection_points = normalize_intersections(G, intersections)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="intersections.pdf",
    intersections=intersection_points,
    show_intersections=True
)

################################################################################
# Topology
################################################################################
print(f'\n β₁ = {betti_1(G)} (independent cycles)')

bounded_faces, outer_nodes, areas, centers = extract_faces(G, scale)
print(f'\nOuter face: {len(outer_nodes)} nodes')
print('  ', outer_nodes, '\n')
for i, (center, area) in enumerate(zip(centers, areas)):
    print(f'Face {i}: center=({center[0]:.2f}, {center[1]:.2f}), area={area:.2f}')

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="faces.pdf",
    intersections=intersection_points,
    show_intersections=True,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    show_face_areas=True
)