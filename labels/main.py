"""
main.py
-------
Entry point: loads a concept lattice, builds its planar graph,
extracts faces, and produces the face-area diagram.
"""

from data.parser import Parser
from fca.lattice import Lattice
from intersection import find_intersections, build_planar_graph
from topology import betti_1, extract_faces
from plot import plot_lattice

# ---------------------------------------------------------------------------
# Load context and coordinates
# ---------------------------------------------------------------------------

FILE = 'living_beings_and_water'

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

# ---------------------------------------------------------------------------
# Detect intersections and build planar graph
# ---------------------------------------------------------------------------

intersections = find_intersections(edges_list, coords)
print(f"{len(intersections)} intersections found")
for i, j, pt in intersections:
    print(f"  Edge {i} {edges_list[i]} × Edge {j} {edges_list[j]} @ {pt.coords[0]}")

G = build_planar_graph(edges_list, coords, intersections)
print(f"β₁ = {betti_1(G)} independent cycles")

# ---------------------------------------------------------------------------
# Extract faces
# ---------------------------------------------------------------------------

bounded_faces, outer_nodes, areas, centers = extract_faces(G)

print(f"\nOuter face: {len(outer_nodes)} nodes")
print("  ", outer_nodes)

for i, (center, area) in enumerate(zip(centers, areas)):
    print(f"Face {i}: center={center}, area={area:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

intersection_points = [pt.coords[0] for _, _, pt in intersections]

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="size_of_faces.pdf",
    annotations=True,
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
)
