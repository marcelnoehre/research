"""
main.py
-------
Entry point: loads a concept lattice, builds its planar graph,
extracts faces, and produces the face-area diagram.
"""
from data.parser import Parser
from fca.lattice import Lattice
from intersection import find_intersections, build_planar_graph
from topology import betti_1, extract_faces, normalize_positions, label_fits
from plot import plot_lattice
from label import measure_ink_mm
import matplotlib

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
    print(f"  Edge {i} {edges_list[i]} x Edge {j} {edges_list[j]} @ {pt.coords[0]}")

G = build_planar_graph(edges_list, coords, intersections)
print(f"β₁ = {betti_1(G)} independent cycles")

# ---------------------------------------------------------------------------
# Normalize coordinates (height = 10 units) and set physical scale
# ---------------------------------------------------------------------------
normalize_positions(G, target_height=10.0)

PHYSICAL_HEIGHT_MM = 100.0   # change to match your actual drawing height

# ---------------------------------------------------------------------------
# Extract faces
# ---------------------------------------------------------------------------
bounded_faces, outer_nodes, areas, centers = extract_faces(G)
print(f"\nOuter face: {len(outer_nodes)} nodes")
print("  ", outer_nodes)
for i, (center, area) in enumerate(zip(centers, areas)):
    print(f"Face {i}: center={center}, area={area:.4f}")

# ---------------------------------------------------------------------------
# Label fit check
# ---------------------------------------------------------------------------
fontsize_pt = matplotlib.rcParams.get('font.size', 10.0)
label_text  = r"Hello, \LaTeX!"   # replace with your actual label text
ink_w_mm, ink_h_mm = measure_ink_mm(label_text, fontsize_pt)
print(f"\nLabel ink size: {ink_w_mm:.2f} mm x {ink_h_mm:.2f} mm")

for i, face in enumerate(bounded_faces):
    fits = label_fits(G, face, ink_w_mm, ink_h_mm, PHYSICAL_HEIGHT_MM)
    print(f"Face {i}: label {'fits' if fits else 'does not fit'}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
# Transform raw intersection points into normalized coordinates
_s   = G.graph['norm_scale']
_mx  = G.graph['norm_min_x']
_my  = G.graph['norm_min_y']
intersection_points = [
    ((pt.coords[0][0] - _mx) * _s, (pt.coords[0][1] - _my) * _s)
    for _, _, pt in intersections
]
plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="size_of_faces.pdf",
    annotations=True,
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
)
