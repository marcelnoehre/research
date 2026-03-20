"""
main.py
-------
Entry point: loads a concept lattice, builds its planar graph,
extracts faces, computes label placement candidates, and produces
the face-area diagram with all candidate positions shown.
"""
from data.parser import Parser
from fca.lattice import Lattice
from intersection import find_intersections, build_planar_graph
from topology import betti_1, extract_faces, normalize_positions, label_fits
from plot import plot_lattice
from label import measure_ink_mm
from placement import compute_label_candidates, filter_candidates_by_edges
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
# Label helpers
# ---------------------------------------------------------------------------
def node_label(node_id: int) -> str:
    return rf"$c_{{{node_id}}}$"


def latex_upscale_factor(latex_text: str, plain_text: str, fontsize_pt: float) -> float:
    """
    Compute how much to scale up fontsize_pt so that the LaTeX formula
    renders at the same ink height as the plain text would at fontsize_pt.
    Only applies when latex_text is actually a LaTeX formula (wrapped in $...$).
    """
    if not (latex_text.startswith('$') and latex_text.endswith('$')):
        return 1.0
    plain_h = measure_ink_mm(plain_text, fontsize_pt)[1]
    latex_h = measure_ink_mm(latex_text, fontsize_pt)[1]
    if latex_h == 0:
        return 1.0
    return plain_h / latex_h


# ---------------------------------------------------------------------------
# Label setup
# ---------------------------------------------------------------------------
fontsize_pt = matplotlib.rcParams.get('font.size', 10.0)
# LaTeX subscripts render smaller than plain text — measure the ratio and
# scale up so the formula appears at the intended size.
_sample_latex = node_label(0)
_sample_plain = "C0"
_scale = latex_upscale_factor(_sample_latex, _sample_plain, fontsize_pt)
candidate_fontsize_pt = fontsize_pt * _scale
print(f"LaTeX upscale factor: {_scale:.3f} → candidate fontsize: {candidate_fontsize_pt:.1f}pt")

# ---------------------------------------------------------------------------
# Label fit check per face (use the largest label as a conservative bound)
# ---------------------------------------------------------------------------
max_label = max((node_label(n) for n in lattice.nodes), key=lambda t: sum(measure_ink_mm(t, candidate_fontsize_pt)))
ink_w_mm, ink_h_mm = measure_ink_mm(max_label, candidate_fontsize_pt)
print(f"\nLargest label ink size: {ink_w_mm:.2f} mm x {ink_h_mm:.2f} mm")

for i, face in enumerate(bounded_faces):
    fits = label_fits(G, face, ink_w_mm, ink_h_mm, PHYSICAL_HEIGHT_MM)
    print(f"Face {i}: label {'fits' if fits else 'does not fit'}")

# ---------------------------------------------------------------------------
# Compute label placement candidates (4 per node, using each node's own label)
# ---------------------------------------------------------------------------
label_candidates = {}
for node_id in lattice.nodes:
    text = node_label(node_id)
    label_candidates.update(compute_label_candidates(
        G,
        concepts=[node_id],
        label_text=text,
        physical_height_mm=PHYSICAL_HEIGHT_MM,
        padding_x_mm=3.0,
        padding_y_mm=2.0,
        fontsize_pt=candidate_fontsize_pt,
    ))
print(f"\nLabel candidates computed for {len(label_candidates)} nodes")

# ---------------------------------------------------------------------------
# Filter candidates that collide with incident face edges
# ---------------------------------------------------------------------------
label_candidates = filter_candidates_by_edges(
    G, label_candidates, bounded_faces, outer_nodes
)
for node_id, candidates in label_candidates.items():
    print(f"  Node {node_id}: {len(candidates)} candidates remaining")

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

label_texts = {node_id: node_label(node_id) for node_id in lattice.nodes}

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="size_of_faces.pdf",
    annotations=False,
    intersections=intersection_points,
    show_intersections=False,
    show_face_areas=False,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    show_label_candidates=True,
    label_candidates=label_candidates,
    label_texts=label_texts,
    fontsize_pt=candidate_fontsize_pt,
)