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
from topology import betti_1, extract_faces, normalize_positions
from plot import plot_lattice
from label import measure_ink_mm
from placement import (compute_label_candidates, filter_candidates_by_edges,
                       restrict_outer_node_candidates, filter_candidates_by_nodes,
                       filter_candidates_by_neighbor_direction, hybrid_label_placement,
                       place_overflow_labels)
import matplotlib

# ---------------------------------------------------------------------------
# Load context and coordinates
# ---------------------------------------------------------------------------
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
# Label variant helpers
# ---------------------------------------------------------------------------

MAX_ROW_CHARS = 10   # split into 2 balanced rows if full text exceeds this


def wrap_label_text(raw_items: list, formatter=str) -> str:
    """
    Build a LaTeX string for a label.

    If the full text (all items joined by spaces) is <= MAX_ROW_CHARS,
    return it as a single formatted string.  Otherwise split into exactly
    2 rows at the word boundary closest to the midpoint, making both rows
    as equal in length as possible.  A single word longer than MAX_ROW_CHARS
    is never broken — it gets one row.
    """
    if not raw_items:
        return ''

    # Join all items into one string (items are separate concepts but we treat
    # the whole label as a single text block for splitting purposes).
    full_text = ' '.join(raw_items)

    # No split needed if short enough
    if len(full_text) <= MAX_ROW_CHARS:
        return formatter(full_text)

    # Split into at most 2 rows as evenly as possible.
    # Find the word boundary closest to the midpoint of the full string.
    words = full_text.split()
    if len(words) == 1:
        # Single word longer than limit — can't split, return as-is
        return formatter(full_text)
    mid = len(full_text) / 2
    best_split = 1  # index into words: first row = words[:best_split]
    best_diff = float('inf')
    for i in range(1, len(words)):
        row1 = ' '.join(words[:i])
        diff = abs(len(row1) - mid)
        if diff < best_diff:
            best_diff = diff
            best_split = i

    row1 = formatter(' '.join(words[:best_split]))
    row2 = formatter(' '.join(words[best_split:]))
    return rf'\begin{{tabular}}{{@{{}}c@{{}}}}{row1} \\[-4pt] {row2}\end{{tabular}}'


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
# Compute label placement candidates (extent=below, intent=above)
# ---------------------------------------------------------------------------
label_candidates = {}        # node_id → list of LabelCandidate
label_texts      = {}        # (node_id, label_type) → text string

intent_fmt = lambda x: rf'\textit{{{x}}}'

for node_id in lattice.nodes:
    per_node = []

    # ── Extent (objects) ──────────────────────────────────────────────────
    objects = lattice.lattice.get_concept_new_extent(node_id)
    if objects:
        ext_text = wrap_label_text(sorted(str(o) for o in objects), formatter=str)
        label_texts[(node_id, 'extent')] = ext_text
        per_node += compute_label_candidates(
            G, concepts=[node_id], label_text=ext_text,
            physical_height_mm=PHYSICAL_HEIGHT_MM,
            label_type='extent',
            padding_x_mm=3.0, padding_y_mm=2.0,
            fontsize_pt=candidate_fontsize_pt,
        )[node_id]

    # ── Intent (attributes) ───────────────────────────────────────────────
    attributes = lattice.lattice.get_concept_new_intent(node_id)
    if attributes:
        int_text = wrap_label_text(sorted(str(a) for a in attributes), formatter=intent_fmt)
        label_texts[(node_id, 'intent')] = int_text
        per_node += compute_label_candidates(
            G, concepts=[node_id], label_text=int_text,
            physical_height_mm=PHYSICAL_HEIGHT_MM,
            label_type='intent',
            padding_x_mm=3.0, padding_y_mm=2.0,
            fontsize_pt=candidate_fontsize_pt,
        )[node_id]

    label_candidates[node_id] = per_node

print(f"\nLabel candidates computed for {len(label_candidates)} nodes")

# ---------------------------------------------------------------------------
# Restrict outer nodes to outward-facing candidates
# ---------------------------------------------------------------------------
top_node    = min(lattice.nodes)   # node 0 = top of lattice
bottom_node = max(lattice.nodes)   # node N = bottom of lattice
label_candidates = restrict_outer_node_candidates(
    G, label_candidates, outer_nodes, top_node, bottom_node
)

# ---------------------------------------------------------------------------
# Filter candidates that contain another concept node in their outer bbox
# ---------------------------------------------------------------------------
label_candidates = filter_candidates_by_nodes(G, label_candidates, lattice.nodes)

# ---------------------------------------------------------------------------
# Filter candidates that collide with incident face edges
# ---------------------------------------------------------------------------
label_candidates = filter_candidates_by_edges(
    G, label_candidates, bounded_faces, outer_nodes,
    skip_nodes={top_node, bottom_node}
)

# ---------------------------------------------------------------------------
# Filter by neighbor direction if 2 candidates of same type remain
# ---------------------------------------------------------------------------
label_candidates = filter_candidates_by_neighbor_direction(
    G, label_candidates, lattice.lattice, top_node, bottom_node
)

for node_id, candidates in label_candidates.items():
    print(f"  Node {node_id}: {len(candidates)} candidates remaining: {[c.anchor for c in candidates]}")

# ---------------------------------------------------------------------------
# Resolve remaining ink-box conflicts with the Hybrid algorithm (Wolff §3.2.1)
# ---------------------------------------------------------------------------
chosen_labels, hybrid_log = hybrid_label_placement(label_candidates, verbose=True)

n_extent = sum(1 for lst in chosen_labels.values() for c in lst if c.label_type == 'extent')
n_intent = sum(1 for lst in chosen_labels.values() for c in lst if c.label_type == 'intent')
print(f"\nHybrid placement: {n_extent} extent + {n_intent} intent labels placed")

# ---------------------------------------------------------------------------
# Overflow labels — place unresolved labels outside the diagram
# ---------------------------------------------------------------------------
overflow_labels = place_overflow_labels(
    G,
    label_candidates=label_candidates,
    chosen_labels=chosen_labels,
    label_texts=label_texts,
    physical_height_mm=PHYSICAL_HEIGHT_MM,
    fontsize_pt=candidate_fontsize_pt,
    padding_x_mm=3.0,
    padding_y_mm=2.0,
)
print(f"Overflow labels: {len(overflow_labels)}")

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
    annotations=False,
    intersections=intersection_points,
    show_intersections=False,
    show_face_areas=False,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    show_label_candidates=True,
    label_candidates=label_candidates,
    chosen_labels=chosen_labels,
    label_texts=label_texts,
    fontsize_pt=candidate_fontsize_pt,
    overflow_labels=overflow_labels,
)