from data.parser import Parser
from fca.lattice import Lattice

from intersection import *
from utils import *
from topology import *
from label import *
from filter import *
from hybrid import *
from plotting import *
from overflow_bounded import *
from overflow_unbounded import *
from post_processing import *

################################################################################ 
# Data
################################################################################
FILE = 'convex-ordinal'
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
    output_path="figs/input.pdf",
    intersections=intersection_points
)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/intersections.pdf",
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
    output_path="figs/faces.pdf",
    intersections=intersection_points,
    show_intersections=True,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    show_face_areas=True
)

################################################################################
# Label Candidates
################################################################################
label_config = {
    'general': True,
    'extent':  False,
    'intent':  False
}
label_candidates = {}
label_texts = {}

for node_id in lattice.nodes:
    per_node = []

    if label_config['general']:
        general_txt = wrap_label_text(f'Concept {node_id}', formatter=str)
        label_texts[(node_id, 'general')] = general_txt

    if label_config['extent']:
        objects = sorted(str(g) for g in lattice.lattice.get_concept_new_extent(node_id))
        extent_txt = wrap_label_text(', '.join(objects), formatter=str)
        if extent_txt:
            label_texts[(node_id, 'extent')] = extent_txt

    if label_config['intent']:
        attributes = sorted(str(m) for m in lattice.lattice.get_concept_new_intent(node_id))
        intent_txt = wrap_label_text(', '.join(attributes), formatter=str)
        if intent_txt:
            label_texts[(node_id, 'intent')] = intent_txt

    for label_type, is_active in label_config.items():
        if not is_active or not label_texts.get((node_id, label_type)):
            continue
        per_node += compute_label_candidates(G, [node_id], label_texts[(node_id, label_type)],
                        label_type)[node_id]

    label_candidates[node_id] = per_node

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/general_label_candidates.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)

################################################################################
# Restrict outer nodes to outward-facing candidates
################################################################################
top_node = min(lattice.nodes)
bottom_node = max(lattice.nodes)
label_candidates = restrict_outer_node_candidates(G, label_candidates, outer_nodes, top_node, bottom_node)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/filtered_outer.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)

################################################################################
# Filter candidates that contain another concept node in their outer bbox
################################################################################
label_candidates = filter_candidates_by_nodes(G, label_candidates, lattice.nodes)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/filter_unclear.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)

################################################################################
# Filter candidates that collide with incident face edges
################################################################################
label_candidates = filter_candidates_by_edges(G, label_candidates, bounded_faces, outer_nodes)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/filtered_ink_edges.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)

################################################################################
# Filter by neighbor direction if 2 candidates of same type remain
################################################################################
label_candidates = filter_candidates_by_neighbor_direction(G, label_candidates, lattice.lattice)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/filtered_neighbor.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)

################################################################################
# Hybrid Approach
################################################################################
label_candidates, _ = hybrid_label_placement(label_candidates, True)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/hybrid.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)

################################################################################
# Filter to optimize gaps
################################################################################
label_candidates = filter_optimize_gaps(G, label_candidates, bounded_faces, outer_nodes)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/optimize_gaps.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)

################################################################################
# Overflow Candidates
################################################################################
overflow_candidates = {}
if label_config['general']:
    for node_id in lattice.nodes:
        if not label_candidates[node_id]:
            overflow_candidates[node_id] = compute_overflow_label(G, node_id, label_texts.get((node_id, 'general'), ''), label_type='general')

skipping_inner = len(overflow_candidates) >= len(outer_nodes)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/overflow_candidates.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True,
    overflow_labels=overflow_candidates,
    show_overflow_labels=True
)

################################################################################
# Inner Overflow Labels
################################################################################
if not skipping_inner:
    overflow_candidates = inner_overflow_labels(G, label_candidates, overflow_candidates, bounded_faces, centers)

    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/inner_overflow_candidates.pdf",
        intersections=intersection_points,
        cycles=bounded_faces,
        areas=areas,
        centers=centers,
        label_candidates=label_candidates,
        label_texts=label_texts,
        show_label_candidates=True,
        colored_label_candidates=True,
        overflow_labels=overflow_candidates,
        show_overflow_labels=True
    )

################################################################################
# Outer Overflow Labels
################################################################################
unbounded_overflow_labels = [
    ol.node_id
    for ol in overflow_candidates.values()
    if ol.anchor == 'overflow'
]
overflow_candidates = outer_overflow_labels(G, label_candidates, overflow_candidates, outer_nodes)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/outer_overflow_candidates.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True,
    overflow_labels=overflow_candidates,
    show_overflow_labels=True
)

################################################################################
# Post-Processing - anchor adjustment
################################################################################
overflow_candidates = adjust_anchors(G, label_candidates, overflow_candidates, unbounded_overflow_labels, outer_nodes)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/adjust_anchors.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True,
    overflow_labels=overflow_candidates,
    show_overflow_labels=True
)

################################################################################
# Post-Processing - binder adjustment
################################################################################
overflow_candidates = adjust_binders(G, label_candidates, overflow_candidates, unbounded_overflow_labels, outer_nodes)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="figs/adjust_binders.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True,
    overflow_labels=overflow_candidates,
    show_overflow_labels=True
)

################################################################################
# Final Drawing
################################################################################
plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path=f"figs/{FILE}.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    overflow_labels=overflow_candidates,
    show_overflow_labels=True
)
