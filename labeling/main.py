from data.parser import Parser
from fca.lattice import Lattice

from intersection import *
from utils import *
from topology import *
from label import *
from filter import *
from plotting import *
from hybrid import *

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

################################################################################
# Label Candidates
################################################################################
label_config = {
    'general': False,
    'extent':  True,
    'intent':  True
}
label_candidates = {}
label_texts = {}

for node_id in lattice.nodes:
    per_node = []

    if label_config['general']:
        general_txt = wrap_label_text(f'Concept {node_id}', formatter=str)
        label_texts[(node_id, 'general')] = general_txt
        per_node += compute_label_candidates(G, concepts=[node_id], label_text=general_txt, label_type='extent', padding_x_mm=2.0, padding_y_mm=2.0)[node_id]
    
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
        
        per_node += compute_label_candidates(G, [node_id], label_texts[(node_id, label_type)], label_type)[node_id]

    label_candidates[node_id] = per_node

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="general_label_candidates.pdf",
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
    output_path="filtered_outer.pdf",
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
    output_path="filter_unclear.pdf",
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
    output_path="filtered_ink_edges.pdf",
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
    output_path="filtered_neighbor.pdf",
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
label_candidates, _ = hybrid_label_placement(label_candidates, False)

plot_lattice(
    G, cxt, lattice.nodes, coords,
    output_path="hybrid.pdf",
    intersections=intersection_points,
    cycles=bounded_faces,
    areas=areas,
    centers=centers,
    label_candidates=label_candidates,
    label_texts=label_texts,
    show_label_candidates=True,
    colored_label_candidates=True
)