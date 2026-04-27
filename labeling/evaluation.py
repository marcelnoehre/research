from data.parser import Parser
from fca.lattice import Lattice

from config import Config
from intersection import *
from overflow_bounded import _label_bbox_polygon, _label_wh
from post_processing import _get_anchor_pt
from utils import *
from topology import *
from label import *
from filter import *
from hybrid import *
from plotting import *
from overflow_bounded import *
from overflow_unbounded import *
from post_processing import *
from forces import *

import csv
import time
import statistics

################################################################################ 
# Data
################################################################################
VISUALIZE = False
cfg = Config()
cfg.file = 'car_original'
output_file = "results.csv"
headers = ['top_k', 'grid_step', 'max_dist_to_drawing', 'hungarian_iterations', 'duration_cost', 'final_cost', 'duration_force', 'iterations_force', 'avg_dist_drawing', 'avg_min_dist_ol_obstacle', 'avg_min_dist_binder_obstacle']
parser = Parser()
cxt = parser.decode_cxt(f'../data/{cfg.file}.cxt')

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
#print(cxt.print_data())

with open(f'../data/{cfg.file}.pos', 'r') as f:
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
#print(f'{len(intersections)} intersections found')
# for i, j, pt in intersections:
    #print(f'  Edge {i} {edges_list[i]} x Edge {j} {edges_list[j]} @ {pt.coords[0]}')

G = build_planar_graph(edges_list, coords, intersections)

################################################################################
# Normalize coordinates and set physical scale
################################################################################
scale = normalize_positions(G, target_height=10.0)
intersection_points = normalize_intersections(G, intersections)

if VISUALIZE:
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
#print(f'\n β₁ = {betti_1(G)} (independent cycles)')

bounded_faces, outer_nodes, areas, centers = extract_faces(G, scale)
#print(f'\nOuter face: {len(outer_nodes)} nodes')
#print('  ', outer_nodes, '\n')
# for i, (center, area) in enumerate(zip(centers, areas)):
    #print(f'Face {i}: center=({center[0]:.2f}, {center[1]:.2f}), area={area:.2f}')

if VISUALIZE:
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
types = ['general', 'extent', 'intent']
label_config = {
    'general': True,
    'extent':  False,
    'intent':  False
}
label_candidates = {}
label_texts = {}
label_scale = {}

_, desired_height, _ = measure_ink_mm('Calibrate', FONT_SIZE, -1.0)

for node_id in lattice.nodes:
    per_node = []

    if label_config['general']:
        general_txt = wrap_label_text(f'Node {node_id}', 'general', formatter=str)
        label_texts[(node_id, 'general')] = general_txt

    if label_config['extent']:
        objects = sorted(str(g) for g in lattice.lattice.get_concept_new_extent(node_id))
        extent_txt = wrap_label_text(', '.join(objects), 'extent', formatter=str)
        if extent_txt:
            label_texts[(node_id, 'extent')] = extent_txt

    if label_config['intent']:
        attributes = sorted(str(m) for m in lattice.lattice.get_concept_new_intent(node_id))
        intent_txt = wrap_label_text(', '.join(attributes), 'intent', formatter=str)
        if intent_txt:
            label_texts[(node_id, 'intent')] = intent_txt

    for label_type, is_active in label_config.items():
        if not is_active or not label_texts.get((node_id, label_type)):
            continue
        scale, scaled_cands = compute_label_candidates(G, [node_id], label_texts[(node_id, label_type)],
                        label_type, desired_height)
        
        per_node += scaled_cands[node_id]
        label_scale[(node_id, label_type)] = scale

    label_candidates[node_id] = per_node

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/general_label_candidates.pdf",
        label_scale=label_scale,
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

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/filtered_outer.pdf",
        label_scale=label_scale,
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

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/filter_unclear.pdf",
        label_scale=label_scale,
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

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/filtered_ink_edges.pdf",
        label_scale=label_scale,
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

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/filtered_neighbor.pdf",
        label_scale=label_scale,
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

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/hybrid.pdf",
        label_scale=label_scale,
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
label_candidates = filter_optimal_space(G, label_candidates, bounded_faces, outer_nodes)

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/optimize_space.pdf",
        label_scale=label_scale,
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
for node_id in lattice.nodes:
    if not label_candidates[node_id]:
        for type in types:
            #print(node_id, type, label_texts.get((node_id, type), ''))
            if label_texts.get((node_id, type), ''):
                overflow_candidates[len(overflow_candidates)] = compute_overflow_label(G, node_id, label_texts.get((node_id, type), ''), label_type=type, desired_height=desired_height)

skipping_inner = len(overflow_candidates) >= len(outer_nodes)

if VISUALIZE:
    plot_lattice(
        G, cxt, lattice.nodes, coords,
        output_path="figs/overflow_candidates.pdf",
        label_scale=label_scale,
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

    if VISUALIZE:
        plot_lattice(
            G, cxt, lattice.nodes, coords,
            output_path="figs/inner_overflow_candidates.pdf",
            label_scale=label_scale,
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
    lid
    for lid, ol in overflow_candidates.items()
    if ol.anchor == 'overflow'
]

TOP_Ks = list(range(50, 1050, 50))
GRID_STEPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_MAX_DISTS = [(1.0, 2.0), (1.0, 3.0), (1.0, 4.0), (1.0, 5.0), (1.0, 7.5), (1.0, 10.0)]
ITERATIVE_HUNGARIAN_MAX_ITERATIONS = [1, 2, 5, 10, 25, 40, 55, 70, 85, 100, 125, 150, 175, 200]
for tk in TOP_Ks:
    cfg.OUTER_CANDIDATE_POOL = tk
    for gs in GRID_STEPS:
        cfg.GRID_STEP = gs
        for min_dist, max_dist in MIN_MAX_DISTS:
            for iter_amount in ITERATIVE_HUNGARIAN_MAX_ITERATIONS:
                cfg.MIN_LABEL_DIST = min_dist
                cfg.MAX_LABEL_DIST = max_dist
                cfg.ITERATIVE_HUNGARIAN_MAX_ITERS = iter_amount

                start_time = time.perf_counter()
                all_overflow_candidates, overflow_candidates, assignment = outer_overflow_labels(G, label_candidates, overflow_candidates, outer_nodes, cfg)
                end_time = time.perf_counter()
                duration_cost = end_time - start_time

                final_cost = 0.0
                for _, chosen in assignment.items():
                    final_cost += chosen['cost']

                # plot all unbounded overflow candidates
                if VISUALIZE:
                    plot_lattice(
                        G, cxt, lattice.nodes, coords,
                        output_path="figs/all_outer_overflow_candidates.pdf",
                        label_scale=label_scale,
                        intersections=intersection_points,
                        cycles=bounded_faces,
                        areas=areas,
                        centers=centers,
                        label_candidates=label_candidates,
                        label_texts=label_texts,
                        show_label_candidates=True,
                        colored_label_candidates=True,
                        overflow_labels=all_overflow_candidates,
                        show_overflow_labels=True
                    )

                    plot_lattice(
                        G, cxt, lattice.nodes, coords,
                        output_path="figs/outer_overflow_candidates.pdf",
                        label_scale=label_scale,
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
                ################################################################################
                ################################################################################
                overflow_candidates = adjust_anchors(G, label_candidates, overflow_candidates, unbounded_overflow_labels, outer_nodes)

                if VISUALIZE:
                    plot_lattice(
                        G, cxt, lattice.nodes, coords,
                        output_path="figs/adjust_anchors.pdf",
                        label_scale=label_scale,
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
                ################################################################################
                ################################################################################


                ################################################################################
                # Force Based Refinement
                ################################################################################
                
                start_time = time.perf_counter()
                overflow_candidates, iters_converged = optimize_overflow_labels(G, label_candidates, overflow_candidates, unbounded_overflow_labels, outer_nodes)
                end_time = time.perf_counter()
                duration_force = end_time - start_time

                outer_polygon = Polygon([G.nodes[n]['pos'] for n in outer_nodes])
                ink_polys = [
                    Polygon(cand.inner_bbox_corners)
                    for cands in label_candidates.values()
                    for cand in cands
                ]
                outer_circles = []
                for onid in outer_nodes:
                    outer_pos = G.nodes[onid]['pos']
                    circle_radius = 0.15
                    outer_circles.append(Point(outer_pos).buffer(circle_radius))

                # avg_dist_drawing
                # avg_min_dist_ol_obstacle
                dists_drawing = []
                dists_obstacles = []
                for lid, ol in overflow_candidates.items():
                    _size = _label_wh(ol)
                    ol_poly = _label_bbox_polygon(*ol.center, *_size)
                    dists_drawing.append(outer_polygon.distance(ol_poly))

                    # min distance to obstacles
                    _dists = []
                    for obstacle_poly in ink_polys + outer_circles:
                        _dists.append(obstacle_poly.distance(ol_poly))
                    dists_obstacles.append(min(_dists))
                
                # avg_min_dist_binder_obstacle
                dists_binder = []
                for lid, ol in overflow_candidates.items():
                    anchor_pt = _get_anchor_pt(ol)
                    lid_binder = LineString([anchor_pt, G.nodes[ol.node_id]['pos']])
                    _dists = []
                    for obstacle_poly in ink_polys + outer_circles:
                        _dists.append(obstacle_poly.distance(lid_binder))
                    dists_binder.append(min(_dists))

                # top_k, grid_step, max_dist_to_drawing, hungarian_iterations, duration_cost, final_cost,
                # duration_force, iterations_force, avg_dist_drawing, avg_min_dist_ol_obstacle, avg_min_dist_binder_obstacle
                row_to_save = [
                    cfg.OUTER_CANDIDATE_POOL, 
                    cfg.GRID_STEP,
                    cfg.MAX_LABEL_DIST,
                    cfg.ITERATIVE_HUNGARIAN_MAX_ITERS,
                    duration_cost,
                    final_cost,
                    duration_force,
                    iters_converged,
                    statistics.mean(dists_drawing),
                    statistics.mean(dists_obstacles),
                    statistics.mean(dists_binder)
                ]

                with open(output_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row_to_save)


                if VISUALIZE:
                    plot_lattice(
                        G, cxt, lattice.nodes, coords,
                        output_path="figs/force_refinement.pdf",
                        label_scale=label_scale,
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
                if VISUALIZE:
                    plot_lattice(
                        G, cxt, lattice.nodes, coords,
                        output_path=f"figs/{cfg.file}.pdf",
                        label_scale=label_scale,
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
