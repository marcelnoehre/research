import math
import numpy as np
import networkx as nx

from shapely.geometry import LineString, Point, Polygon
from shapely import unary_union
from typing import Dict, List, Optional, Tuple

from label import LabelCandidate, OverflowLabel
from overflow_bounded import (
    _label_wh,
    _label_wh_expanded,
    _label_bbox_polygon,
    _anchor_points,
    update_overflow_label_position,
    binding_line_valid,
)

OUTER_STEP = 0.5
OUTER_MAX_STEPS = 500
WEDGE_RADIUS = 1e4
OUTER_CANDIDATE_POOL = 12

def _ink_wh(label: OverflowLabel) -> Tuple[float, float]:
    '''
    Ink bbox dimensions.
    '''
    bl, br, _, tl = label.inner_bbox_corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])

def _angle_from_centroid(centroid: Tuple[float, float], point: Tuple[float, float]) -> float:
    '''
    angle from the centroid of the drawing to a node
    [-pi, pi]
    '''
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


def _angular_gap_between(a1: float, a2: float) -> float:
    '''
    Measures the angular gap between two outer nodes
    [0, 2pi]
    '''
    return (a2 - a1) % (2 * math.pi)


def _sort_by_node_angle(
    assigned: List[int],
    gap: Dict,
    centroid: Tuple[float, float],
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
) -> List[int]:
    '''
    Sort assigned label ids by their node's angle from centroid within the gap,
    so slot order matches node order and binding lines don't cross each other.
    '''
    a_left = gap['a_left']

    def node_angle_in_gap(node_id):
        node_pos = G.nodes[overflow_candidates[node_id].node_id]['pos']
        angle = _angle_from_centroid(centroid, node_pos)
        return _angular_gap_between(a_left, angle)

    return sorted(assigned, key=node_angle_in_gap)

def _wedge_polygon(
    centroid: Tuple[float, float],
    a_left: float,
    a_right: float,
    outer_polygon: Polygon,
    radius: float = WEDGE_RADIUS,
) -> Polygon:
    '''
    Pie-slice from centroid outward between node angles a_left and a_right (CCW), minus the outer polygon.
    '''
    cx, cy = centroid
    arc_steps = max(16, int(math.degrees(_angular_gap_between(a_left, a_right))))
    arc = []
    for i in range(arc_steps + 1):
        t = i / arc_steps
        angle = a_left + _angular_gap_between(a_left, a_right) * t
        arc.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    wedge = Polygon([(cx, cy)] + arc + [(cx, cy)])
    # pie slice without bounded space
    return wedge.difference(outer_polygon)

def _calculate_cost(
    dist: float, 
    angle_offset: float, 
    binder_length: float, 
    is_cluttered: bool
) -> float:
    '''
    Lower is better. Adjust weights to tune the "feel" of the placement.
    '''
    W_DIST = 1.5      # Penalize pushing labels too far out
    W_ANGLE = 50.0    # Penalize drifting away from the intended slot
    W_BINDER = 2.0    # Penalize long lines
    W_CLUTTER = 100.0 # Heavy penalty for tight spacing

    cost = (dist * W_DIST) + (abs(angle_offset) * W_ANGLE) + (binder_length * W_BINDER)
    
    if is_cluttered:
        cost += W_CLUTTER
        
    return cost

def _place_in_fan(
    central_angle: float,
    gap_size: float,
    centroid: Tuple[float, float],
    outer_polygon: Polygon,
    placed_union: Polygon,
    w: float, 
    h: float,
    node_pos: Tuple[float, float],
    own_node_id: int,
    own_label_id: int,
    G: nx.Graph,
    placed: List[dict],
    placed_binders: List[LineString],
    overflow_candidates: Dict[int, OverflowLabel],
    label_candidates: Dict[int, List[LabelCandidate]]
) -> Optional[Tuple[float, float, str, Tuple]]:
    '''
    '''
    cx, cy = centroid
    node_pt = np.array(node_pos)
    label = overflow_candidates[own_label_id]
    ew, eh = _label_wh_expanded(label)
    tw, th = _label_wh(label)
    iw, ih = _ink_wh(label)

    # already placed outer overflow bboxes
    placed_bboxes = [
        _label_bbox_polygon(*rec['position'], *_label_wh_expanded(overflow_candidates[rec['label_id']]))
        for rec in placed if rec['label_id'] != own_label_id
    ]

    num_rays = 15 
    spread = min(math.radians(40), gap_size * 0.9)
    offsets = np.linspace(-spread/2, spread/2, num_rays)
    
    scored_candidates = []

    for off in offsets:
        angle = central_angle + off
        dx, dy = math.cos(angle), math.sin(angle)
        
        # exit distance for this specific ray
        ray_line = LineString([(cx, cy), (cx + WEDGE_RADIUS * dx, cy + WEDGE_RADIUS * dy)])
        intersection = outer_polygon.exterior.intersection(ray_line)
        
        # ray never touches outer cycle
        if intersection.is_empty:
            exit_dist = 0.0
        # hits outer cycle multiple times - final exit point
        elif hasattr(intersection, 'geoms'):
            exit_dist = max(math.hypot(pt.x - cx, pt.y - cy) for pt in intersection.geoms)
        # hits outer cycle at exactly one point
        else:
            exit_dist = math.hypot(intersection.x - cx, intersection.y - cy)

        # start slightly inside to find true minimum
        start_dist = max(0.0, exit_dist - OUTER_STEP * 2)

        for step in range(OUTER_MAX_STEPS):
            dist = start_dist + OUTER_STEP * step
            
            if scored_candidates and dist > min(c[0] for c in scored_candidates) + 50:
                break

            origin_x, origin_y = cx + dx * dist, cy + dy * dist
            expanded_bbox = _label_bbox_polygon(origin_x, origin_y, ew, eh)
            tight_bbox = _label_bbox_polygon(origin_x, origin_y, tw, th) 

            # overlaps with bounded space
            if outer_polygon.intersects(expanded_bbox): continue
            # overlaps with placed label candidate
            if placed_union.intersects(expanded_bbox): continue
            # overlaps with placed outer overflow candidate
            if any(tight_bbox.intersects(pb) for pb in placed_bboxes): continue
            # overlaps binder
            if any(tight_bbox.intersects(binder) for binder in placed_binders): continue
            # overlaps with a node of the drawing
            if any(
                expanded_bbox.contains(Point(data['pos']))
                for node_id, data in G.nodes(data=True)
                if isinstance(node_id, int) and node_id != own_node_id
            ):
                continue

            # anchor selection
            label_center = np.array([origin_x, origin_y])
            vector_to_node = node_pt - label_center
            dist_to_node = np.linalg.norm(vector_to_node)
            unit_to_node = vector_to_node / dist_to_node if dist_to_node > 0 else vector_to_node

            own_ink_bbox = _label_bbox_polygon(origin_x, origin_y, iw, ih)
            anchors = _anchor_points(origin_x, origin_y, tw, th)

            # Score anchors by alignment (dot product)
            # We want the anchor that points MOST towards the node
            scored_anchors = []
            for name, pt in anchors.items():
                vector_to_anchor = np.array(pt) - label_center
                a_norm = np.linalg.norm(vector_to_anchor)
                alignment = np.dot(unit_to_node, vector_to_anchor / a_norm) if a_norm > 0 else -1.0
                scored_anchors.append((alignment, name, pt))
            
            # Sort by alignment descending (best alignment first)
            scored_anchors.sort(key=lambda x: x[0], reverse=True)

            for alignment, name, pt in scored_anchors:
                line = LineString([pt, node_pos])
                
                # line crosses ink of own label
                if own_ink_bbox.intersects(line): continue
                # intersects with binder
                if any(line.intersects(binder) for binder in placed_binders): continue

                if binding_line_valid(line, own_node_id, own_label_id, G, placed, overflow_candidates, label_candidates):
                    alignment_penalty = (1.0 - alignment) * 10 
                    cost = _calculate_cost(dist, off, line.length, False) + alignment_penalty
                    
                    scored_candidates.append((cost, (origin_x, origin_y, name, pt)))
                    break 
            
            if any(c[1][0] == origin_x and c[1][1] == origin_y for c in scored_candidates):
                break 

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda x: x[0])
    return scored_candidates[0][1]

def outer_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    outer_nodes: List[int],
) -> Dict[int, OverflowLabel]:
    '''
    '''
    outer_positions = [G.nodes[n]["pos"] for n in outer_nodes]
    # bounded space
    outer_polygon = Polygon(outer_positions)
    # centroid of drawing
    cx, cy = outer_polygon.centroid.x, outer_polygon.centroid.y
    centroid = (cx, cy)

    # space blocked by label candidates
    placed_union = unary_union([
        Polygon(cand.expanded_bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]) if label_candidates else Polygon()

    # angles from centroid to each outer node
    node_angles = [
        (_angle_from_centroid(centroid, G.nodes[n]['pos']), n)
        for n in outer_nodes
    ]
    node_angles.sort(key=lambda x: x[0])

    n_outer = len(node_angles)

    gaps = []
    for i in range(n_outer):
        a_left, node_left = node_angles[i]
        a_right, node_right = node_angles[(i + 1) % n_outer]
        gap_size = _angular_gap_between(a_left, a_right)
        wedge = _wedge_polygon(centroid, a_left, a_right, outer_polygon)
        gaps.append({
            "a_left":     a_left,
            "a_right":    a_right,
            "gap_size":   gap_size,
            "node_left":  node_left,
            "node_right": node_right,
            "wedge":      wedge,
            "assigned":   [],
        })

    # Assign unplaced overflow labels to their natural gap
    unplaced = {
        nid: ol for nid, ol in overflow_candidates.items()
        if ol.anchor == 'overflow'
    }

    for node_id, overflow_label in unplaced.items():
        lx, ly = overflow_label.center
        outward_angle = _angle_from_centroid(centroid, (lx, ly))

        best_gap = None
        best_size = -1.0
        for gap in gaps:
            into = _angular_gap_between(gap["a_left"], outward_angle)
            if into <= gap["gap_size"] and gap["gap_size"] > best_size:
                best_size = gap["gap_size"]
                best_gap = gap

        if best_gap is None:
            best_gap = max(gaps, key=lambda g: g["gap_size"])

        best_gap["assigned"].append(node_id)

    # Place labels gap by gap
    placed: List[dict] = []
    placed_binders = []
    result_map = dict(overflow_candidates)

    for gap in gaps:
        if not gap["assigned"]:
            continue

        assigned = _sort_by_node_angle(gap["assigned"], gap, centroid, G, overflow_candidates)
        n = len(assigned)

        for i, node_id in enumerate(assigned):
            slot_angle = gap["a_left"] + gap["gap_size"] * (i + 1) / (n + 1)

            overflow_label = overflow_candidates[node_id]
            w, h = _label_wh(overflow_label)
            node_pos = G.nodes[overflow_label.node_id]["pos"]

            pos = _place_in_fan(
                slot_angle, 
                gap["gap_size"],
                centroid, 
                outer_polygon,
                placed_union, 
                w, h, 
                node_pos,
                overflow_label.node_id, 
                node_id,
                G, placed, placed_binders,
                overflow_candidates,
                label_candidates
            )

            if pos is None:
                print(f"Warning: could not place overflow label for node {node_id} "
                      f"in gap ({gap['node_left']}, {gap['node_right']})")
                continue
            print(f'Success: placed label for node {node_id}')

            new_cx, new_cy, anchor_name, anchor_pt = pos

            binder = LineString([anchor_pt, node_pos])
            placed.append({
                'label_id': node_id,
                'position': (new_cx, new_cy),
                'anchor': anchor_name,
                'anchor_pt': anchor_pt,
                'binding_line': binder,
            })
            placed_binders.append(binder)

            # Carve out expanded bbox for next placements
            ew, eh = _label_wh_expanded(overflow_label)
            placed_union = placed_union.union(_label_bbox_polygon(new_cx, new_cy, ew, eh))

            update_overflow_label_position(overflow_label, new_cx, new_cy, anchor_name)
            result_map[node_id] = overflow_label

    return result_map