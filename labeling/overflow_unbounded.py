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
    overflow_candidates: Dict[int, OverflowLabel],
    label_candidates: Dict[int, List[LabelCandidate]]
) -> Optional[Tuple[float, float, str, Tuple]]:
    '''
    '''
    cx, cy = centroid
    node_pt = np.array(node_pos)
    label = overflow_candidates[own_label_id]
    ew, eh = _label_wh_expanded(label)
    iw, ih = _ink_wh(label)

    # already placed outer overflow bboxes
    placed_bboxes = [
        _label_bbox_polygon(*rec['position'], *_label_wh_expanded(overflow_candidates[rec['label_id']]))
        for rec in placed if rec['label_id'] != own_label_id
    ]

    # define candidate rays and sort by offset of central angle
    num_rays = 10
    spread = min(math.radians(30), gap_size * 0.8)
    offsets = sorted(np.linspace(-spread/2, spread/2, num_rays), key=abs)
    candidate_angles = [central_angle + off for off in offsets]

    best_candidate = None
    min_dist_found = float('inf')

    # check each ray
    for angle in candidate_angles:
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
        start_dist = max(0.0, exit_dist - OUTER_STEP * 4)

        for step in range(OUTER_MAX_STEPS):
            dist = start_dist + OUTER_STEP * step
            
            # If this ray is already worse than another ray's success, move on
            if dist >= min_dist_found:
                break

            origin_x, origin_y = cx + dx * dist, cy + dy * dist
            expanded_bbox = _label_bbox_polygon(origin_x, origin_y, ew, eh)

            # overlaps with bounded space
            if outer_polygon.intersects(expanded_bbox): continue
            # overlaps with placed label candidate
            if placed_union.intersects(expanded_bbox): continue
            # overlaps with placed outer overflow candidate
            if any(expanded_bbox.intersects(pb) for pb in placed_bboxes): continue
            # overlaps with a node of the drawing
            if any(
                expanded_bbox.contains(Point(data['pos']))
                for node_id, data in G.nodes(data=True)
                if isinstance(node_id, int) and node_id != own_node_id
            ):
                continue

            # sort all anchors based on their distance to the respective node
            own_ink_bbox = _label_bbox_polygon(origin_x, origin_y, iw, ih)
            anchors = _anchor_points(origin_x, origin_y, w, h)
            sorted_anchors = sorted(
                anchors.keys(),
                key=lambda name: np.hypot(
                    anchors[name][0] - node_pt[0],
                    anchors[name][1] - node_pt[1],
                )
            )

            for anchor_name in sorted_anchors:
                anchor_pt = anchors[anchor_name]
                line = LineString([anchor_pt, node_pos])

                # line crosses ink of own label
                if own_ink_bbox.intersects(line):
                    continue

                if binding_line_valid(line, own_node_id, own_label_id, G, placed, overflow_candidates, label_candidates):
                    min_dist_found = dist
                    best_candidate = (origin_x, origin_y, anchor_name, anchor_pt)
                    break # best position for this ray
            
            if best_candidate and dist == min_dist_found:
                break # check next ray

    return best_candidate

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
                G, placed, 
                overflow_candidates,
                label_candidates
            )

            if pos is None:
                print(f"Warning: could not place overflow label for node {node_id} "
                      f"in gap ({gap['node_left']}, {gap['node_right']})")
                continue

            new_cx, new_cy, anchor_name, anchor_pt = pos

            placed.append({
                'label_id': node_id,
                'position': (new_cx, new_cy),
                'anchor': anchor_name,
                'anchor_pt': anchor_pt,
                'binding_line': LineString([anchor_pt, node_pos]),
            })

            # Carve out expanded bbox for next placements
            ew, eh = _label_wh_expanded(overflow_label)
            placed_union = placed_union.union(_label_bbox_polygon(new_cx, new_cy, ew, eh))

            update_overflow_label_position(overflow_label, new_cx, new_cy, anchor_name)
            result_map[node_id] = overflow_label

    return result_map