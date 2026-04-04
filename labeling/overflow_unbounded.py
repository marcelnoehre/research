import math
import numpy as np
import networkx as nx

from shapely.geometry import LineString, Point, Polygon
from shapely import unary_union
from typing import Dict, List, Optional, Tuple

from label import LabelCandidate, OverflowLabel
from overflow_bounded import (
    _label_wh,
    _label_bbox_polygon,
    _anchor_points,
    _binding_line_valid,
    _update_overflow_label_position,
)

OUTER_STEP = 0.5
OUTER_MAX_STEPS = 500
WEDGE_RADIUS = 1e4  # large enough to always extend beyond any label


def _angle_from_centroid(centroid: Tuple[float, float], point: Tuple[float, float]) -> float:
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


def _angular_gap_between(a1: float, a2: float) -> float:
    """CCW arc from a1 to a2 in [0, 2pi)."""
    return (a2 - a1) % (2 * math.pi)


def _wedge_polygon(
    centroid: Tuple[float, float],
    a_left: float,
    a_right: float,
    outer_polygon: Polygon,
    radius: float = WEDGE_RADIUS,
) -> Polygon:
    """
    Pie-slice from centroid outward between a_left and a_right (CCW),
    minus the outer polygon itself.
    """
    cx, cy = centroid
    # Sample arc points for a smooth wedge
    arc_steps = max(16, int(math.degrees(_angular_gap_between(a_left, a_right))))
    arc = []
    for i in range(arc_steps + 1):
        t = i / arc_steps
        angle = a_left + _angular_gap_between(a_left, a_right) * t
        arc.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    wedge = Polygon([(cx, cy)] + arc + [(cx, cy)])
    return wedge.difference(outer_polygon)

def _binding_line_debug(
    line: LineString,
    own_node_id: int,
    own_label_id: int,
    G: nx.Graph,
    placed: List[dict],
    overflow_candidates: Dict[int, OverflowLabel],
    outer_polygon: Optional[Polygon] = None,
) -> None:
    """Log exactly what blocks this binding line."""
    if outer_polygon is not None:
        if outer_polygon.contains(line):
            print(f"    [binding] blocked by outer_polygon.contains")
        elif outer_polygon.crosses(line):
            print(f"    [binding] blocked by outer_polygon.crosses")

    for rec in placed:
        if rec["label_id"] == own_label_id:
            continue
        cx, cy = rec["position"]
        w, h = _label_wh(overflow_candidates[rec["label_id"]])
        bbox = _label_bbox_polygon(cx, cy, w, h)
        if line.intersects(bbox):
            print(f"    [binding] blocked by placed label {rec['label_id']} at {rec['position']}")

def _place_along_ray(
    angle: float,
    centroid: Tuple[float, float],
    outer_polygon: Polygon,          # positional — used for exit_dist + bbox check
    placed_union: Polygon,
    w: float,
    h: float,
    node_pos: Tuple[float, float],
    own_node_id: int,
    own_label_id: int,
    G: nx.Graph,
    placed: List[dict],
    overflow_candidates: Dict[int, OverflowLabel],
    debug: bool = False,
) -> Optional[Tuple[float, float, str, Tuple]]:
    cx, cy = centroid
    dx, dy = math.cos(angle), math.sin(angle)
    node_pt = np.array(node_pos)

    placed_bboxes = [
        _label_bbox_polygon(*rec["position"], *_label_wh(overflow_candidates[rec["label_id"]]))
        for rec in placed if rec["label_id"] != own_label_id
    ]

    ray_line = LineString([(cx, cy), (cx + WEDGE_RADIUS * dx, cy + WEDGE_RADIUS * dy)])
    intersection = outer_polygon.exterior.intersection(ray_line)

    if intersection.is_empty:
        exit_dist = 0.0
    elif hasattr(intersection, "geoms"):
        exit_dist = max(
            math.hypot(pt.x - cx, pt.y - cy)
            for pt in intersection.geoms
        )
    else:
        exit_dist = math.hypot(intersection.x - cx, intersection.y - cy)

    fail_outer = 0
    fail_placed_union = 0
    fail_placed_bboxes = 0
    fail_binding = 0

    best_binding_fallback: Optional[Tuple[float, float, str, Tuple]] = None

    for step in range(OUTER_MAX_STEPS):
        dist = exit_dist + OUTER_STEP * step
        origin_x = cx + dx * dist
        origin_y = cy + dy * dist

        bbox = _label_bbox_polygon(origin_x, origin_y, w, h)

        if outer_polygon.intersects(bbox):
            fail_outer += 1
            continue
        if placed_union.intersects(bbox):
            fail_placed_union += 1
            continue
        if any(bbox.intersects(pb) for pb in placed_bboxes):
            fail_placed_bboxes += 1
            continue

        anchors = _anchor_points(origin_x, origin_y, w, h)
        sorted_anchors = sorted(
            anchors.items(),
            key=lambda kv: np.hypot(kv[1][0] - node_pt[0], kv[1][1] - node_pt[1])
        )
        for anchor_name, anchor_pt in sorted_anchors:
            line = LineString([anchor_pt, node_pos])
            if _binding_line_valid(
                line, own_node_id, own_label_id,
                G, placed, overflow_candidates
            ):
                return origin_x, origin_y, anchor_name, anchor_pt

        # All anchors failed binding — save first step as fallback
        if best_binding_fallback is None:
            best_anchor_name, best_anchor_pt = sorted_anchors[0]
            best_binding_fallback = (origin_x, origin_y, best_anchor_name, best_anchor_pt)
            if debug:
                fallback_line = LineString([best_anchor_pt, node_pos])
                print(f"    [binding] label_center=({origin_x:.2f},{origin_y:.2f}) "
                    f"anchor_pt={best_anchor_pt} node_pos={node_pos}")
                print(f"    [binding] line: {list(fallback_line.coords)}")
                _binding_line_debug(
                    fallback_line, own_node_id, own_label_id,
                    G, placed, overflow_candidates,
                )
        fail_binding += 1

    if debug:
        print(
            f"  [debug] label {own_label_id} / node {own_node_id} — "
            f"angle {math.degrees(angle):.1f}°  exit_dist {exit_dist:.1f}  "
            f"steps {OUTER_MAX_STEPS}  "
            f"fail_outer={fail_outer}  fail_placed_union={fail_placed_union}  "
            f"fail_placed_bboxes={fail_placed_bboxes}  fail_binding={fail_binding}  "
            f"fallback={'yes' if best_binding_fallback else 'no'}"
        )

    return best_binding_fallback

def outer_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    outer_nodes: List[int],
) -> Dict[int, OverflowLabel]:

    outer_positions = [G.nodes[n]["pos"] for n in outer_nodes]
    outer_polygon = Polygon(outer_positions)
    cx, cy = outer_polygon.centroid.x, outer_polygon.centroid.y
    centroid = (cx, cy)

    # Already-placed label union to subtract
    placed_union = unary_union([
        Polygon(cand.bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]) if label_candidates else Polygon()

    # --- Build gaps from adjacent outer nodes ---
    node_angles = [
        (_angle_from_centroid(centroid, G.nodes[n]["pos"]), n)
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
            "a_left":    a_left,
            "a_right":   a_right,
            "gap_size":  gap_size,
            "node_left": node_left,
            "node_right": node_right,
            "wedge":     wedge,
            "assigned":  [],
        })

    # --- Assign unplaced overflow labels to their natural gap ---
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

    # --- Place labels gap by gap ---
    placed: List[dict] = []
    result_map = dict(overflow_candidates)

    for gap in gaps:
        assigned = gap["assigned"]
        if not assigned:
            continue

        n = len(assigned)
        wedge = gap["wedge"]

        for i, node_id in enumerate(assigned):
            # Even angular spacing: 1 label → centered, N labels → evenly subdivided
            slot_angle = gap["a_left"] + gap["gap_size"] * (i + 1) / (n + 1)

            overflow_label = overflow_candidates[node_id]
            w, h = _label_wh(overflow_label)
            node_pos = G.nodes[overflow_label.node_id]["pos"]

            pos = _place_along_ray(
                slot_angle, centroid, outer_polygon,
                placed_union, w, h, node_pos,
                overflow_label.node_id, node_id,
                G, placed, overflow_candidates,
                debug=True
            )

            if pos is None:
                print(
                    f"Warning: could not place overflow label for node {node_id} "
                    f"in gap ({gap['node_left']}, {gap['node_right']})"
                )
                continue

            new_cx, new_cy, anchor_name, anchor_pt = pos

            placed.append({
                "label_id":     node_id,
                "position":     (new_cx, new_cy),
                "anchor":       anchor_name,
                "anchor_pt":    anchor_pt,
                "binding_line": LineString([anchor_pt, node_pos]),
            })

            placed_union = placed_union.union(_label_bbox_polygon(new_cx, new_cy, w, h))

            _update_overflow_label_position(overflow_label, new_cx, new_cy)
            overflow_label.anchor = anchor_name
            result_map[node_id] = overflow_label

    return result_map