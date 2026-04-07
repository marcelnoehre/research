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
    _anchor_point_from_bbox,
    _binding_line_valid,
    _update_overflow_label_position,
)

OUTER_STEP = 0.5
OUTER_MAX_STEPS = 500
WEDGE_RADIUS = 1e4


def _angle_from_centroid(centroid: Tuple[float, float], point: Tuple[float, float]) -> float:
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


def _angular_gap_between(a1: float, a2: float) -> float:
    """CCW arc from a1 to a2 in [0, 2pi)."""
    return (a2 - a1) % (2 * math.pi)


def _sort_by_node_angle(
    assigned: List[int],
    gap: Dict,
    centroid: Tuple[float, float],
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
) -> List[int]:
    """
    Sort assigned label ids by their node's angle from centroid within the gap,
    so slot order matches node order and binding lines don't cross each other.
    """
    a_left = gap["a_left"]

    def node_angle_in_gap(node_id):
        node_pos = G.nodes[overflow_candidates[node_id].node_id]["pos"]
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
    """Pie-slice from centroid outward between a_left and a_right (CCW), minus the outer polygon."""
    cx, cy = centroid
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
) -> None:
    """Log exactly what blocks this binding line."""
    for rec in placed:
        if rec["label_id"] == own_label_id:
            continue
        cx, cy = rec["position"]
        w, h = _label_wh_expanded(overflow_candidates[rec["label_id"]])
        bbox = _label_bbox_polygon(cx, cy, w, h)
        if line.intersects(bbox):
            print(f"    [binding] blocked by placed label {rec['label_id']} at {rec['position']}")
        if "binding_line" in rec and rec["binding_line"] is not None:
            if line.crosses(rec["binding_line"]):
                print(f"    [binding] crosses binding line of label {rec['label_id']}")

    for node_id, data in G.nodes(data=True):
        if node_id == own_node_id:
            continue
        if not isinstance(node_id, int):
            continue
        if line.distance(Point(data["pos"])) < 1e-6:
            print(f"    [binding] blocked by node {node_id} at {data['pos']}")


def _place_along_ray(
    angle: float,
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
    label_candidates: Dict[int, List[LabelCandidate]],
    debug: bool = False,
) -> Optional[Tuple[float, float, str, Tuple]]:
    cx, cy = centroid
    dx, dy = math.cos(angle), math.sin(angle)
    node_pt = np.array(node_pos)
    hw, hh = w / 2, h / 2

    # Expanded dims for collision
    label = overflow_candidates[own_label_id]
    ew, eh = _label_wh_expanded(label)

    # Inner bbox half-dims for anchor interpolation
    ibl, ibr, itr, itl = label.inner_bbox_corners
    half_iw = (ibr[0] - ibl[0]) / 2
    half_ih = (itr[1] - ibr[1]) / 2

    placed_bboxes = [
        _label_bbox_polygon(*rec["position"], *_label_wh_expanded(overflow_candidates[rec["label_id"]]))
        for rec in placed if rec["label_id"] != own_label_id
    ]

    # Find where ray exits the outer polygon
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
    fail_node_in_bbox = 0
    fail_binding = 0
    best_binding_fallback: Optional[Tuple[float, float, str, Tuple]] = None

    for step in range(OUTER_MAX_STEPS):
        dist = exit_dist + OUTER_STEP * step
        origin_x = cx + dx * dist
        origin_y = cy + dy * dist

        expanded_bbox = _label_bbox_polygon(origin_x, origin_y, ew, eh)

        if outer_polygon.intersects(expanded_bbox):
            fail_outer += 1
            continue
        if placed_union.intersects(expanded_bbox):
            fail_placed_union += 1
            continue
        if any(expanded_bbox.intersects(pb) for pb in placed_bboxes):
            fail_placed_bboxes += 1
            continue
        if any(
            expanded_bbox.contains(Point(data["pos"]))
            for node_id, data in G.nodes(data=True)
            if isinstance(node_id, int) and node_id != own_node_id
        ):
            fail_node_in_bbox += 1
            continue

        # Tight bbox corners for anchor interpolation
        candidate_bbox = (
            (origin_x - hw, origin_y - hh),
            (origin_x + hw, origin_y - hh),
            (origin_x + hw, origin_y + hh),
            (origin_x - hw, origin_y + hh),
        )
        candidate_inner = (
            (origin_x - half_iw, origin_y - half_ih),
            (origin_x + half_iw, origin_y - half_ih),
            (origin_x + half_iw, origin_y + half_ih),
            (origin_x - half_iw, origin_y + half_ih),
        )

        sorted_anchors = sorted(
            _anchor_points(origin_x, origin_y, w, h).keys(),
            key=lambda name: np.hypot(
                _anchor_point_from_bbox(name, candidate_bbox, candidate_inner)[0] - node_pt[0],
                _anchor_point_from_bbox(name, candidate_bbox, candidate_inner)[1] - node_pt[1],
            )
        )

        found = False
        for anchor_name in sorted_anchors:
            anchor_pt = _anchor_point_from_bbox(anchor_name, candidate_bbox, candidate_inner)
            line = LineString([anchor_pt, node_pos])
            if _binding_line_valid(line, own_node_id, own_label_id, G, placed, overflow_candidates, label_candidates):
                return origin_x, origin_y, anchor_name, anchor_pt
            found = False

        if not found:
            if best_binding_fallback is None:
                best_anchor_name = sorted_anchors[0]
                best_anchor_pt = _anchor_point_from_bbox(best_anchor_name, candidate_bbox, candidate_inner)
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
            f"fail_placed_bboxes={fail_placed_bboxes}  "
            f"fail_node_in_bbox={fail_node_in_bbox}  fail_binding={fail_binding}  "
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

    # Already-placed label union using expanded bboxes
    placed_union = unary_union([
        Polygon(cand.expanded_bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]) if label_candidates else Polygon()

    # Build gaps from adjacent outer nodes
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

            pos = _place_along_ray(
                slot_angle, centroid, outer_polygon,
                placed_union, w, h, node_pos,
                overflow_label.node_id, node_id,
                G, placed, overflow_candidates,
                label_candidates
            )

            if pos is None:
                # Retry with debug
                _place_along_ray(
                    slot_angle, centroid, outer_polygon,
                    placed_union, w, h, node_pos,
                    overflow_label.node_id, node_id,
                    G, placed, overflow_candidates,
                    label_candidates, debug=True,
                )
                print(f"Warning: could not place overflow label for node {node_id} "
                      f"in gap ({gap['node_left']}, {gap['node_right']})")
                continue

            new_cx, new_cy, anchor_name, anchor_pt = pos

            placed.append({
                "label_id":     node_id,
                "position":     (new_cx, new_cy),
                "anchor":       anchor_name,
                "anchor_pt":    anchor_pt,
                "binding_line": LineString([anchor_pt, node_pos]),
            })

            # Carve out expanded bbox for next placements
            ew, eh = _label_wh_expanded(overflow_label)
            placed_union = placed_union.union(_label_bbox_polygon(new_cx, new_cy, ew, eh))

            _update_overflow_label_position(overflow_label, new_cx, new_cy)
            overflow_label.anchor = anchor_name
            result_map[node_id] = overflow_label

    return result_map