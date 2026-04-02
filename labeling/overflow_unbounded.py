import networkx as nx
import math
from typing import Dict, List, Tuple
from shapely.geometry import Polygon
from label import LabelCandidate, OverflowLabel

OVERFLOW_PADDING = 0.5

def _angle_from_centroid(centroid: Tuple[float, float], point: Tuple[float, float]) -> float:
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


def _angular_gap_between(a1: float, a2: float) -> float:
    """CCW arc from a1 to a2, in [0, 2pi)."""
    return (a2 - a1) % (2 * math.pi)


def _bbox_corners_from_center_and_existing(
    center: Tuple[float, float],
    reference: OverflowLabel,
) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
    """
    Recompute bbox corners by translating the reference label's bbox
    to the new center. Preserves label size/shape.
    BL, BR, TR, TL order.
    """
    ref_cx, ref_cy = reference.center
    dx = center[0] - ref_cx
    dy = center[1] - ref_cy
    return tuple(
        (corner[0] + dx, corner[1] + dy)
        for corner in reference.bbox_corners
    )


def outer_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    outer_nodes: List[int],
) -> Dict[int, OverflowLabel]:

    # --- Build outer polygon + centroid ---
    outer_positions = [G.nodes[n]["pos"] for n in outer_nodes]
    outer_polygon = Polygon(outer_positions)
    cx, cy = outer_polygon.centroid.x, outer_polygon.centroid.y

    # --- Compute angle + radius for each outer node label ---
    outer_label_angles: List[Tuple[float, int, float]] = []  # (angle, node_id, radius)

    for node_id in outer_nodes:
        candidates = label_candidates.get(node_id, [])
        if candidates:
            lc = candidates[0]
            pt = lc.center
        else:
            pt = G.nodes[node_id]["pos"]

        angle = _angle_from_centroid((cx, cy), pt)
        radius = math.hypot(pt[0] - cx, pt[1] - cy)
        outer_label_angles.append((angle, node_id, radius))

    outer_label_angles.sort(key=lambda x: x[0])
    n_outer = len(outer_label_angles)

    mean_outer_radius = sum(r for _, _, r in outer_label_angles) / n_outer
    placement_radius = mean_outer_radius + OVERFLOW_PADDING

    # --- Build gaps (circular, sorted CCW) ---
    # gap key: (node_left, node_right) — stable identity for grouping
    gaps: Dict[Tuple[int, int], Dict] = {}

    for i in range(n_outer):
        a1, node_left, _ = outer_label_angles[i]
        a2, node_right, _ = outer_label_angles[(i + 1) % n_outer]
        gap_size = _angular_gap_between(a1, a2)
        gaps[(node_left, node_right)] = {
            "a_left": a1,
            "gap_size": gap_size,
            "node_left": node_left,
            "node_right": node_right,
            "assigned": [],  # overflow node_ids assigned here
        }

    # --- Assign each unplaced overflow label to its best gap ---
    unplaced = {
        nid: ol for nid, ol in overflow_candidates.items()
        if ol.anchor == 'overflow'
    }

    for node_id, overflow_label in unplaced.items():
        lx, ly = overflow_label.center
        outward_angle = _angle_from_centroid((cx, cy), (lx, ly))

        best_gap_key = None
        best_gap_size = -1.0

        for gap_key, gap in gaps.items():
            angle_into_gap = _angular_gap_between(gap["a_left"], outward_angle)
            if angle_into_gap <= gap["gap_size"]:
                if gap["gap_size"] > best_gap_size:
                    best_gap_size = gap["gap_size"]
                    best_gap_key = gap_key

        if best_gap_key is None:
            # Fallback: widest gap
            best_gap_key = max(gaps, key=lambda k: gaps[k]["gap_size"])

        gaps[best_gap_key]["assigned"].append(node_id)
        print(
            f"Overflow label for node {node_id} → gap between "
            f"[node {gaps[best_gap_key]['node_left']}] and [node {gaps[best_gap_key]['node_right']}] "
            f"(gap: {math.degrees(gaps[best_gap_key]['gap_size']):.1f}°)"
        )

    # --- Place labels evenly within each gap ---
    result = dict(overflow_candidates)

    for gap in gaps.values():
        assigned = gap["assigned"]
        if not assigned:
            continue

        a_left = gap["a_left"]
        gap_size = gap["gap_size"]
        n = len(assigned)

        # Divide gap into (n+1) slots so labels don't sit on the boundary labels
        for i, node_id in enumerate(assigned):
            slot_angle = a_left + gap_size * (i + 1) / (n + 1)

            new_cx = cx + placement_radius * math.cos(slot_angle)
            new_cy = cy + placement_radius * math.sin(slot_angle)
            new_center = (new_cx, new_cy)

            overflow_label = overflow_candidates[node_id]
            new_bbox = _bbox_corners_from_center_and_existing(new_center, overflow_label)

            # Write back
            result[node_id] = OverflowLabel(
                node_id=overflow_label.node_id,
                label_type=overflow_label.label_type,
                bbox_corners=new_bbox,
                inner_bbox_corners=overflow_label.inner_bbox_corners,  # unchanged for now
                center=new_center,
                anchor=overflow_label.anchor,  # still 'overflow', you handle anchoring elsewhere
                text=overflow_label.text,
            )

            print(
                f"  → node {node_id} placed at angle {math.degrees(slot_angle):.1f}° "
                f"radius {placement_radius:.1f} → center {new_center}"
            )

    return result