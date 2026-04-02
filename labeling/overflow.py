import numpy as np
import networkx as nx

from shapely import unary_union
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import voronoi_diagram
from typing import Dict, List, Optional, Tuple

from label import LabelCandidate, OverflowLabel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_BUFFER = 0.1


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _label_wh(label: OverflowLabel) -> Tuple[float, float]:
    bl, br, tr, tl = label.bbox_corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])


def _label_bbox_polygon(cx: float, cy: float, w: float, h: float) -> Polygon:
    hw, hh = w / 2, h / 2
    return Polygon([
        (cx - hw, cy - hh), (cx + hw, cy - hh),
        (cx + hw, cy + hh), (cx - hw, cy + hh),
    ])


def _anchor_points(cx: float, cy: float, w: float, h: float) -> Dict[str, Tuple[float, float]]:
    hw, hh = w / 2, h / 2
    return {
        "top":          (cx,      cy + hh),
        "bottom":       (cx,      cy - hh),
        "left":         (cx - hw, cy     ),
        "right":        (cx + hw, cy     ),
        "top_left":     (cx - hw, cy + hh),
        "top_right":    (cx + hw, cy + hh),
        "bottom_left":  (cx - hw, cy - hh),
        "bottom_right": (cx + hw, cy - hh),
    }


def _fits(space: Polygon, w: float, h: float) -> bool:
    """Check whether a w×h rectangle can be placed somewhere inside space."""
    if space.is_empty:
        return False
    s_minx, s_miny, s_maxx, s_maxy = space.bounds
    if w > (s_maxx - s_minx) or h > (s_maxy - s_miny) or w * h > space.area:
        return False
    # Two-pass erosion: first by smaller half-extent, then by remaining half
    smaller, larger = min(w, h), max(w, h)
    eroded = space.buffer(-smaller / 2, join_style=2)
    if eroded.is_empty:
        return False
    eroded = eroded.buffer(-(larger - smaller) / 2, join_style=2)
    return not eroded.is_empty


def _eroded_space(space: Polygon, w: float, h: float) -> Polygon:
    """Return the set of valid center positions for a w×h label inside space."""
    smaller, larger = min(w, h), max(w, h)
    eroded = space.buffer(-smaller / 2, join_style=2)
    if eroded.is_empty:
        return eroded
    return eroded.buffer(-(larger - smaller) / 2, join_style=2)


def _angle_around(origin: Tuple, point: Tuple) -> float:
    return np.arctan2(point[1] - origin[1], point[0] - origin[0])


def _voronoi_region_for_node(
    face_polygon: Polygon,
    node_positions: List[Tuple],
    idx: int,
) -> Polygon:
    """
    Return the Voronoi cell inside face_polygon that belongs to node_positions[idx].
    Falls back to the full face if only one node or if voronoi fails.
    """
    if len(node_positions) == 1:
        return face_polygon

    points_geom = GeometryCollection([Point(p) for p in node_positions])
    try:
        regions = voronoi_diagram(points_geom, envelope=face_polygon)
    except Exception:
        return face_polygon

    target = Point(node_positions[idx])
    clipped = [geom.intersection(face_polygon) for geom in regions.geoms]

    for region in clipped:
        if not region.is_empty and region.contains(target):
            return region

    # Fallback: closest clipped region
    return min(clipped, key=lambda g: g.distance(target))


# ---------------------------------------------------------------------------
# Binding-line validation
# ---------------------------------------------------------------------------

def _binding_line_valid(
    line: LineString,
    own_node_id: int,
    G: nx.Graph,
    placed: List[dict],
    overflow_candidates: Dict[int, OverflowLabel],
) -> bool:
    """
    A binding line is valid if it:
      - does not intersect any placed label's inner_bbox
      - does not come within NODE_BUFFER of any graph node (except the label's own node)
    """
    # Check placed labels' inner bboxes
    for rec in placed:
        inner_bbox = Polygon(overflow_candidates[rec["label_id"]].inner_bbox_corners)
        if line.intersects(inner_bbox):
            return False

    # Check graph nodes (buffered), excluding own node
    for node_id, data in G.nodes(data=True):
        if node_id == own_node_id:
            continue
        if line.intersects(Point(data["pos"]).buffer(NODE_BUFFER)):
            return False

    return True


# ---------------------------------------------------------------------------
# Position search: find valid (cx, cy) + anchor inside the eroded space
# ---------------------------------------------------------------------------

def _find_valid_position(
    space: Polygon,
    w: float,
    h: float,
    node_pos: Tuple,
    own_node_id: int,
    G: nx.Graph,
    placed: List[dict],
    overflow_candidates: Dict[int, OverflowLabel],
) -> Optional[Tuple[float, float, str, Tuple]]:
    """
    Sample candidate center positions within the eroded space.
    For each, try anchors in order of proximity to node_pos.
    Returns (cx, cy, anchor_name, anchor_pt) or None if nothing valid found.
    """
    eroded = _eroded_space(space, w, h)
    if eroded.is_empty:
        return None

    minx, miny, maxx, maxy = eroded.bounds
    step = min(w, h) / 2

    # Centroid first, then grid — sorted by distance to centroid so we
    # prefer visually centered positions
    centroid = eroded.centroid
    candidates = [(centroid.x, centroid.y)]

    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            pt = Point(x, y)
            if eroded.contains(pt):
                candidates.append((x, y))
            y += step
        x += step

    candidates.sort(
        key=lambda p: np.hypot(p[0] - centroid.x, p[1] - centroid.y)
    )

    for cx, cy in candidates:
        anchors = _anchor_points(cx, cy, w, h)
        sorted_anchors = sorted(
            anchors.items(),
            key=lambda kv: np.hypot(kv[1][0] - node_pos[0], kv[1][1] - node_pos[1])
        )
        for anchor_name, anchor_pt in sorted_anchors:
            line = LineString([anchor_pt, node_pos])
            if _binding_line_valid(line, own_node_id, G, placed, overflow_candidates):
                return cx, cy, anchor_name, anchor_pt

    return None


# ---------------------------------------------------------------------------
# Fitting check: which faces does each label fit in?
# ---------------------------------------------------------------------------

def _compute_results(
    overflow_candidates: Dict[int, OverflowLabel],
    processed_faces: List[Tuple],       # (face_id, remaining_space)
) -> Dict[int, List[int]]:
    """For each overflow label, return the list of face_ids where it fits."""
    results: Dict[int, List[int]] = {}
    for label_id, label in overflow_candidates.items():
        w, h = _label_wh(label)
        results[label_id] = [
            fid for fid, space in processed_faces
            if _fits(space, w, h)
        ]
    return results


# ---------------------------------------------------------------------------
# Multi-label repositioning within a face
# ---------------------------------------------------------------------------

def _reposition_face_labels(
    face_id: int,
    face_placements: Dict[int, List[dict]],
    space_map: Dict[int, Polygon],
    centers: List[Tuple],
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
):
    """
    When multiple labels share a face, sort them by angle of their nodes
    around the face center (prevents crossing lines), assign each to its
    Voronoi region, and update placement records in-place.
    """
    records = face_placements[face_id]
    if len(records) <= 1:
        return

    fc = np.array(centers[face_id])
    face_poly = space_map[face_id].convex_hull  # approximate full face for voronoi

    # Sort labels by angle of their node around the face center
    sorted_records = sorted(
        records,
        key=lambda rec: _angle_around(
            fc, G.nodes[overflow_candidates[rec["label_id"]].node_id]["pos"]
        )
    )

    node_positions = [
        G.nodes[overflow_candidates[rec["label_id"]].node_id]["pos"]
        for rec in sorted_records
    ]

    for i, rec in enumerate(sorted_records):
        label = overflow_candidates[rec["label_id"]]
        w, h = _label_wh(label)
        node_pos = G.nodes[label.node_id]["pos"]

        region = _voronoi_region_for_node(face_poly, node_positions, i)
        c = region.centroid
        cx, cy = c.x, c.y

        anchors = _anchor_points(cx, cy, w, h)
        anchor_name, anchor_pt = min(
            anchors.items(),
            key=lambda kv: np.hypot(kv[1][0] - node_pos[0], kv[1][1] - node_pos[1])
        )

        rec["position"] = (cx, cy)
        rec["anchor"] = anchor_name
        rec["anchor_pt"] = anchor_pt
        rec["binding_line"] = LineString([anchor_pt, node_pos])


# ---------------------------------------------------------------------------
# Main placement loop
# ---------------------------------------------------------------------------

def place_overflow_labels(
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
    results: Dict[int, List[int]],          # label_id -> fitting face_ids
    processed_faces: List[Tuple],           # (face_id, remaining_space)
    centers: List[Tuple],                   # centers[face_id] = (x, y)
) -> List[dict]:
    """
    Greedily place overflow labels into faces, largest face first.
    Each face gets the label whose node is closest to the face center.
    After each placement the face's remaining space is updated and faces
    are re-sorted, so a face can receive multiple labels if it is still
    the largest after each subtraction.
    Multiple labels in a face are repositioned to Voronoi regions sorted
    by angle, guaranteeing non-crossing binding lines.
    """
    space_map: Dict[int, Polygon] = dict(processed_faces)
    placed: List[dict] = []
    placed_label_ids: set = set()
    face_placements: Dict[int, List[dict]] = {}

    # Invert results for fast lookup: face_id -> [label_ids that fit here]
    face_to_labels: Dict[int, List[int]] = {}
    for label_id, face_ids in results.items():
        for fid in face_ids:
            face_to_labels.setdefault(fid, []).append(label_id)

    while True:
        # Re-sort faces by current remaining area (largest first)
        face_order = sorted(
            space_map.keys(),
            key=lambda fid: space_map[fid].area,
            reverse=True,
        )

        chosen_face = None
        chosen_label = None

        for face_id in face_order:
            space = space_map[face_id]
            face_center = np.array(centers[face_id])

            candidates = [
                lid for lid in face_to_labels.get(face_id, [])
                if lid not in placed_label_ids
                and _fits(space, *_label_wh(overflow_candidates[lid]))
            ]
            if not candidates:
                continue

            # Pick the candidate whose node is closest to this face's center
            chosen_label = min(
                candidates,
                key=lambda lid: np.linalg.norm(
                    np.array(G.nodes[overflow_candidates[lid].node_id]["pos"]) - face_center
                )
            )
            chosen_face = face_id
            break

        if chosen_face is None:
            break   # No more labels can be placed anywhere

        label = overflow_candidates[chosen_label]
        w, h = _label_wh(label)
        node_pos = G.nodes[label.node_id]["pos"]
        space = space_map[chosen_face]

        # Find a valid center position with a clear binding line
        result = _find_valid_position(
            space, w, h, node_pos, label.node_id,
            G, placed, overflow_candidates,
        )

        if result is None:
            # No valid position in this face — exclude and retry in next iteration
            results[chosen_label] = [f for f in results[chosen_label] if f != chosen_face]
            face_to_labels[chosen_face] = [
                lid for lid in face_to_labels[chosen_face] if lid != chosen_label
            ]
            continue

        cx, cy, anchor_name, anchor_pt = result
        binding_line = LineString([anchor_pt, node_pos])

        record = {
            "label_id":     chosen_label,
            "face_id":      chosen_face,
            "position":     (cx, cy),
            "anchor":       anchor_name,
            "anchor_pt":    anchor_pt,
            "binding_line": binding_line,
        }

        placed.append(record)
        face_placements.setdefault(chosen_face, []).append(record)
        placed_label_ids.add(chosen_label)

        # Subtract placed label bbox from face's remaining space
        space_map[chosen_face] = space_map[chosen_face].difference(
            _label_bbox_polygon(cx, cy, w, h)
        )

        # If this face now has multiple labels, reposition all of them
        if len(face_placements[chosen_face]) > 1:
            _reposition_face_labels(
                chosen_face, face_placements, space_map,
                centers, G, overflow_candidates,
            )

    return placed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def inner_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    bounded_faces: List[List],
    centers: List[Tuple],
) -> List[dict]:
    """
    Full pipeline:
      1. Build face polygons, subtract already-placed label bboxes.
      2. Determine which overflow labels fit in which faces.
      3. Greedily place labels, largest face first, with valid binding lines.

    Returns a list of placement records, each containing:
      label_id, face_id, position (cx, cy), anchor name,
      anchor_pt, and binding_line (LineString).
    """
    # 1. Build face polygons sorted largest first
    face_data = sorted(
        [
            {
                "original_id": idx,
                "polygon": Polygon([G.nodes[n]["pos"] for n in face]),
            }
            for idx, face in enumerate(bounded_faces)
        ],
        key=lambda x: x["polygon"].area,
        reverse=True,
    )

    # 2. Subtract already-placed label bboxes from each face
    placed_union = unary_union([
        Polygon(cand.bbox_corners)
        for candidates in label_candidates.values()
        for cand in candidates
    ])

    processed_faces = [
        (d["original_id"], d["polygon"].difference(placed_union))
        for d in face_data
    ]

    # 3. Determine which faces each overflow label fits in
    results = _compute_results(overflow_candidates, processed_faces)

    # 4. Place labels
    return place_overflow_labels(G, overflow_candidates, results, processed_faces, centers)