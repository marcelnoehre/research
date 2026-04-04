import numpy as np
import networkx as nx

from shapely import unary_union
from shapely.geometry import LineString, Point, Polygon
from typing import Dict, List, Optional, Tuple

from label import LabelCandidate, OverflowLabel


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


def _eroded_space(space: Polygon, w: float, h: float) -> Polygon:
    """Return the set of valid center positions for a w×h label inside space."""
    smaller, larger = min(w, h), max(w, h)
    eroded = space.buffer(-smaller / 2, join_style=2)
    if eroded.is_empty:
        return eroded
    return eroded.buffer(-(larger - smaller) / 2, join_style=2)


def _fits(space: Polygon, w: float, h: float) -> bool:
    if space.is_empty:
        return False
    s_minx, s_miny, s_maxx, s_maxy = space.bounds
    if w > (s_maxx - s_minx) or h > (s_maxy - s_miny) or w * h > space.area:
        return False
    return not _eroded_space(space, w, h).is_empty

# ---------------------------------------------------------------------------
# Binding-line validation
# ---------------------------------------------------------------------------

def _binding_line_valid(
    line: LineString,
    own_node_id: int,
    own_label_id: int,
    G: nx.Graph,
    placed: List[dict],
    overflow_candidates: Dict[int, OverflowLabel],
) -> bool:
    for rec in placed:
        if rec["label_id"] == own_label_id:
            continue
        cx, cy = rec["position"]
        w, h = _label_wh(overflow_candidates[rec["label_id"]])
        if line.intersects(_label_bbox_polygon(cx, cy, w, h)):
            return False

    for node_id, data in G.nodes(data=True):
        if node_id == own_node_id:
            continue
        if not isinstance(node_id, int):
            continue
        if line.distance(Point(data["pos"])) < 1e-6:
            return False

    return True


# ---------------------------------------------------------------------------
# Position search
# ---------------------------------------------------------------------------

def _find_valid_position(
    space: Polygon,
    w: float,
    h: float,
    node_pos: Tuple,
    own_node_id: int,
    own_label_id: int,
    G: nx.Graph,
    placed: List[dict],
    overflow_candidates: Dict[int, OverflowLabel],
) -> Optional[Tuple[float, float, str, Tuple]]:
    """
    Find the valid center position inside the eroded space that is closest
    to node_pos, with a clear binding line. Returns (cx, cy, anchor, anchor_pt)
    or None.
    """
    eroded = _eroded_space(space, w, h)
    if eroded.is_empty:
        return None

    node_pt = np.array(node_pos)
    node_point = Point(node_pos)
    hw, hh = w / 2, h / 2

    # Seed candidates: for each of the 8 anchor offsets, compute the center
    # that would place that anchor exactly on the node (zero binding line length).
    # These are the optimal positions if they fall inside the eroded space.
    anchor_offsets = [
        (0,   -hh), (0,   hh),
        (-hw,  0),  (hw,  0),
        (-hw, -hh), (hw, -hh),
        (-hw,  hh), (hw,  hh),
    ]
    candidates = [
        (float(node_pt[0] - dx), float(node_pt[1] - dy))
        for dx, dy in anchor_offsets
    ]

    # Also add nearest point in eroded space as a fallback seed
    if eroded.contains(node_point):
        candidates.append((float(node_pt[0]), float(node_pt[1])))
    else:
        nearest_pt = eroded.boundary.interpolate(eroded.boundary.project(node_point))
        candidates.append((nearest_pt.x, nearest_pt.y))

    # Fine grid as further fallback
    minx, miny, maxx, maxy = eroded.bounds
    step = min(w, h) / 8
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if eroded.contains(Point(x, y)):
                candidates.append((x, y))
            y += step
        x += step

    # Sort by minimum anchor distance to node — true binding line length metric
    candidates.sort(key=lambda p: min(
        np.hypot(apt[0] - node_pt[0], apt[1] - node_pt[1])
        for apt in _anchor_points(p[0], p[1], w, h).values()
    ))

    # Pre-build placed bboxes for overlap check
    placed_bboxes = [
        _label_bbox_polygon(*rec["position"], *_label_wh(overflow_candidates[rec["label_id"]]))
        for rec in placed if rec["label_id"] != own_label_id
    ]

    for cx, cy in candidates:
        if not eroded.contains(Point(cx, cy)):
            continue

        bbox = _label_bbox_polygon(cx, cy, w, h)

        if not space.contains(bbox):
            continue

        if any(bbox.intersects(pb) for pb in placed_bboxes):
            continue

        # Try anchors closest to node first
        anchors = _anchor_points(cx, cy, w, h)
        sorted_anchors = sorted(
            anchors.items(),
            key=lambda kv: np.hypot(kv[1][0] - node_pos[0], kv[1][1] - node_pos[1])
        )
        for anchor_name, anchor_pt in sorted_anchors:
            line = LineString([anchor_pt, node_pos])
            if _binding_line_valid(line, own_node_id, own_label_id, G, placed, overflow_candidates):
                return cx, cy, anchor_name, anchor_pt

    return None


# ---------------------------------------------------------------------------
# Fitting check
# ---------------------------------------------------------------------------

def _compute_results(
    overflow_candidates: Dict[int, OverflowLabel],
    processed_faces: List[Tuple],
) -> Dict[int, List[int]]:
    """For each overflow label, return the face_ids where it fits."""
    results: Dict[int, List[int]] = {}
    for label_id, label in overflow_candidates.items():
        w, h = _label_wh(label)
        results[label_id] = [
            fid for fid, space in processed_faces
            if _fits(space, w, h)
        ]
    return results


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def place_inner_overflow_labels(
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
    results: Dict[int, List[int]],
    processed_faces: List[Tuple],
    centers: List[Tuple],
) -> List[dict]:
    """
    Greedy placement: always pick the largest remaining face, find the
    closest unplaced node that fits, place it as close to the node as
    possible, update remaining space, repeat.
    """
    space_map: Dict[int, Polygon] = dict(processed_faces)
    placed: List[dict] = []
    placed_label_ids: set = set()

    # face_id -> [label_ids that fit there]
    face_to_labels: Dict[int, List[int]] = {}
    for label_id, face_ids in results.items():
        for fid in face_ids:
            face_to_labels.setdefault(fid, []).append(label_id)

    while True:
        # Pick largest face with at least one unplaced fitting label
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
                    np.array(G.nodes[overflow_candidates[lid].node_id]["pos"])
                    - face_center
                )
            )
            chosen_face = face_id
            break

        if chosen_face is None:
            break

        label = overflow_candidates[chosen_label]
        w, h = _label_wh(label)
        node_pos = G.nodes[label.node_id]["pos"]
        space = space_map[chosen_face]

        result = _find_valid_position(
            space, w, h, node_pos, label.node_id, chosen_label,
            G, placed, overflow_candidates,
        )

        if result is None:
            # Can't place in this face — remove from candidates and retry
            results[chosen_label] = [f for f in results[chosen_label] if f != chosen_face]
            face_to_labels[chosen_face] = [
                lid for lid in face_to_labels[chosen_face] if lid != chosen_label
            ]
            continue

        cx, cy, anchor_name, anchor_pt = result

        record = {
            "label_id":     chosen_label,
            "face_id":      chosen_face,
            "position":     (cx, cy),
            "anchor":       anchor_name,
            "anchor_pt":    anchor_pt,
            "binding_line": LineString([anchor_pt, node_pos]),
        }

        placed.append(record)
        placed_label_ids.add(chosen_label)

        # Subtract placed bbox from this face's remaining space
        space_map[chosen_face] = space_map[chosen_face].difference(
            _label_bbox_polygon(cx, cy, w, h)
        )

    return placed


# ---------------------------------------------------------------------------
# Update overflow candidates with final positions
# ---------------------------------------------------------------------------

def _update_overflow_label_position(label: OverflowLabel, cx: float, cy: float):
    """Shift bbox and inner_bbox corners to a new center (cx, cy)."""
    bl, br, tr, tl = label.bbox_corners
    half_w = (br[0] - bl[0]) / 2
    half_h = (tl[1] - bl[1]) / 2

    ibl, ibr, itr, itl = label.inner_bbox_corners
    half_iw = (ibr[0] - ibl[0]) / 2
    half_ih = (itl[1] - ibl[1]) / 2

    label.center = (cx, cy)
    label.bbox_corners = (
        (cx - half_w,  cy - half_h),
        (cx + half_w,  cy - half_h),
        (cx + half_w,  cy + half_h),
        (cx - half_w,  cy + half_h),
    )
    label.inner_bbox_corners = (
        (cx - half_iw, cy - half_ih),
        (cx + half_iw, cy - half_ih),
        (cx + half_iw, cy + half_ih),
        (cx - half_iw, cy + half_ih),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def inner_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    bounded_faces: List[List],
    centers: List[Tuple],
) -> Dict[int, OverflowLabel]:

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

    # 3. Which faces does each overflow label fit in?
    results = _compute_results(overflow_candidates, processed_faces)

    # 4. Greedy inner placement
    placements = place_inner_overflow_labels(
        G, overflow_candidates, results, processed_faces, centers
    )

    # 5. Update overflow_candidates with final positions
    placement_by_id = {p["label_id"]: p for p in placements}
    for label_id, label in overflow_candidates.items():
        if label_id in placement_by_id:
            cx, cy = placement_by_id[label_id]["position"]
            _update_overflow_label_position(label, cx, cy)
            label.anchor = placement_by_id[label_id]["anchor"]

    return overflow_candidates