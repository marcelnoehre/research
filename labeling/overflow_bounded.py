import numpy as np
import networkx as nx

from shapely import unary_union
from shapely.geometry import LineString, Point, Polygon
from typing import Dict, List, Optional, Tuple

from label import LabelCandidate, OverflowLabel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_BUFFER = 1.0

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _label_wh(label: OverflowLabel) -> Tuple[float, float]:
    """Tight bbox dimensions — used for anchor interpolation only."""
    bl, br, tr, tl = label.bbox_corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])


def _label_wh_expanded(label: OverflowLabel) -> Tuple[float, float]:
    """Expanded bbox dimensions — used for all collision checks."""
    bl, br, tr, tl = label.expanded_bbox_corners
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


def _anchor_point_from_bbox(
    anchor: str,
    bbox_corners: Tuple,
    inner_bbox_corners: Tuple,
    c_factor: float = 1.0,
    s_factor: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute the visual anchor point by interpolating between outer and inner bbox.
    Matches exactly how anchors are plotted.
    """
    outer = bbox_corners
    inner = inner_bbox_corners

    def get_pt(out_pt, inn_pt, factor):
        return (
            out_pt[0] + (inn_pt[0] - out_pt[0]) * factor,
            out_pt[1] + (inn_pt[1] - out_pt[1]) * factor,
        )

    anchor_lookup = {
        'bottom_left':  get_pt(outer[0], inner[0], c_factor),
        'bottom_right': get_pt(outer[1], inner[1], c_factor),
        'top_right':    get_pt(outer[2], inner[2], c_factor),
        'top_left':     get_pt(outer[3], inner[3], c_factor),
        'bottom': get_pt(((outer[0][0]+outer[1][0])/2, outer[0][1]), ((inner[0][0]+inner[1][0])/2, inner[1][1]), s_factor),
        'top':    get_pt(((outer[3][0]+outer[2][0])/2, outer[3][1]), ((inner[3][0]+inner[2][0])/2, inner[3][1]), s_factor),
        'left':   get_pt((outer[0][0], (outer[0][1]+outer[3][1])/2), (inner[0][0], (inner[0][1]+inner[3][1])/2), s_factor),
        'right':  get_pt((outer[1][0], (outer[1][1]+outer[2][1])/2), (inner[1][0], (inner[1][1]+inner[2][1])/2), s_factor),
    }
    return anchor_lookup[anchor]


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
    label_candidates: Dict[int, List[LabelCandidate]],
) -> bool:
    for rec in placed:
        if rec["label_id"] == own_label_id:
            continue
        cx, cy = rec["position"]
        w, h = _label_wh_expanded(overflow_candidates[rec["label_id"]])
        if line.intersects(_label_bbox_polygon(cx, cy, w, h)):
            return False
        if "binding_line" in rec and rec["binding_line"] is not None:
            if line.crosses(rec["binding_line"]):
                return False
    
    for candidates in label_candidates.values():
        for cand in candidates:
            cand_poly = Polygon(cand.bbox_corners).buffer(0)
            if line.intersects(cand_poly):
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
    label_candidates: Dict[int, List[LabelCandidate]],
) -> Optional[Tuple[float, float, str, Tuple]]:
    """
    Find the valid center position inside the eroded space that is closest
    to node_pos, with a clear binding line. Returns (cx, cy, anchor, anchor_pt)
    or None.
    """
    label = overflow_candidates[own_label_id]

    # Expanded dims for collision, tight dims for anchor interpolation
    ew, eh = _label_wh_expanded(label)
    hw, hh = w / 2, h / 2
    half_ew, half_eh = ew / 2, eh / 2

    # Inner bbox half-dimensions for anchor interpolation
    ibl, ibr, itr, itl = label.inner_bbox_corners
    half_iw = (ibr[0] - ibl[0]) / 2
    half_ih = (itr[1] - ibr[1]) / 2

    eroded = _eroded_space(space, ew, eh)
    if eroded.is_empty:
        return None

    node_pt = np.array(node_pos)
    node_point = Point(node_pos)

    # Seed candidates from 8 anchor offsets around node
    anchor_offsets = [
        (0,    -hh), (0,    hh),
        (-hw,   0),  (hw,   0),
        (-hw,  -hh), (hw,  -hh),
        (-hw,   hh), (hw,   hh),
    ]
    candidates = [
        (float(node_pt[0] - dx), float(node_pt[1] - dy))
        for dx, dy in anchor_offsets
    ]

    if eroded.contains(node_point):
        candidates.append((float(node_pt[0]), float(node_pt[1])))
    else:
        nearest_pt = eroded.boundary.interpolate(eroded.boundary.project(node_point))
        candidates.append((nearest_pt.x, nearest_pt.y))

    # Fine grid fallback
    minx, miny, maxx, maxy = eroded.bounds
    step = min(ew, eh) / 8
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if eroded.contains(Point(x, y)):
                candidates.append((x, y))
            y += step
        x += step

    # Sort by closest visual anchor to node
    def _min_anchor_dist(p):
        cx_, cy_ = p
        cand_bbox = (
            (cx_ - hw,  cy_ - hh), (cx_ + hw,  cy_ - hh),
            (cx_ + hw,  cy_ + hh), (cx_ - hw,  cy_ + hh),
        )
        cand_inner = (
            (cx_ - half_iw, cy_ - half_ih), (cx_ + half_iw, cy_ - half_ih),
            (cx_ + half_iw, cy_ + half_ih), (cx_ - half_iw, cy_ + half_ih),
        )
        return min(
            np.hypot(
                _anchor_point_from_bbox(name, cand_bbox, cand_inner)[0] - node_pt[0],
                _anchor_point_from_bbox(name, cand_bbox, cand_inner)[1] - node_pt[1],
            )
            for name in _anchor_points(cx_, cy_, w, h)
        )

    candidates.sort(key=_min_anchor_dist)

    # Placed expanded bboxes for overlap check
    placed_bboxes = [
        _label_bbox_polygon(*rec["position"], *_label_wh_expanded(overflow_candidates[rec["label_id"]]))
        for rec in placed if rec["label_id"] != own_label_id
    ]

    for cx, cy in candidates:
        if not eroded.contains(Point(cx, cy)):
            continue

        expanded_bbox = _label_bbox_polygon(cx, cy, ew, eh)

        if not space.contains(expanded_bbox):
            continue
        if any(expanded_bbox.intersects(pb) for pb in placed_bboxes):
            continue
        if any(
            expanded_bbox.contains(Point(data["pos"]))
            for node_id, data in G.nodes(data=True)
            if isinstance(node_id, int) and node_id != own_node_id
        ):
            continue

        # Anchor interpolation uses tight bbox corners
        candidate_bbox = (
            (cx - hw,  cy - hh), (cx + hw,  cy - hh),
            (cx + hw,  cy + hh), (cx - hw,  cy + hh),
        )
        candidate_inner = (
            (cx - half_iw, cy - half_ih), (cx + half_iw, cy - half_ih),
            (cx + half_iw, cy + half_ih), (cx - half_iw, cy + half_ih),
        )

        sorted_anchors = sorted(
            _anchor_points(cx, cy, w, h).keys(),
            key=lambda name: np.hypot(
                _anchor_point_from_bbox(name, candidate_bbox, candidate_inner)[0] - node_pt[0],
                _anchor_point_from_bbox(name, candidate_bbox, candidate_inner)[1] - node_pt[1],
            )
        )

        for anchor_name in sorted_anchors:
            anchor_pt = _anchor_point_from_bbox(anchor_name, candidate_bbox, candidate_inner)
            line = LineString([anchor_pt, node_pos])
            if _binding_line_valid(line, own_node_id, own_label_id, G, placed, overflow_candidates, label_candidates):
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
        ew, eh = _label_wh_expanded(label)
        results[label_id] = [
            fid for fid, space in processed_faces
            if _fits(space, ew, eh)
        ]
    return results


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def place_inner_overflow_labels(
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
    label_candidates: Dict[int, List[LabelCandidate]],
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

    face_to_labels: Dict[int, List[int]] = {}
    for label_id, face_ids in results.items():
        for fid in face_ids:
            face_to_labels.setdefault(fid, []).append(label_id)

    while True:
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
                and _fits(space, *_label_wh_expanded(overflow_candidates[lid]))
            ]
            if not candidates:
                continue

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
            G, placed, overflow_candidates, label_candidates
        )

        if result is None:
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

        # Subtract expanded bbox from remaining space
        ew, eh = _label_wh_expanded(overflow_candidates[chosen_label])
        space_map[chosen_face] = space_map[chosen_face].difference(
            _label_bbox_polygon(cx, cy, ew, eh)
        )

    return placed


# ---------------------------------------------------------------------------
# Update overflow candidates with final positions
# ---------------------------------------------------------------------------

def _update_overflow_label_position(label: OverflowLabel, cx: float, cy: float, anchor: str):
    """Shift bbox, inner_bbox and expanded_bbox corners to a new center (cx, cy)."""
    scale = (1 / np.sqrt(2)) if '_' in anchor else 1.0

    # 2. Get the INK dimensions (these never change)
    ibl, ibr, itr, itl = label.inner_bbox_corners
    half_iw = (ibr[0] - ibl[0]) / 2.0
    half_ih = (itl[1] - ibl[1]) / 2.0

    # 3. Get the original PADDING from the existing expanded bbox
    # (Expanded Width - Ink Width) / 2 = Original Padding in graph units
    ebl, ebr, etr, etl = label.expanded_bbox_corners
    orig_px = ((ebr[0] - ebl[0]) / 2.0) - half_iw
    orig_py = ((etl[1] - ebl[1]) / 2.0) - half_ih

    # 4. Apply the scale to the padding
    adj_px = orig_px * scale
    adj_py = orig_py * scale

    # 5. Calculate new half-extents for Visual (Expanded) and Collision (BBox)
    # We maintain your 2/3 ratio for the collision bbox
    half_ew = half_iw + adj_px
    half_eh = half_ih + adj_py
    
    half_w = half_iw + (adj_px * (2/3))
    half_h = half_ih + (adj_py * (2/3))

    # 6. Apply to the label
    label.center = (cx, cy)
    
    # Visual Box (Expanded)
    label.expanded_bbox_corners = (
        (cx - half_ew,  cy - half_eh),
        (cx + half_ew,  cy - half_eh),
        (cx + half_ew,  cy + half_eh),
        (cx - half_ew,  cy + half_eh),
    )
    
    # Collision Box (Tight)
    label.bbox_corners = (
        (cx - half_w,  cy - half_h),
        (cx + half_w,  cy - half_h),
        (cx + half_w,  cy + half_h),
        (cx - half_w,  cy + half_h),
    )
    
    # Ink Box (Shifted center, dimensions unchanged)
    label.inner_bbox_corners = (
        (cx - half_iw, cy - half_ih),
        (cx + half_iw, cy - half_ih),
        (cx + half_iw, cy + half_ih),
        (cx - half_iw, cy + half_ih),
    )

    label.anchor = anchor


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

    # 2. Subtract already-placed label expanded bboxes from each face
    placed_union = unary_union([
        Polygon(cand.expanded_bbox_corners)
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
        G, overflow_candidates, label_candidates, results, processed_faces, centers
    )

    # 5. Update overflow_candidates with final positions
    placement_by_id = {p["label_id"]: p for p in placements}
    for label_id, label in overflow_candidates.items():
        if label_id in placement_by_id:
            cx, cy = placement_by_id[label_id]["position"]
            anchor = placement_by_id[label_id]["anchor"]
            _update_overflow_label_position(label, cx, cy, anchor)

    return overflow_candidates