import numpy as np
import networkx as nx

from shapely import unary_union
from shapely.geometry import Polygon, LineString, Point
from typing import Dict, List, Tuple

from label import LabelCandidate, OverflowLabel

def _anchor_points(cx: float, cy: float, w: float, h: float) -> Dict[str, tuple]:
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

def place_overflow_labels(
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
    results: Dict[int, List[int]],           # label_id -> list of fitting face_ids
    result_list: List[int],                  # face_ids sorted largest first
    processed_faces: List[tuple],            # (original_id, remaining_space polygon)
    centers: List[tuple],                    # centers[face_id] = (x, y) of face
) -> List[dict]:

    space_lookup = dict(processed_faces)
    placed: List[dict] = []
    placed_label_ids: set = set()

    # Which labels fit in which face
    face_to_labels: Dict[int, List[int]] = {}
    for label_id, face_ids in results.items():
        for fid in face_ids:
            face_to_labels.setdefault(fid, []).append(label_id)

    # Walk faces largest-first
    for face_id in result_list:
        candidate_label_ids = [
            lid for lid in face_to_labels.get(face_id, [])
            if lid not in placed_label_ids
        ]
        if not candidate_label_ids:
            continue

        face_center = np.array(centers[face_id])

        # Pick the label whose node is closest to this face's center
        label_id = min(
            candidate_label_ids,
            key=lambda lid: np.linalg.norm(
                np.array(G.nodes[overflow_candidates[lid].node_id]["pos"]) - face_center
            )
        )

        label = overflow_candidates[label_id]
        space = space_lookup[face_id]

        centroid = space.centroid
        cx, cy = centroid.x, centroid.y

        bl, br, tr, tl = label.bbox_corners
        w = abs(br[0] - bl[0])
        h = abs(tl[1] - bl[1])

        node_pos = G.nodes[label.node_id]["pos"]
        anchors = _anchor_points(cx, cy, w, h)
        best_anchor_name, best_anchor_pt = min(
            anchors.items(),
            key=lambda kv: np.hypot(kv[1][0] - node_pos[0], kv[1][1] - node_pos[1])
        )

        placed.append({
            "label_id":     label_id,
            "face_id":      face_id,
            "position":     (cx, cy),
            "anchor":       best_anchor_name,
            "anchor_pt":    best_anchor_pt,
            "binding_line": LineString([best_anchor_pt, node_pos]),
        })

        placed_label_ids.add(label_id)

    return placed

def inner_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    bounded_faces: List[List],
    centers: List[Tuple]
) -> Dict[int, List[int]]:

    # 1. Build face polygons, sorted largest first
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

    # 2. Subtract already-placed label boxes from each face
    placed_union = unary_union([
        Polygon(cand.bbox_corners)
        for candidates in label_candidates.values()
        for cand in candidates
    ])

    processed_faces = [
        (d["original_id"], d["polygon"].difference(placed_union))
        for d in face_data
    ]

    # 3. For each overflow label, find faces where it geometrically fits
    results: Dict[int, List[int]] = {}

    for label_id, label in overflow_candidates.items():
        l_poly = Polygon(label.bbox_corners)
        minx, miny, maxx, maxy = l_poly.bounds
        w, h = maxx - minx, maxy - miny
        label_area = w * h

        fitting_face_ids = []

        for original_id, space in processed_faces:
            if space.is_empty:
                continue

            # Fast pre-flight: bounding box and area
            s_minx, s_miny, s_maxx, s_maxy = space.bounds
            if (
                w > (s_maxx - s_minx)
                or h > (s_maxy - s_miny)
                or label_area > space.area
            ):
                continue

            # Geometric fit: erode by half-extents on each axis.
            # A point surviving both erosions means a w×h rectangle fits there.
            eroded = space.buffer(-h / 2, join_style=2).buffer(
                -(w - h) / 2, join_style=2
            ) if w >= h else space.buffer(-w / 2, join_style=2).buffer(
                -(h - w) / 2, join_style=2
            )

            if not eroded.is_empty:
                fitting_face_ids.append(original_id)

        results[label_id] = fitting_face_ids

    # 4. Collect faces that fit at least one label, preserving largest-first order
    faces_with_fits = {f_id for ids in results.values() for f_id in ids}
    result_list = [oid for oid, _ in processed_faces if oid in faces_with_fits]

    print(place_overflow_labels(G, overflow_candidates, results, result_list, processed_faces, centers))
