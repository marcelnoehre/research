"""
placement.py
------------
For each concept node, computes 4 candidate label positions using a
fixed-position approach.

Each candidate is defined by anchoring one corner of the label's outer
bounding box to the node's position. The 4 corners are:
    top-left, top-right, bottom-left, bottom-right

All coordinates are in normalised graph units. Conversion to mm is done
via the mm_per_unit factor derived from the physical drawing height.

Requires normalize_positions() to have been called on G first.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import networkx as nx
from shapely.geometry import LineString, box

from label import measure_ink_mm


@dataclass
class LabelCandidate:
    """
    One candidate placement for a node label.

    Attributes
    ----------
    anchor            : which corner of the outer bbox is placed at the node position
    bbox_corners      : outer bbox (BL, BR, TR, TL) in graph units
    inner_bbox_corners: inner (ink) bbox (BL, BR, TR, TL) in graph units
    center            : center of the outer bbox in graph units
    """
    anchor: str
    bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]         # outer: BL, BR, TR, TL
    inner_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]   # inner: BL, BR, TR, TL
    center: Tuple[float, float]


def compute_label_candidates(
        G: nx.Graph,
        concepts: List[int],
        label_text: str,
        physical_height_mm: float,
        padding_x_mm: float = 3.0,
        padding_y_mm: float = 2.0,
        fontsize_pt: float = 10.0,
) -> Dict[int, List[LabelCandidate]]:
    """
    For every concept node, return 4 candidate label placements.

    Each placement anchors one corner of the label's outer bounding box
    to the node's (normalised) position.

    Parameters
    ----------
    G                  : graph with normalised 'pos' attributes
    concepts           : list of concept node ids
    label_text         : the label string (used to measure ink size)
    physical_height_mm : actual drawing height in mm (sets the unit scale)
    padding_x_mm       : horizontal padding around the ink box
    padding_y_mm       : vertical padding around the ink box
    fontsize_pt        : font size in points

    Returns
    -------
    dict mapping concept node id → list of 4 LabelCandidate objects
    """
    if 'normalized_height' not in G.graph:
        raise ValueError("Call normalize_positions(G) before compute_label_candidates().")

    mm_per_unit = physical_height_mm / G.graph['normalized_height']
    units_per_mm = 1.0 / mm_per_unit

    # Measure label in mm, convert to graph units
    ink_w_mm, ink_h_mm = measure_ink_mm(label_text, fontsize_pt)

    outer_w_mm = ink_w_mm + 2 * padding_x_mm
    outer_h_mm = ink_h_mm + 2 * padding_y_mm

    half_w  = (outer_w_mm * units_per_mm) / 2.0
    half_h  = (outer_h_mm * units_per_mm) / 2.0
    # Inner half-extents (ink only, no padding)
    half_iw = (ink_w_mm  * units_per_mm) / 2.0
    half_ih = (ink_h_mm  * units_per_mm) / 2.0

    anchor_offsets: Dict[str, Tuple[float, float]] = {
        'top_left':     ( half_w, -half_h),
        'top_right':    (-half_w, -half_h),
        'bottom_left':  ( half_w,  half_h),
        'bottom_right': (-half_w,  half_h),
    }

    candidates: Dict[int, List[LabelCandidate]] = {}

    for node in concepts:
        node_x, node_y = G.nodes[node]['pos']

        node_candidates: List[LabelCandidate] = []
        for anchor, (dx, dy) in anchor_offsets.items():
            cx = node_x + dx
            cy = node_y + dy

            bl = (cx - half_w,  cy - half_h)
            br = (cx + half_w,  cy - half_h)
            tr = (cx + half_w,  cy + half_h)
            tl = (cx - half_w,  cy + half_h)

            ibl = (cx - half_iw, cy - half_ih)
            ibr = (cx + half_iw, cy - half_ih)
            itr = (cx + half_iw, cy + half_ih)
            itl = (cx - half_iw, cy + half_ih)

            node_candidates.append(LabelCandidate(
                anchor=anchor,
                bbox_corners=(bl, br, tr, tl),
                inner_bbox_corners=(ibl, ibr, itr, itl),
                center=(cx, cy),
            ))

        candidates[node] = node_candidates

    return candidates


def _face_edges(face: List) -> Set[frozenset]:
    """Return the set of undirected edges for a face."""
    n = len(face)
    return {frozenset((face[i], face[(i + 1) % n])) for i in range(n)}


def _node_to_face_edges(
        node: int,
        bounded_faces: List[List],
        outer_nodes: List,
) -> List[Tuple]:
    """
    Collect all unique edges from faces that contain `node`,
    including the outer face. Returns a list of (u, v) tuples.
    """
    all_faces = bounded_faces + [outer_nodes]
    seen: Set[frozenset] = set()
    edges = []
    for face in all_faces:
        if node not in face:
            continue
        for e in _face_edges(face):
            if e not in seen:
                seen.add(e)
                u, v = tuple(e)
                edges.append((u, v))
    return edges


def filter_candidates_by_edges(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        bounded_faces: List[List],
        outer_nodes: List,
) -> Dict[int, List[LabelCandidate]]:
    """
    Remove candidates whose inner bounding box (ink area, no padding)
    is crossed by any edge of the faces the node belongs to,
    excluding edges directly incident to the node itself.

    Parameters
    ----------
    G             : graph with 'pos' attributes on every node
    candidates    : output of compute_label_candidates
    bounded_faces : bounded face node-lists from extract_faces
    outer_nodes   : outer face node-list from extract_faces

    Returns
    -------
    dict with same keys, each value being the surviving candidates
    """
    filtered: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        # All edges of incident faces, excluding those touching the node itself
        incident_edges = _node_to_face_edges(node, bounded_faces, outer_nodes)
        lines = [
            LineString([G.nodes[u]['pos'], G.nodes[v]['pos']])
            for u, v in incident_edges
        ]

        surviving = []
        for candidate in node_candidates:
            ibl, ibr, itr, itl = candidate.inner_bbox_corners
            inner_shape = box(ibl[0], ibl[1], itr[0], itr[1])
            if not any(inner_shape.intersects(line) for line in lines):
                surviving.append(candidate)

        filtered[node] = surviving

    return filtered