import numpy as np
import networkx as nx

from shapely import unary_union
from shapely.geometry import LineString, Point, Polygon
from typing import Dict, List, Optional, Tuple

from label import LabelCandidate, OverflowLabel

################################################################################
# Geometry helpers
################################################################################

def _label_wh(label: OverflowLabel) -> Tuple[float, float]:
    '''
    Tight bbox dimensions for anchor interpolation only.
    '''
    bl, br, _, tl = label.bbox_corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])

def _label_wh_expanded(label: OverflowLabel) -> Tuple[float, float]:
    '''
    Expanded bbox dimensions for collision checks.
    '''
    bl, br, _, tl = label.expanded_bbox_corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])

def _label_bbox_polygon(cx: float, cy: float, w: float, h: float) -> Polygon:
    '''
    Create a polygon for the label bbox centered at (cx, cy) with width w and height h.
    '''
    hw, hh = w / 2, h / 2
    return Polygon([
        (cx - hw, cy - hh), (cx + hw, cy - hh),
        (cx + hw, cy + hh), (cx - hw, cy + hh),
    ])

def _anchor_points(cx: float, cy: float, w: float, h: float) -> Dict[str, Tuple[float, float]]:
    '''
    Compute the anchor points for a label centered at (cx, cy) with width w and height h.
    '''
    hw, hh = w / 2, h / 2
    return {
        'top':          (cx,      cy + hh),
        'bottom':       (cx,      cy - hh),
        'left':         (cx - hw, cy     ),
        'right':        (cx + hw, cy     ),
        'top_left':     (cx - hw, cy + hh),
        'top_right':    (cx + hw, cy + hh),
        'bottom_left':  (cx - hw, cy - hh),
        'bottom_right': (cx + hw, cy - hh),
    }

def _eroded_space(space: Polygon, w: float, h: float) -> Polygon:
    '''
    Return the set of valid center positions for a label inside space
    '''
    smaller, larger = min(w, h), max(w, h)
    eroded = space.buffer(-smaller / 2, join_style=2)
    if eroded.is_empty:
        return eroded
    return eroded.buffer(-(larger - smaller) / 2, join_style=2)


def _fits(space: Polygon, w: float, h: float) -> bool:
    '''
    Check wether label fits into polygon.
    '''
    if space.is_empty:
        return False
    s_minx, s_miny, s_maxx, s_maxy = space.bounds
    if w > (s_maxx - s_minx) or h > (s_maxy - s_miny) or w * h > space.area:
        return False
    return not _eroded_space(space, w, h).is_empty

def _fitting_faces(
    overflow_candidates: Dict[int, OverflowLabel],
    processed_faces: List[Tuple],
) -> Dict[int, List[int]]:
    '''
    For each overflow label, return the face_ids where it fits.
    '''
    results: Dict[int, List[int]] = {}
    for label_id, label in overflow_candidates.items():
        ew, eh = _label_wh_expanded(label)
        results[label_id] = [
            fid for fid, space in processed_faces
            if _fits(space, ew, eh)
        ]
    return results

def binding_line_valid(
    line: LineString,
    own_node_id: int,
    own_label_id: int,
    G: nx.Graph,
    placed: List[dict],
    overflow_candidates: Dict[int, OverflowLabel],
    label_candidates: Dict[int, List[LabelCandidate]],
) -> bool:
    '''
    Validate wether a binder is valid.
    '''
    for rec in placed:
        if rec['label_id'] == own_label_id:
            continue
        cx, cy = rec['position']
        w, h = _label_wh_expanded(overflow_candidates[rec['label_id']])

        # binder intersects already placed overflow label
        if line.intersects(_label_bbox_polygon(cx, cy, w, h)):
            return False
        
        # binder conflicts with another binder
        if 'binding_line' in rec and rec['binding_line'] is not None:
            if line.crosses(rec['binding_line']):
                return False
    
    for candidates in label_candidates.values():
        for cand in candidates:
            cand_poly = Polygon(cand.bbox_corners).buffer(0)

            # binder intersects bbox of a label
            if line.intersects(cand_poly):
                return False

    for node_id, data in G.nodes(data=True):
        if node_id == own_node_id:
            continue
        if not isinstance(node_id, int):
            continue
        
        # binder intersects with a node
        if line.distance(Point(data['pos'])) < 1e-6:
            return False

    # valid - no conflict found
    return True

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
    '''
    Find the valid center position inside the eroded space that is closest
    to node_pos with a clear binding line.
    
    Returns
    -------
    (cx, cy, anchor, anchor_pt) or None.
    '''
    label = overflow_candidates[own_label_id]

    # expanded bbox for collision checks, tight bbox for anchor interpolation
    ew, eh = _label_wh_expanded(label)
    hw, hh = w / 2, h / 2

    # eroded space
    eroded = _eroded_space(space, ew, eh)
    if eroded.is_empty:
        return None

    # node for overflow label
    node_pt = np.array(node_pos)
    node_point = Point(node_pos)

    # anchors
    corner_multiplier = 1 / np.sqrt(2)
    side_multiplier = 1 / corner_multiplier
    anchor_offsets = [
        (0, -hh * side_multiplier), (0, hh * side_multiplier),
        (-hw * side_multiplier,   0), (hw * side_multiplier, 0),
        (-hw * corner_multiplier, -hh), (hw * corner_multiplier, -hh),
        (-hw * corner_multiplier, hh), (hw * corner_multiplier, hh),
    ]

    # one candidate per anchor
    candidates = [
        (float(node_pt[0] - dx), float(node_pt[1] - dy))
        for dx, dy in anchor_offsets
    ]

    # place as close as possible to original node
    if eroded.contains(node_point):
        candidates.append((float(node_pt[0]), float(node_pt[1])))
    else:
        nearest_pt = eroded.boundary.interpolate(eroded.boundary.project(node_point))
        candidates.append((nearest_pt.x, nearest_pt.y))

    # fine grid fallback
    minx, miny, maxx, maxy = eroded.bounds
    step = min(ew, eh) / 16
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if eroded.contains(Point(x, y)):
                candidates.append((x, y))
            y += step
        x += step

    def _min_anchor_dist(p):
        '''
        sort anchors by distance to node
        '''
        cx_, cy_ = p
        anchors = _anchor_points(cx_, cy_, w, h)
        return min(
            np.hypot(coords[0] - node_pt[0], coords[1] - node_pt[1])
            for coords in anchors.values()
        )

    # sort by anchor closest to node
    candidates.sort(key=_min_anchor_dist)

    # already placed overflow labels
    placed_bboxes = [
        _label_bbox_polygon(*rec['position'], *_label_wh_expanded(overflow_candidates[rec['label_id']]))
        for rec in placed if rec['label_id'] != own_label_id
    ]

    for cx, cy in candidates:
        # center not in eroded space
        if not eroded.contains(Point(cx, cy)):
            continue

        expanded_bbox = _label_bbox_polygon(cx, cy, ew, eh)

        # expanded bbox does not fit into the eroded space
        if not space.contains(expanded_bbox):
            continue
        
        # expanded bbox overlaps with another overflow label
        if any(expanded_bbox.intersects(pb) for pb in placed_bboxes):
            continue
        
        # expanded bbox overlaps with node
        if any(
            expanded_bbox.contains(Point(data['pos']))
            for node_id, data in G.nodes(data=True)
            if isinstance(node_id, int) and node_id != own_node_id
        ):
            continue

        # anchor
        # sort anchor by distance to the respective node
        real_anchors = _anchor_points(cx, cy, w, h)
        sorted_anchors = sorted(
            real_anchors.keys(),
            key=lambda name: np.hypot(
                real_anchors[name][0] - node_pt[0],
                real_anchors[name][1] - node_pt[1]
            )
        )

        # closest valid anchor
        for anchor_name in sorted_anchors:
            anchor_pt = real_anchors[anchor_name]
            line = LineString([anchor_pt, node_pos])
            
            # keep if valid
            if binding_line_valid(line, own_node_id, own_label_id, G, placed, overflow_candidates, label_candidates):
                return cx, cy, anchor_name, anchor_pt

    # no valid position found
    return None

def update_overflow_label_position(label: OverflowLabel, cx: float, cy: float, anchor: str):
    '''
    Shift bbox, inner_bbox and expanded_bbox corners to a new center (cx, cy).
    '''
    # s = (1.0 / corner_multiplier)
    # padding = ink_h_mm * 0.75


    # ink size
    ibl, ibr, _, itl = label.inner_bbox_corners
    half_iw = (ibr[0] - ibl[0]) / 2.0
    half_ih = (itl[1] - ibl[1]) / 2.0

    # original padding from expanded bbox
    ebl, ebr, _, etl = label.expanded_bbox_corners
    orig_px = ((ebr[0] - ebl[0]) / 2.0) - half_iw
    orig_py = ((etl[1] - ebl[1]) / 2.0) - half_ih

    # scaled half-extents
    half_ew = half_iw + orig_px
    half_eh = half_ih + orig_py

    # anchor placement
    is_top    = 'top' in anchor
    is_bottom = 'bottom' in anchor
    is_left   = 'left' in anchor
    is_right  = 'right' in anchor

    corner_multiplier = 1 / np.sqrt(2)
    padding = half_ih

    p_top    = (padding * corner_multiplier) if is_top else padding
    p_bottom = (padding * corner_multiplier) if is_bottom else padding
    p_left   = (padding * corner_multiplier) if is_left else padding
    p_right  = (padding * corner_multiplier) if is_right else padding

    # update label
    label.center = (cx, cy)
    label.anchor = anchor
    label.inner_bbox_corners = (
        (cx - half_iw, cy - half_ih),
        (cx + half_iw, cy - half_ih),
        (cx + half_iw, cy + half_ih),
        (cx - half_iw, cy + half_ih),
    )
    label.expanded_bbox_corners = (
        (cx - half_ew,  cy - half_eh),
        (cx + half_ew,  cy - half_eh),
        (cx + half_ew,  cy + half_eh),
        (cx - half_ew,  cy + half_eh),
    )
    label.bbox_corners = (
        (cx - half_iw - p_left,  cy - half_ih - p_bottom),
        (cx + half_iw + p_right,  cy - half_ih - p_bottom),
        (cx + half_iw + p_right,  cy + half_ih + p_top),
        (cx - half_iw - p_left,  cy + half_ih + p_top),
    )

def place_inner_overflow_labels(
    G: nx.Graph,
    overflow_candidates: Dict[int, OverflowLabel],
    label_candidates: Dict[int, List[LabelCandidate]],
    results: Dict[int, List[int]],
    processed_faces: List[Tuple],
    centers: List[Tuple],
) -> List[dict]:
    '''
    pick the largest remaining face and find the closest unplaced node that fits
    place it as close to the node as possible, update remaining space, repeat.
    '''
    space_map: Dict[int, Polygon] = dict(processed_faces)
    placed: List[dict] = []
    placed_label_ids: set = set()

    face_to_labels: Dict[int, List[int]] = {}
    for label_id, face_ids in results.items():
        for fid in face_ids:
            face_to_labels.setdefault(fid, []).append(label_id)

    while True:
        # sort by remaining size in face
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

            # labels that possibly fit into the face
            candidates = [
                lid for lid in face_to_labels.get(face_id, [])
                if lid not in placed_label_ids
                and _fits(space, *_label_wh_expanded(overflow_candidates[lid]))
            ]

            # skip face if no label fits
            if not candidates:
                continue

            # node closest to the center of remaining space in the face
            chosen_label = min(
                candidates,
                key=lambda lid: np.linalg.norm(
                    np.array(G.nodes[overflow_candidates[lid].node_id]["pos"])
                    - face_center
                )
            )

            chosen_face = face_id
            break

        # no label fits into a face - stop searching
        if chosen_face is None:
            break

        label = overflow_candidates[chosen_label]
        w, h = _label_wh(label)
        node_pos = G.nodes[label.node_id]["pos"]
        space = space_map[chosen_face]

        # find the best valid position for the chosen label
        result = _find_valid_position(
            space, w, h, node_pos, label.node_id, chosen_label,
            G, placed, overflow_candidates, label_candidates
        )

        # if no valid position found block the combination of face and label
        if result is None:
            results[chosen_label] = [f for f in results[chosen_label] if f != chosen_face]
            face_to_labels[chosen_face] = [
                lid for lid in face_to_labels[chosen_face] if lid != chosen_label
            ]
            continue

        # valid position found
        cx, cy, anchor_name, anchor_pt = result

        record = {
            'label_id':     chosen_label,
            'face_id':      chosen_face,
            'position':     (cx, cy),
            'anchor':       anchor_name,
            'anchor_pt':    anchor_pt,
            'binding_line': LineString([anchor_pt, node_pos]),
        }

        # track placed labels
        placed.append(record)
        placed_label_ids.add(chosen_label)

        # subtract expanded bbox from remaining space
        ew, eh = _label_wh_expanded(overflow_candidates[chosen_label])
        space_map[chosen_face] = space_map[chosen_face].difference(
            _label_bbox_polygon(cx, cy, ew, eh)
        )

    return placed

def inner_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    bounded_faces: List[List],
    centers: List[Tuple],
) -> Dict[int, OverflowLabel]:

    # face polygons sorted by size
    face_data = sorted(
        [
            {
                'original_id': idx,
                'polygon': Polygon([G.nodes[n]['pos'] for n in face]),
            }
            for idx, face in enumerate(bounded_faces)
        ],
        key=lambda x: x['polygon'].area,
        reverse=True,
    )

    # subtract expanded bboxes of placed labels from each face
    placed_union = unary_union([
        Polygon(cand.expanded_bbox_corners)
        for candidates in label_candidates.values()
        for cand in candidates
    ])

    # already processed faces
    processed_faces = [
        (d['original_id'], d['polygon'].difference(placed_union))
        for d in face_data
    ]

    # which faces does each overflow label fit in
    results = _fitting_faces(overflow_candidates, processed_faces)

    # inner placement
    placements = place_inner_overflow_labels(
        G, overflow_candidates, label_candidates, results, processed_faces, centers
    )

    # update overflow candidates with final positions
    placement_by_id = {p['label_id']: p for p in placements}
    for label_id, label in overflow_candidates.items():
        if label_id in placement_by_id:
            cx, cy = placement_by_id[label_id]['position']
            anchor = placement_by_id[label_id]['anchor']
            update_overflow_label_position(label, cx, cy, anchor)

    return overflow_candidates