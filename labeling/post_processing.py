import copy
import numpy as np
import networkx as nx

from typing import Dict, List

from shapely import LineString, Point, Polygon

from label import LabelCandidate, OverflowLabel
from overflow_bounded import _anchor_points, _label_wh, update_overflow_label_position

def _distance_to_drawing(label_id, overflow_candidates, outer_boundary):
        ol = overflow_candidates[label_id]
        ax, ay = _get_anchor_pt(ol)
        return Point(ax, ay).distance(outer_boundary)

def _get_anchor_pt(candidate: OverflowLabel):
    tl, tr, br, bl = candidate.bbox_corners
    
    anchors = {
        'top':          ((tl[0] + tr[0]) / 2, tl[1]),
        'bottom':       ((bl[0] + br[0]) / 2, bl[1]),
        'left':         (tl[0], (tl[1] + bl[1]) / 2),
        'right':        (tr[0], (tr[1] + br[1]) / 2),
        'top_left':     tl,
        'top_right':    tr,
        'bottom_left':  bl,
        'bottom_right': br,
        'overflow':     candidate.center
    }
    return anchors[candidate.anchor]

def _tmp_candidate_valid(
        tmp_poly: Polygon, 
        tmp_poly_tight: Polygon, 
        label_candidates: Dict[int, List[LabelCandidate]], 
        overflow_candidates: Dict[int, OverflowLabel], 
        node_id: int, 
        G: nx.Graph, 
        outer_polygon: Polygon
    ) -> bool:
    '''
    Check if the temporary candidate is valid by ensuring it does not intersect with:
    1. Existing label candidates
    2. Other overflow candidates (except itself)
    3. Binders of other overflow candidates (except itself)
    4. Label overlaps with the drawing (outer polygon)
    '''
    # overlaps with existing label
    if any(tmp_poly.intersects(Polygon(lc[0].bbox_corners)) for lc in label_candidates.values() if lc):
        return False
    
    # overlaps another overflow label
    if any(tmp_poly.intersects(Polygon(oc.expanded_bbox_corners)) for oid, oc in overflow_candidates.items() if oid != node_id):
        return False

    # overlaps with a binder of another overflow label 
    if any(
        tmp_poly_tight.intersects(LineString([G.nodes[oid]["pos"], _get_anchor_pt(oc)])) 
        for oid, oc in overflow_candidates.items() if oid != node_id
    ):
        return False
    
    if tmp_poly.intersects(outer_polygon):
        return False
    
    return True


def adjust_anchors(
        G: nx.Graph, 
        label_candidates: Dict[int, List[LabelCandidate]], 
        overflow_candidates: Dict[int, OverflowLabel], 
        unbounded_overflow_labels: List[int],
        outer_nodes: List
    ):
    outer_positions = [G.nodes[n]['pos'] for n in outer_nodes]
    outer_polygon   = Polygon(outer_positions)
    outer_boundary = outer_polygon.boundary

    # closer to the drawing first
    sorted_unbounded_overflow_labels = sorted(
        unbounded_overflow_labels, 
        key=lambda nid: _distance_to_drawing(nid, overflow_candidates, outer_boundary)
    )

    for node_id in sorted_unbounded_overflow_labels:
        ol = overflow_candidates[node_id]
        cx, cy = ol.center
        tw, th = _label_wh(ol)
        anchors = _anchor_points(cx, cy, tw, th)
        label_center = np.array([cx, cy])
        node_pos = G.nodes[ol.node_id]["pos"]
        node_pt = np.array(node_pos)
        vec = node_pt - label_center
        dist_to_node = np.linalg.norm(vec)
        unit = vec / dist_to_node if dist_to_node > 0 else vec
        scored_anchors = []
        for name, pt in anchors.items():
            va = np.array(pt) - label_center
            na = np.linalg.norm(va)
            align = float(np.dot(unit, va / na)) if na > 0 else -1.0
            scored_anchors.append((align, name, pt))
        scored_anchors.sort(reverse=True)

        for _, name, pt in scored_anchors:
            # best anchor chosen
            if ol.anchor == name:
                break
            
            # translate the label, while fixing the anchor point
            old_anchor_pt = anchors[ol.anchor]
            new_anchor_offset = np.array(pt) - label_center
            new_center = np.array(old_anchor_pt) - new_anchor_offset

            # check if we can translate the label to this anchor without
            tmp_candidate = copy.deepcopy(ol)
            update_overflow_label_position(tmp_candidate, new_center[0], new_center[1], name)

            # validity of new label
            tmp_poly = Polygon(tmp_candidate.expanded_bbox_corners)
            tmp_poly_tight = Polygon(tmp_candidate.bbox_corners)
            if not _tmp_candidate_valid(tmp_poly, tmp_poly_tight, label_candidates, overflow_candidates, node_id, G, outer_polygon):
                continue

            print(f"Adjusting overflow label for node {node_id} from {ol.anchor} to {name}")
            overflow_candidates[node_id] = tmp_candidate
            break

    return overflow_candidates

def adjust_binders(
        G: nx.Graph, 
        label_candidates: Dict[int, List[LabelCandidate]], 
        overflow_candidates: Dict[int, OverflowLabel], 
        unbounded_overflow_labels: List[int],
        outer_nodes: List,
        step_size: float = 0.1,
        min_binder_length: float = 0.5
    ):
    outer_positions = [G.nodes[n]['pos'] for n in outer_nodes]
    outer_polygon   = Polygon(outer_positions)
    outer_boundary = outer_polygon.boundary

    # closer to the drawing first
    sorted_unbounded_overflow_labels = sorted(
        unbounded_overflow_labels, 
        key=lambda nid: _distance_to_drawing(nid, overflow_candidates, outer_boundary)
    )

    for node_id in sorted_unbounded_overflow_labels:
        ol = overflow_candidates[node_id]
        node_pos = G.nodes[node_id]['pos']
        nx, ny = node_pos

        tmp_candidate = copy.deepcopy(ol)

        while True:
            ax, ay = _get_anchor_pt(tmp_candidate)

            # vector from anchor to node
            vx = nx - ax
            vy = ny - ay
            dist_to_node = np.sqrt(vx**2 + vy**2)

            if dist_to_node <= step_size:
                break

            # normalized
            ux, uy = vx / dist_to_node, vy / dist_to_node

            new_ax = ax + ux * step_size
            new_ay = ay + uy * step_size

            new_cx = tmp_candidate.center[0] + ux * step_size
            new_cy = tmp_candidate.center[1] + uy * step_size
            
            update_overflow_label_position(tmp_candidate, new_cx, new_cy, ol.anchor)

            # validity of new label
            dist_to_drawing = Point(new_ax, new_ay).distance(outer_boundary)
            if dist_to_drawing < min_binder_length:
                print(f"Binder for node {node_id} is too close to the drawing boundary ({dist_to_drawing}).")
                break

            tmp_poly = Polygon(tmp_candidate.expanded_bbox_corners)
            tmp_poly_tight = Polygon(tmp_candidate.bbox_corners)
            if not _tmp_candidate_valid(tmp_poly, tmp_poly_tight, label_candidates, overflow_candidates, node_id, G, outer_polygon):
                print(f"Adjusted binder for node {node_id} yields invalid position.")
                break

            print(f"Pulling {node_id} closer to the drawing")
            overflow_candidates[node_id] = tmp_candidate

    return overflow_candidates