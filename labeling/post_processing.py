import copy
import numpy as np
import networkx as nx

from typing import Dict, List

from shapely import LineString, Polygon

from label import LabelCandidate, OverflowLabel
from overflow_bounded import _anchor_points, _label_wh, update_overflow_label_position

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

def adjust_anchors(
        G: nx.Graph, 
        label_candidates: Dict[int, List[LabelCandidate]], 
        overflow_candidates: Dict[int, OverflowLabel], 
        unbounded_overflow_labels: List[int]
    ):
    for node_id in unbounded_overflow_labels:
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

            # overlaps with existing label
            if any(tmp_poly.intersects(Polygon(lc[0].bbox_corners)) for lc in label_candidates.values() if lc):
                continue
            
            # overlaps another overflow label
            if any(tmp_poly.intersects(Polygon(oc.expanded_bbox_corners)) for oid, oc in overflow_candidates.items() if oid != node_id):
                continue

            # overlaps with a binder of another overflow label 
            if any(
                tmp_poly_tight.intersects(LineString([G.nodes[oid]["pos"], _get_anchor_pt(oc)])) 
                for oid, oc in overflow_candidates.items() if oid != node_id
            ):
                continue

            print(f"Adjusting overflow label for node {node_id} from {ol.anchor} to {name}")
            overflow_candidates[node_id] = tmp_candidate
            break

    return overflow_candidates