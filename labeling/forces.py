import numpy as np
import copy
from shapely import unary_union
from shapely.geometry import Polygon, Point, LineString

from overflow_bounded import _anchor_points, _label_wh, update_overflow_label_position

# --- Scaled Configuration for Small Coordinate Systems (~15x15 units) ---
W_INNER_PROXIMITY = 2.5   # Push away from internal drawing
W_GLOBAL_PROXIMITY = 2.5  # Push away from other overflow labels (Tangential)
W_CLARITY = 1.0           # Push away from other anchors
W_SPRING = 4.0            # Pull toward node (Stronger to keep binders short)
W_BINDER_DODGE = 2.5      # Push binder away from fixed labels
ITERATIONS = 200          
STEP_SIZE = 0.1     
MIN_BINDER_LENGTH = 0.25      

def optimize_overflow_labels(
    G, 
    label_candidates, 
    overflow_candidates, 
    unbounded_overflow_labels, 
    outer_nodes
):
    """
    Refines overflow label positions using force-directed optimization 
    calibrated for small-scale coordinate systems.
    """
    # 1. Setup Static Geometry
    outer_polygon = Polygon([G.nodes[n]['pos'] for n in outer_nodes])
    drawing_centroid = np.array([outer_polygon.centroid.x, outer_polygon.centroid.y])
    
    # Static obstacles for the Label Box
    placed_polys = [
        Polygon(cand.bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]
    placed_union = unary_union(placed_polys + [outer_polygon])

    # Static obstacles for the Binder Line (Fixed internal labels)
    fixed_data = []
    fixed_polys_for_line = []
    for cands in label_candidates.values():
        for cand in cands:
            poly = Polygon(cand.inner_bbox_corners)
            fixed_polys_for_line.append(poly)
            fixed_data.append((poly, np.array(cand.center)))

    # 2. Optimization Loop
    for _ in range(ITERATIONS):
        pending_updates = {}

        for node_id in unbounded_overflow_labels:
            ol = overflow_candidates[node_id]
            node_pos = np.array(G.nodes[node_id]['pos'])
            current_center = np.array(ol.center)
            exp_bbox_poly = Polygon(ol.expanded_bbox_corners)
            
            # --- PHASE A: FORCE CALCULATION ---
            force_vector = np.array([0.0, 0.0])

            # A1. Repulsion from Drawing Boundary
            dist_to_boundary = outer_polygon.distance(exp_bbox_poly)
            if dist_to_boundary < 3.0: # Scaled threshold
                dir_away = current_center - drawing_centroid
                mag = (1.0 / (dist_to_boundary + 0.2)) * W_INNER_PROXIMITY
                force_vector += (dir_away / (np.linalg.norm(dir_away) + 1e-6)) * mag

            # A2. Repulsion from Internal Fixed Labels (Box-to-Box)
            for f_poly, f_center in fixed_data:
                dist_to_fixed = f_poly.distance(exp_bbox_poly)
                if dist_to_fixed < 5.0:
                    dir_away = current_center - f_center
                    mag = (1.0 / (dist_to_fixed + 0.2)) * W_INNER_PROXIMITY
                    force_vector += (dir_away / (np.linalg.norm(dir_away) + 1e-6)) * mag

            # A3. Tangential Repulsion: Between Overflow Labels
            # Pushes labels along the ring to open gaps
            for other_id in unbounded_overflow_labels:
                if node_id == other_id: continue
                other_ol = overflow_candidates[other_id]
                dist = Polygon(ol.bbox_corners).distance(Polygon(other_ol.bbox_corners))
                
                threshold = 8.0 # Approx 3 label widths
                if dist < threshold:
                    other_center = np.array(other_ol.center)
                    radial_vec = current_center - drawing_centroid
                    radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
                    tangent_vec = np.array([-radial_unit[1], radial_unit[0]])
                    
                    vec_to_other = other_center - drawing_centroid
                    side = np.cross(radial_unit, vec_to_other / (np.linalg.norm(vec_to_other) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    
                    mag = (threshold - dist) * W_GLOBAL_PROXIMITY
                    force_vector += slide_dir * mag

            # A4. Spring Force (Pull to Node)
            force_vector += (node_pos - current_center) * W_SPRING

            # A5. Binder-to-Anchor Repulsion (Clarity)
            current_anchor_pt = _get_actual_anchor_pt(ol)
            binder_line = LineString([current_anchor_pt, node_pos])
            for other_id in unbounded_overflow_labels:
                if node_id == other_id: continue
                other_anchor_pt = _get_actual_anchor_pt(overflow_candidates[other_id])
                dist_to_anchor = binder_line.distance(Point(other_anchor_pt))
                if dist_to_anchor < 1.0:
                    dir_away = current_center - np.array(other_anchor_pt)
                    mag = (1.0 / (dist_to_anchor + 0.2)) * W_CLARITY
                    force_vector += (dir_away / (np.linalg.norm(dir_away) + 1e-6)) * mag

            # A6. Binder-to-Box Repulsion (Dodging fixed labels)
            for f_poly, f_center in fixed_data:
                dist_binder_to_fixed = binder_line.distance(f_poly)
                if dist_binder_to_fixed < 1.5:
                    radial_vec = current_center - drawing_centroid
                    radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
                    tangent_vec = np.array([-radial_unit[1], radial_unit[0]])
                    
                    vec_to_collision = f_center - drawing_centroid
                    side = np.cross(radial_unit, vec_to_collision / (np.linalg.norm(vec_to_collision) + 1e-6))
                    slide_direction = tangent_vec if side < 0 else -tangent_vec
                    
                    mag = (1.0 / (dist_binder_to_fixed + 0.1)) * W_BINDER_DODGE
                    force_vector += slide_direction * mag

            # Displacement
            proposed_center = current_center + (force_vector * 0.01 * STEP_SIZE)
            
            # --- PHASE B: ANCHOR SELECTION ---
            tw, th = _label_wh(ol)
            anchors = _anchor_points(proposed_center[0], proposed_center[1], tw, th)
            best_anchor_name = ol.anchor
            max_align = -1.0
            node_vec = node_pos - proposed_center
            for name, pt in anchors.items():
                anchor_vec = np.array(pt) - proposed_center
                align = np.dot(node_vec, anchor_vec) / (np.linalg.norm(node_vec) * np.linalg.norm(anchor_vec) + 1e-6)
                if align > max_align:
                    max_align = align
                    best_anchor_name = name

            # --- PHASE C: GUARD VALIDATION ---
            valid = True
            proposed_poly_tight = Polygon(_get_tight_corners(proposed_center, tw, th))
            proposed_anchor_pt = anchors[best_anchor_name]
            proposed_binder = LineString([proposed_anchor_pt, node_pos])

            # G0: MIN_BINDER_LENGTH Guard
            dist_to_drawing_proposed = outer_polygon.distance(proposed_poly_tight)
            if dist_to_drawing_proposed < MIN_BINDER_LENGTH:
                valid = False

            # G1: Directional
            if np.dot(node_pos - drawing_centroid, proposed_center - node_pos) <= 0:
                valid = False
            # G2: Box Collision
            if valid and proposed_poly_tight.intersects(placed_union):
                valid = False
            # G3: Binder vs Fixed Labels
            if valid:
                for f_poly in fixed_polys_for_line:
                    if proposed_binder.intersects(f_poly):
                        valid = False; break
            # G4: Dynamic Interaction
            if valid:
                for other_id in unbounded_overflow_labels:
                    if other_id == node_id: continue
                    other = overflow_candidates[other_id]
                    other_poly_tight = Polygon(other.bbox_corners)
                    other_anchor_pt = _get_actual_anchor_pt(other)
                    other_binder = LineString([other_anchor_pt, G.nodes[other_id]['pos']])
                    
                    if proposed_poly_tight.intersects(other_poly_tight):
                        valid = False; break
                    if proposed_binder.intersects(other_poly_tight):
                        valid = False; break
                    if proposed_poly_tight.intersects(other_binder):
                        valid = False; break

            if valid:
                pending_updates[node_id] = (proposed_center, best_anchor_name)

        for nid, (new_center, new_anchor) in pending_updates.items():
            update_overflow_label_position(
                overflow_candidates[nid], new_center[0], new_center[1], new_anchor
            )

    return overflow_candidates

def _get_tight_corners(center, w, h):
    cx, cy = center
    return [
        (cx - w/2, cy + h/2), (cx + w/2, cy + h/2),
        (cx + w/2, cy - h/2), (cx - w/2, cy - h/2)
    ]

def _get_actual_anchor_pt(candidate):
    anchors = _anchor_points(candidate.center[0], candidate.center[1], *_label_wh(candidate))
    return anchors[candidate.anchor]