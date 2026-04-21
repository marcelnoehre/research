import numpy as np
import copy
from shapely import unary_union
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points

from overflow_bounded import _anchor_points, _label_wh, _label_wh_expanded, binding_line_valid, update_overflow_label_position

W_INNER_PROXIMITY = 5.0   # Push away from internal drawing
W_GLOBAL_PROXIMITY = 5.0  # Push away from other overflow labels (Tangential)
W_SPRING = 1.0            # Pull toward node (Stronger to keep binders short)
W_BINDER_DODGE = 5.0      # Push binder away from fixed labels
ITERATIONS = 100
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
    #########################
    # Setup Static Geometry #
    #########################
    outer_polygon = Polygon([G.nodes[n]['pos'] for n in outer_nodes])
    drawing_centroid = np.array([outer_polygon.centroid.x, outer_polygon.centroid.y])
    # bboxes of label candidates 
    placed_polys = [
        Polygon(cand.bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]
    placed_union = unary_union(placed_polys + [outer_polygon])
    # ink of label candidates
    ink_polys = [
        Polygon(cand.inner_bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]
    #####################
    # Optimization Loop #
    #####################
    for i in range(ITERATIONS):
        pending_updates = {}
        unbounded_overflow_labels.sort(key=lambda ol: Point(overflow_candidates[ol].center).distance(placed_union))


        for node_id in unbounded_overflow_labels:
            ol = overflow_candidates[node_id]
            node_pos = np.array(G.nodes[node_id]['pos'])
            current_center = np.array(ol.center)
            bbox_poly = Polygon(ol.bbox_corners)
            w, h  = _label_wh_expanded(ol)
            force_vector = np.array([0.0, 0.0])

            ################################################
            # Repulsion from label candidates (Box-to-Box) #
            ################################################
            threshold = max(w, h)

            for ink in ink_polys:
                p1, p2 = nearest_points(ink, bbox_poly)
                dist_to_ink = np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y]))

                if dist_to_ink < threshold:
                    radial_vec = current_center - drawing_centroid
                    radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
                    tangent_vec = np.array([-radial_unit[1], radial_unit[0]])

                    ink_center = np.array([ink.centroid.x, ink.centroid.y])
                    vec_to_ink = ink_center - drawing_centroid
                    side = np.cross(radial_unit, vec_to_ink / (np.linalg.norm(vec_to_ink) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec

                    mag = (threshold - dist_to_ink) * W_INNER_PROXIMITY
                    force_vector += slide_dir * mag

            #####################
            # Pushes labels along the ring to open gaps
            #####################
            threshold = max(w, h) * 0.5

            for other_id in unbounded_overflow_labels:
                if node_id == other_id: continue
                other_ol = overflow_candidates[other_id]
                other_poly = Polygon(other_ol.bbox_corners)
                p1, p2 = nearest_points(bbox_poly, other_poly)
                dist = np.linalg.norm(np.array([p2.x - p1.x, p2.y - p1.y]))
                if dist < threshold:
                    # radial line from center of the map to this label
                    radial_vec = current_center - drawing_centroid
                    radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
                    
                    # tangent (perpendicular) vector for 'sliding' (rotating the radial vector 90 degrees)
                    tangent_vec = np.array([-radial_unit[1], radial_unit[0]])
                    
                    # clockwise or counter-clockwise (slide to open the gap).
                    vec_to_other = np.array(other_ol.center) - drawing_centroid
                    side = np.cross(radial_unit, vec_to_other / (np.linalg.norm(vec_to_other) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    
                    mag = (threshold - dist) * W_GLOBAL_PROXIMITY
                    force_vector += slide_dir * mag

            #####################
            # spring force pulling binder
            #####################
            # point on the drawing closest to the label

            node_pos = np.array(G.nodes[node_id]['pos'])
            anchor_pos = np.array(_get_actual_anchor_pt(ol))
            binder = LineString([node_pos, anchor_pos])

            intersect_result = outer_polygon.boundary.intersection(binder)
            if intersect_result.is_empty:
                closest_pt = None  # Or handle as an error
            elif isinstance(intersect_result, Point):
                closest_pt = intersect_result
            else:
                anchor_pt = Point(anchor_pos)
                closest_pt = min(intersect_result.geoms, key=lambda p: p.distance(anchor_pt))

            intersection_pos = np.array(closest_pt.coords[0])

            gap_vec = anchor_pos - intersection_pos
            dist = np.linalg.norm(gap_vec) 

            displacement = dist - MIN_BINDER_LENGTH + 0.1
            print(node_id, dist, displacement)
            mag = (displacement**2) * W_SPRING
               
            unit_dir = gap_vec / (dist + 1e-6)
            force_vector -= unit_dir * mag

            #####################
            # Binder repulsion
            #####################
            current_anchor_pt = _get_actual_anchor_pt(ol)
            binder_line = LineString([current_anchor_pt, node_pos])

            radial_vec = current_center - drawing_centroid
            radial_unit = radial_vec / (np.linalg.norm(radial_vec) + 1e-6)
            tangent_vec = np.array([-radial_unit[1], radial_unit[0]])
            
            binder_threshold = 0.2

            def _apply_binder_repulsion(obstacle_poly, obstacle_center_pt):
                nonlocal force_vector
                dist_binder = binder_line.distance(obstacle_poly)
                
                if dist_binder < binder_threshold:
                    obs_center = np.array([obstacle_center_pt.x, obstacle_center_pt.y])
                    vec_to_obs = obs_center - drawing_centroid
                    side = np.cross(radial_unit, vec_to_obs / (np.linalg.norm(vec_to_obs) + 1e-6))
                    slide_dir = tangent_vec if side < 0 else -tangent_vec
                    mag = (binder_threshold - dist_binder) * W_BINDER_DODGE
                    force_vector += slide_dir * mag

            for ink_poly in ink_polys:
                _apply_binder_repulsion(ink_poly, ink_poly.centroid)

            for other_id, other_ol in overflow_candidates.items():
                if other_id == node_id: continue
                other_poly = Polygon(other_ol.bbox_corners)
                _apply_binder_repulsion(other_poly, other_poly.centroid)

            #####################
            # Displacement
            #####################
            force_mag = np.linalg.norm(force_vector)
            delta = force_vector * STEP_SIZE  # scale but preserve direction ratios
            if force_mag > 1.0:
                delta = (force_vector / force_mag) * STEP_SIZE  # only clamp if too large

            print(node_id, delta)

            proposed_center = current_center + delta

            #####################
            # Guards
            #####################
            valid = True

            tw, th = _label_wh(ol)
            anchors = _anchor_points(proposed_center[0], proposed_center[1], tw, th)
            proposed_poly_tight = Polygon(_get_tight_corners(proposed_center, tw, th))
            proposed_binder = LineString([anchors[ol.anchor], node_pos])
            proposed_anchor = anchors[ol.anchor]

            # minimal binder length
            dist_current = outer_polygon.distance(Polygon(ol.bbox_corners))
            dist_proposed = outer_polygon.distance(proposed_poly_tight)
            if dist_proposed < MIN_BINDER_LENGTH:
                if dist_proposed < dist_current:
                    print(node_id, 'minimal binder length')
                    valid = False

            # keep labels outside
            if valid:
                radial_dir = node_pos - drawing_centroid
                label_dir = proposed_center - node_pos
                if np.dot(radial_dir, label_dir) < -0.5:
                    print(node_id, 'keep labels outside')
                    valid = False

            # prevent intersections with label candidates
            if valid:
                if proposed_poly_tight.intersects(placed_union):
                    print(node_id, 'intersects placed union')
                    valid = False
            if valid:
                for ink in ink_polys:
                    if proposed_binder.intersects(ink):
                        print(node_id, 'binder intersects ink')
                        valid = False
            if valid:
                if Point(proposed_anchor).intersects(placed_union):
                    print(node_id, 'anchor intersects placed union')
                    valid = False
        
            if valid:
                for other_id in unbounded_overflow_labels:
                    if other_id == node_id:
                        continue
                
                    other = overflow_candidates[other_id]
                    other_poly_tight = Polygon(other.bbox_corners)
                    other_anchor_pt = _get_actual_anchor_pt(other)
                    other_binder = LineString([other_anchor_pt, G.nodes[other_id]['pos']])

                    if proposed_poly_tight.intersects(other_poly_tight):
                        print(node_id, 'intersects another overflow label')
                        valid = False
                        break
                    if proposed_poly_tight.intersects(other_binder):
                        print(node_id, 'intersects another binder')
                        valid = False
                        break

                    if Point(other_anchor_pt).distance(proposed_binder) < 0.15:
                        valid = False
                        break

                    if Point(proposed_anchor).distance(other_binder) < 0.15:
                        valid = False
                        break

                    if Point(proposed_anchor).distance(other_poly_tight) < 0.15:
                        valid = False
                        break

            if valid:
                synthetic_placed = []
                for other_lid, other_ol in overflow_candidates.items():
                    if other_lid == node_id:
                        continue
                
                    # Check if we have a fresh candidate position for this label
                    if other_lid in pending_updates and pending_updates[other_lid] is not None:
                        cand = pending_updates[other_lid]
                        synthetic_placed.append({
                            "label_id":     other_lid,
                            "position":     cand[0],
                            "binding_line": cand[2],
                        })
                    # Fallback to the existing committed state
                    else:
                        synthetic_placed.append({
                            "label_id":     other_lid,
                            "position":     other_ol.center,
                            "binding_line": getattr(other_ol, "binding_line", None),
                        })

                # validate binding line
                if not binding_line_valid(
                    proposed_binder,
                    ol.node_id,
                    node_id,
                    G,
                    synthetic_placed,
                    overflow_candidates,
                    label_candidates,
                    soft = True
                ):
                    print(node_id, 'invalid binder')
                    valid = False

            if valid:
                print(node_id, 'updated')
                pending_updates[node_id] = (proposed_center, ol.anchor, proposed_binder)

        if not pending_updates:
            print(f'Converged after {i+1} iterations')
            return overflow_candidates

        for nid, (new_center, new_anchor, _) in pending_updates.items():
            print(pending_updates)
            update_overflow_label_position(
                overflow_candidates[nid], new_center[0], new_center[1], new_anchor
            )

    print('Reached max iterations')
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