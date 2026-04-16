"""
outer_overflow_global.py

Replaces the greedy gap-by-gap loop in outer_overflow_labels with a two-phase
global optimiser:

  Phase 1 : generate a pool of OUTER_CANDIDATE_POOL scored candidates per label
             independently (no cross-label state during search).

  Phase 2 : build a cost matrix [labels x candidates] and solve it with the
             Hungarian algorithm (scipy.optimize.linear_sum_assignment).
             Pairwise overlap between chosen candidates is added to the cost
             so the solver can trade a slight overlap on one label for a much
             better position on another.

Falls back to greedy best-first if scipy is absent or the problem is very large.
"""

import copy
import math
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

from shapely.geometry import LineString, Point, Polygon
from shapely import unary_union
from typing import Dict, List, Optional, Tuple

from label import LabelCandidate, OverflowLabel
from overflow_bounded import (
    _label_wh,
    _label_wh_expanded,
    _label_bbox_polygon,
    _anchor_points,
    update_overflow_label_position,
    binding_line_valid,
)

# ── tunables ────────────────────────────────────────────────────────────────
OUTER_STEP          = 0.5
OUTER_MAX_STEPS     = 50
WEDGE_RADIUS        = 1e4
OUTER_CANDIDATE_POOL = 100   # top-K candidates kept per label after phase 1
CANDIDATES_PER_RAY = 2
RAYS = 180

# cost weights
W_ALIGN         = 1.0   # anchor alignment
W_ANGLE         = 0.5   # natural angle
W_BOUNDARY      = 3.0   # close to polygon boundary
W_BINDER        = 5.0   # length of binding line
W_BINDER_INTERSECT = 50.0 # penalty for binder intersecting another label
W_BINDER_CROSS  = 8.0   # penalty for crossing binding lines
W_TIGHT_OVERLAP = 80.0  # penalty for ink overlaps 
W_OVERLAP       = 10.0  # penalty for padding overlaps
W_PADDING       = 5.0   # penalty for padding
W_MISS          = 1e6   # penalty for unplaced label

ITERATIVE_HUNGARIAN_MAX_ITERS = 20
ITER_PENALTY_MULTIPLIER       = 2.0

def _generate_candidates_task(args: dict) -> tuple[int, list[dict]]:
    node_id = args["node_id"]
    cands = _generate_candidates(
        centroid            = args["centroid"],
        outer_polygon       = args["outer_polygon"],
        placed_union        = args["placed_union"],
        label               = args["label"],
        node_pos            = args["node_pos"],
        own_node_id         = args["own_node_id"],
        own_label_id        = args["own_label_id"],
        G                   = args["G"],
        overflow_candidates = args["overflow_candidates"],
        label_candidates    = args["label_candidates"],
        top_k               = args["top_k"],
    )
    return node_id, cands

# ── helpers (unchanged from original) ───────────────────────────────────────

def _ink_wh(label: OverflowLabel) -> Tuple[float, float]:
    bl, br, _, tl = label.inner_bbox_corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])


def _angle_from_centroid(centroid, point) -> float:
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


def _angular_gap_between(a1: float, a2: float) -> float:
    return (a2 - a1) % (2 * math.pi)


def _sort_by_node_angle(assigned, gap, centroid, G, overflow_candidates):
    a_left = gap['a_left']
    def key(node_id):
        pos = G.nodes[overflow_candidates[node_id].node_id]['pos']
        return _angular_gap_between(a_left, _angle_from_centroid(centroid, pos))
    return sorted(assigned, key=key)

# ── Phase 1: per-label candidate generation ──────────────────────────────────
OUTER_MARGIN = 0.25
GRID_STEP = 0.5

def _generate_candidates(
    centroid, outer_polygon, placed_union,
    label, node_pos, own_node_id, own_label_id,
    G, overflow_candidates, label_candidates,
    top_k=OUTER_CANDIDATE_POOL,
) -> List[dict]:
    """
    Grid-based candidate generation.

    1. Build a regular grid that tiles the bounding box of outer_polygon
       expanded by one label-width on every side.
    2. For each cell center, AABB-reject anything that overlaps the polygon,
       placed labels, or another node.  Only surviving cells get Shapely work.
    3. For each surviving cell pick the best anchor (vectorised dot-product),
       validate the binder once, score, keep.
    4. Sort by cost, return top-K.
    """

    # ── pre-compute once ─────────────────────────────────────────────────────
    ew, eh = _label_wh_expanded(label)
    tw, th = _label_wh(label)
    iw, ih = _ink_wh(label)
    hw, hh = ew / 2, eh / 2

    node_pt = np.array(node_pos, dtype=float)

    other_node_pos = np.array(
        [data['pos'] for nid, data in G.nodes(data=True)
         if isinstance(nid, int) and nid != own_node_id],
        dtype=float
    )
    anchor_names = [
        'top_left','top_right','bottom_left','bottom_right',
        'top','bottom','left','right',
    ]
    anchor_offsets = np.array([
        [-tw/2,  th/2], [ tw/2,  th/2],
        [-tw/2, -th/2], [ tw/2, -th/2],
        [  0.0,  th/2], [  0.0, -th/2],
        [-tw/2,   0.0], [ tw/2,   0.0],
    ], dtype=float) 

    poly_bounds = outer_polygon.bounds
    poly_ext    = outer_polygon.exterior

    # ── build grid ───────────────────────────────────────────────────────────
    margin = max(ew, eh) + OUTER_MARGIN
    gx0 = poly_bounds[0] - margin
    gy0 = poly_bounds[1] - margin
    gx1 = poly_bounds[2] + margin
    gy1 = poly_bounds[3] + margin

    xs = np.arange(gx0 + hw, gx1, GRID_STEP)
    ys = np.arange(gy0 + hh, gy1, GRID_STEP)
    gx, gy = np.meshgrid(xs, ys)
    cx_all = gx.ravel()
    cy_all = gy.ravel()

    # ── pass 1: bulk AABB filter (pure numpy, no Shapely) ────────────────────
    # Expanded bbox corners for every cell.
    ex0 = cx_all - hw;  ey0 = cy_all - hh
    ex1 = cx_all + hw;  ey1 = cy_all + hh

    # Keep cells whose AABB does NOT overlap the polygon bounding box
    overlaps_poly_aabb = ~(
        (ex1 < poly_bounds[0]) | (ex0 > poly_bounds[2]) |
        (ey1 < poly_bounds[1]) | (ey0 > poly_bounds[3])
    )
    keep = np.ones(len(cx_all), dtype=bool)

    # Node containment: reject cells whose expanded bbox contains another node
    if len(other_node_pos):
        ox = other_node_pos[:, 0]
        oy = other_node_pos[:, 1]
        node_inside = (
            (ox[None, :] >= ex0[:, None]) & (ox[None, :] <= ex1[:, None]) &
            (oy[None, :] >= ey0[:, None]) & (oy[None, :] <= ey1[:, None])
        ).any(axis=1)
        keep &= ~node_inside

    candidate_indices = np.where(keep)[0]

    # ── pass 2: Shapely intersection tests on surviving candidates ────────────
    scored: List[Tuple[float, dict]] = []

    for i in candidate_indices:
        ox_i, oy_i = float(cx_all[i]), float(cy_all[i])

        # Shapely polygon checks — only for cells that survived AABB.
        exp_bbox = _label_bbox_polygon(ox_i, oy_i, ew, eh)

        # Must be outside the drawing polygon.
        if overlaps_poly_aabb[i] and outer_polygon.intersects(exp_bbox):
            continue

        # Must not overlap already-placed labels.
        if placed_union.intersects(exp_bbox):
            continue

        tight_bbox  = _label_bbox_polygon(ox_i, oy_i, tw, th)
        own_ink     = _label_bbox_polygon(ox_i, oy_i, iw, ih)
        label_center = np.array([ox_i, oy_i])

        # ── anchor selection (vectorised) ────────────────────────────────────
        vec  = node_pt - label_center
        dist = float(np.linalg.norm(vec))
        unit = vec / dist if dist > 0 else vec

        pts    = label_center + anchor_offsets
        va     = pts - label_center
        na     = np.linalg.norm(va, axis=1)
        safe   = na > 0
        aligns = np.where(safe, (va / np.where(safe[:,None], na[:,None], 1.0)) @ unit, -1.0)

        # Try anchors best-first; stop at first valid binder.
        chosen_name = chosen_align = anchor_binder = anchor_shapely_pt = None
        for idx in np.argsort(-aligns):
            pt  = pts[idx]
            pt_shapely = Point(float(pt[0]), float(pt[1]))
            line = LineString([pt.tolist(), node_pos])

            if own_ink.intersects(line):
                continue
            if not binding_line_valid(
                line, own_node_id, own_label_id,
                G, [], overflow_candidates, label_candidates, soft=True
            ):
                continue

            chosen_name       = anchor_names[idx]
            chosen_align      = float(aligns[idx])
            anchor_binder     = line
            anchor_shapely_pt = pt_shapely
            break

        if chosen_name is None:
            continue

        angle_to_cell = math.atan2(oy_i - centroid[1], ox_i - centroid[0])
        node_angle    = math.atan2(node_pos[1] - centroid[1], node_pos[0] - centroid[0])
        angle_offset  = abs((angle_to_cell - node_angle + math.pi) % (2 * math.pi) - math.pi)

        dist_to_boundary = poly_ext.distance(anchor_shapely_pt)

        c_angle    = angle_offset           * W_ANGLE
        c_binder   = anchor_binder.length   * W_BINDER
        c_align    = (1.0 - chosen_align)   * W_ALIGN
        c_boundary = dist_to_boundary       * W_BOUNDARY
        cost       = c_angle + c_binder + c_align + c_boundary

        print(f"  label={own_label_id} dist={dist:.1f} "
                    f"| c_angle={c_angle:.1f} "
                    f"c_binder={c_binder:.1f} c_align={c_align:.1f} "
                    f"c_boundary={c_boundary:.1f} | total={cost:.1f}")

        scored.append((cost, {
            'cost':        cost,
            'cx':          ox_i,
            'cy':          oy_i,
            'anchor_name': chosen_name,
            'anchor_pt':   (anchor_shapely_pt.x, anchor_shapely_pt.y),
            'bbox':        tight_bbox,
            'exp_bbox':    exp_bbox,
            'binder':      anchor_binder,
        }))

    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:top_k]]

# ── Phase 2: global assignment ────────────────────────────────────────────────
def _find_conflicting_pairs(
    label_ids: List[int],
    assignment: Dict[int, Optional[dict]],
) -> List[Tuple[int, int]]:
    """
    Return every (lid_a, lid_b) pair where the assigned candidates overlap.
    Each pair is reported once (lid_a < lid_b).
    """
    conflicts = []
    ids_with_cand = [lid for lid in label_ids if assignment.get(lid) is not None]
    for idx_a in range(len(ids_with_cand)):
        for idx_b in range(idx_a + 1, len(ids_with_cand)):
            lid_a = ids_with_cand[idx_a]
            lid_b = ids_with_cand[idx_b]
            if _conflict_cost(assignment[lid_a], assignment[lid_b]) > 0.0:
                conflicts.append((lid_a, lid_b))
    return conflicts

def _build_cost_matrix_with_penalties(
    label_ids: List[int],
    cand_by_idx: List[List[dict]],
    top_k: int,
    penalty_table: Dict[Tuple[int, int], float],
) -> np.ndarray:
    """
    Same structure as the old _build_cost_matrix, but:
      - drops the lookahead heuristic (no longer needed — real conflicts are
        captured iteratively instead)
      - adds per-cell penalty from the penalty_table built up across rounds
    """
    n = len(label_ids)
    matrix = np.full((n, n * top_k), W_MISS)

    for i in range(n):
        for k, cand_i in enumerate(cand_by_idx[i]):
            base_cost = cand_i['cost']
            injected  = penalty_table.get((i, k), 0.0)
            matrix[i, i * top_k + k] = base_cost + injected

    return matrix

def _solve_assignment(
    label_ids: List[int],
    candidates: Dict[int, List[dict]],
    top_k: int,
) -> Dict[int, Optional[dict]]:
    n = len(label_ids)
    if not label_ids:
        return {}

    try:
        if n > 200:
            raise ImportError("too large")
        from scipy.optimize import linear_sum_assignment

        cand_by_idx = [candidates.get(lid, []) for lid in label_ids]
        penalty_table: Dict[Tuple[int, int], float] = {}
        
        # Track the globally best state across all iterations
        best_assignment: Dict[int, Optional[dict]] = {}
        min_actual_cost = float('inf')

        for iteration in range(ITERATIVE_HUNGARIAN_MAX_ITERS):
            # Build matrix using cumulative penalties
            matrix = _build_cost_matrix_with_penalties(
                label_ids, cand_by_idx, top_k, penalty_table
            )
            row_ind, col_ind = linear_sum_assignment(matrix)

            current_assignment: Dict[int, Optional[dict]] = {}
            current_chosen_idx: Dict[int, int] = {}
            
            # 1. Sum up the "Inherent" costs (Phase 1 scores)
            current_base_sum = 0.0
            for i, col in zip(row_ind, col_ind):
                lid = label_ids[i]
                k = col - i * top_k
                cands = cand_by_idx[i]
                if 0 <= k < len(cands) and matrix[i, col] < W_MISS:
                    current_assignment[lid] = cands[k]
                    current_chosen_idx[lid] = k
                    current_base_sum += cands[k]['cost']
                else:
                    current_assignment[lid] = None
                    current_chosen_idx[lid] = -1
                    current_base_sum += W_MISS

            # 2. Calculate the "Real" conflict costs for this specific layout
            conflict_pairs = _find_conflicting_pairs(label_ids, current_assignment)
            current_conflict_sum = sum(
                _conflict_cost(current_assignment[la], current_assignment[lb])
                for la, lb in conflict_pairs
            )

            total_actual_cost = current_base_sum + current_conflict_sum
            
            # 3. Best-in-show tracking
            if total_actual_cost < min_actual_cost:
                min_actual_cost = total_actual_cost
                best_assignment = copy.copy(current_assignment)
                print(f"[Hungarian] Iter {iteration+1}: New Best Cost! {total_actual_cost:.2f}")
            else:
                print(f"[Hungarian] Iter {iteration+1}: Cost {total_actual_cost:.2f} (Best: {min_actual_cost:.2f})")

            # Exit early only if we hit a perfect zero-conflict state
            if not conflict_pairs:
                break

            # 4. Apply penalties to push the solver away from these specific choices
            for lid_a, lid_b in conflict_pairs:
                idx_a, idx_b = label_ids.index(lid_a), label_ids.index(lid_b)
                ka, kb = current_chosen_idx[lid_a], current_chosen_idx[lid_b]
                
                pair_cost = _conflict_cost(current_assignment[lid_a], current_assignment[lid_b])
                scale = ITER_PENALTY_MULTIPLIER ** iteration
                
                if ka >= 0:
                    penalty_table[(idx_a, ka)] = penalty_table.get((idx_a, ka), 0.0) + (pair_cost * scale)
                if kb >= 0:
                    penalty_table[(idx_b, kb)] = penalty_table.get((idx_b, kb), 0.0) + (pair_cost * scale)

        assignment = best_assignment

    except ImportError:
        assignment = _greedy_assignment(label_ids, candidates)

    # Final pass to fix any minor local overlaps that don't hurt the global score
    return _repair_overlaps(label_ids, candidates, assignment)

def _greedy_assignment(
    label_ids: List[int],
    candidates: Dict[int, List[dict]],
) -> Dict[int, Optional[dict]]:
    assignment: Dict[int, Optional[dict]] = {}
    placed_bboxes: List = []
    order = sorted(label_ids, key=lambda lid: len(candidates.get(lid, [])))
    for lid in order:
        chosen = None
        for cand in candidates.get(lid, []):
            if not any(cand['bbox'].intersects(pb) for pb in placed_bboxes):
                chosen = cand
                break
        if chosen is None and candidates.get(lid):
            chosen = candidates[lid][0]
        if chosen:
            placed_bboxes.append(chosen['bbox'])
        assignment[lid] = chosen
    return assignment


def _conflict_cost(cand_a: dict, cand_b: dict) -> float:
    cost = 0.0

    if cand_a['bbox'].intersects(cand_b['bbox']):
        overlap_area = cand_a['bbox'].intersection(cand_b['bbox']).area
        cost += W_TIGHT_OVERLAP + W_OVERLAP * (overlap_area / 200.0)

    if cand_a['exp_bbox'].intersects(cand_b['exp_bbox']):
        enc_area = cand_a['exp_bbox'].intersection(cand_b['exp_bbox']).area
        cost += W_PADDING * (enc_area / 400.0)

    if cand_a['binder'].intersects(cand_b['bbox']):
        cost += W_BINDER_INTERSECT
    if cand_b['binder'].intersects(cand_a['bbox']):
        cost += W_BINDER_INTERSECT

    if cand_a['binder'].crosses(cand_b['binder']):
        cost += W_BINDER_CROSS

    return cost

def _repair_overlaps(label_ids, candidates, assignment, max_passes=3):
    for _ in range(max_passes):
        made_progress = False
        for i, lid_a in enumerate(label_ids):
            ca = assignment.get(lid_a)
            if not ca: continue

            # Find who lid_a is currently hitting
            for lid_b in label_ids:
                if lid_a == lid_b or not assignment.get(lid_b): continue
                cb = assignment[lid_b]
                
                if _conflict_cost(ca, cb) > 0:
                    # Try to find a replacement for lid_a that is better
                    best_alt = ca
                    min_conflict = sum(_conflict_cost(ca, assignment[o]) for o in label_ids if o != lid_a and assignment.get(o))

                    for alt in candidates.get(lid_a, []):
                        new_conflict = sum(_conflict_cost(alt, assignment[o]) for o in label_ids if o != lid_a and assignment.get(o))
                        
                        # Fix: Accept if total conflict decreases, even if not zero
                        if new_conflict < min_conflict:
                            min_conflict = new_conflict
                            best_alt = alt
                            made_progress = True
                    
                    assignment[lid_a] = best_alt
        if not made_progress: break
    return assignment

# ── public entry point ────────────────────────────────────────────────────────

def outer_overflow_labels(
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_candidates: Dict[int, OverflowLabel],
    outer_nodes: List[int],
) -> Dict[int, OverflowLabel]:
    """
    Two-phase global placement of outer overflow labels.

    Phase 1: generate candidate pools per label (no cross-label state).
    Phase 2: assign globally via Hungarian min-cost matching + overlap repair.
    """
    outer_positions = [G.nodes[n]["pos"] for n in outer_nodes]
    outer_polygon   = Polygon(outer_positions)
    cx, cy          = outer_polygon.centroid.x, outer_polygon.centroid.y
    centroid        = (cx, cy)

    # space already blocked by inner (bounded) label candidates
    placed_union = unary_union([
        Polygon(cand.expanded_bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]) if label_candidates else Polygon()

    # angular gaps between outer boundary nodes
    node_angles = sorted(
        [(_angle_from_centroid(centroid, G.nodes[n]['pos']), n) for n in outer_nodes],
        key=lambda x: x[0]
    )
    n_outer = len(node_angles)
    gaps = []
    for i in range(n_outer):
        a_left,  node_left  = node_angles[i]
        a_right, node_right = node_angles[(i + 1) % n_outer]
        gap_size = _angular_gap_between(a_left, a_right)
        gaps.append({
            "a_left": a_left, "a_right": a_right,
            "gap_size": gap_size,
            "node_left": node_left, "node_right": node_right,
            "assigned": [],
        })

    # assign unplaced overflow labels to their natural gap
    unplaced = {nid: ol for nid, ol in overflow_candidates.items() if ol.anchor == 'overflow'}
    for node_id, ol in unplaced.items():
        outward_angle = _angle_from_centroid(centroid, ol.center)
        best_gap = max(
            (g for g in gaps if _angular_gap_between(g["a_left"], outward_angle) <= g["gap_size"]),
            key=lambda g: g["gap_size"],
            default=max(gaps, key=lambda g: g["gap_size"]),
        )
        best_gap["assigned"].append(node_id)

    # ── Phase 1: generate candidates per gap ─────────────────────────────────
    all_candidates: Dict[int, List[dict]] = {}

    tasks: list[dict] = []
    for gap in gaps:
        if not gap["assigned"]:
            continue
        assigned = _sort_by_node_angle(
            gap["assigned"], gap, centroid, G, overflow_candidates
        )
        for node_id in assigned:
            ol = overflow_candidates[node_id]
            node_pos = G.nodes[ol.node_id]["pos"]
            tasks.append({
                "node_id":            node_id,
                "centroid":           centroid,
                "outer_polygon":      outer_polygon,
                "placed_union":       placed_union,
                "label":              ol,
                "node_pos":           node_pos,
                "own_node_id":        ol.node_id,
                "own_label_id":       node_id,
                "G":                  G,
                "overflow_candidates": overflow_candidates,
                "label_candidates":   label_candidates,
                "top_k":              OUTER_CANDIDATE_POOL,
            })
    
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_generate_candidates_task, t): t["node_id"] for t in tasks}
        for fut in as_completed(futures):
            node_id, cands = fut.result()
            all_candidates[node_id] = cands
            if not cands:
                print(f"Warning: no candidates generated for label {node_id}")

    # ── Phase 2: global assignment ────────────────────────────────────────────
    label_ids  = list(all_candidates.keys())
    assignment = _solve_assignment(label_ids, all_candidates, OUTER_CANDIDATE_POOL)

    for node_id, chosen in assignment.items():
        pool = all_candidates[node_id]
        if chosen and pool and chosen is not pool[0]:
            print(f"label={node_id} picked rank={pool.index(chosen)+1}/{len(pool)} "
                f"(best={pool[0]['cost']:.1f} chosen={chosen['cost']:.1f}) "
                f"— displaced by conflict")

    # ── Apply results ─────────────────────────────────────────────────────────
    result_map = dict(overflow_candidates)
    all_results = {}

    for node_id, candidates in all_candidates.items():
        for i, cand in enumerate(candidates):
            cand_ol = copy.deepcopy(overflow_candidates[node_id])
            
            update_overflow_label_position(
                cand_ol, 
                cand['cx'], 
                cand['cy'], 
                cand.get('anchor_name', 'center')
            )
            
            all_results[f"{node_id}_cand_{i}"] = cand_ol

    for node_id, chosen in assignment.items():
        ol = overflow_candidates[node_id]
        if chosen is None:
            print(f"Warning: could not place overflow label for node {node_id}")
            continue

        print(f"Success: placed label for node {node_id} "
              f"(cost={chosen['cost']:.2f})")
        update_overflow_label_position(ol, chosen['cx'], chosen['cy'], chosen['anchor_name'])
        result_map[node_id] = ol

    return all_results, result_map
