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
OUTER_MAX_STEPS     = 500
WEDGE_RADIUS        = 1e4
OUTER_CANDIDATE_POOL = 100   # top-K candidates kept per label after phase 1

# cost weights
W_DIST   = 0.08
W_ALIGN = 10.0
W_ANGLE  = 0.5 
W_BINDER = 1.5    # increase slightly since it's now the primary distance signal
W_OVERLAP = 80.0   # pairwise overlap penalty (per overlapping pair)
W_MISS    = 1e6    # sentinel cost for "no valid candidate"
W_TIGHT_OVERLAP = 500.0
W_PADDING = 25.0
W_BINDER_CROSS = 60.0  # add to module-level tunables
W_DIVERSITY_MAX = 2.0

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

def _generate_candidates(
    centroid, outer_polygon, placed_union,
    label, node_pos, own_node_id, own_label_id,
    G, overflow_candidates, label_candidates,
    top_k=OUTER_CANDIDATE_POOL,
) -> List[dict]:
    """
    Search all outward angles, not just a gap-derived slot.
    Gap assignment is only used for ordering/column-block structure;
    geometry is unconstrained here.
    """
    cx, cy = centroid
    node_pt = np.array(node_pos)
    ew, eh = _label_wh_expanded(label)
    tw, th = _label_wh(label)
    iw, ih = _ink_wh(label)

    # natural outward angle from centroid through the node itself
    natural_angle = _angle_from_centroid(centroid, node_pos)

    n_rays = 180
    all_offsets = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)

    scored: List[Tuple[float, dict]] = []

    for off in all_offsets:
        angle = natural_angle + off
        dx, dy = math.cos(angle), math.sin(angle)

        ray = LineString([(cx, cy), (cx + WEDGE_RADIUS * dx, cy + WEDGE_RADIUS * dy)])
        ix = outer_polygon.exterior.intersection(ray)
        if ix.is_empty:
            exit_dist = 0.0
        elif hasattr(ix, 'geoms'):
            exit_dist = max(math.hypot(p.x - cx, p.y - cy) for p in ix.geoms)
        else:
            exit_dist = math.hypot(ix.x - cx, ix.y - cy)

        start_dist = max(0.0, exit_dist)

        found_for_this_ray = 0
        blocked_anchors = set()
        for step in range(OUTER_MAX_STEPS):
            dist = start_dist + step

            ox, oy = cx + dx * dist, cy + dy * dist
            exp_bbox = _label_bbox_polygon(ox, oy, ew, eh)
            tight_bbox = _label_bbox_polygon(ox, oy, tw, th)

            if outer_polygon.intersects(exp_bbox):
                continue
            if placed_union.intersects(exp_bbox):
                continue
            if any(
                exp_bbox.contains(Point(data['pos']))
                for nid, data in G.nodes(data=True)
                if isinstance(nid, int) and nid != own_node_id
            ):
                continue

            label_center = np.array([ox, oy])
            vec = node_pt - label_center
            dist_to_node = np.linalg.norm(vec)
            unit = vec / dist_to_node if dist_to_node > 0 else vec
            own_ink = _label_bbox_polygon(ox, oy, iw, ih)
            anchors = _anchor_points(ox, oy, tw, th)

            scored_anchors = []
            for name, pt in anchors.items():
                # Skip if previously blocked on this ray
                if name in blocked_anchors:
                    continue
                va = np.array(pt) - label_center
                na = np.linalg.norm(va)
                align = float(np.dot(unit, va / na)) if na > 0 else -1.0
                scored_anchors.append((align, name, pt))
            scored_anchors.sort(reverse=True)

            for align, name, pt in scored_anchors:
                line = LineString([pt, node_pos])
                if own_ink.intersects(line):
                    blocked_anchors.add(name)
                    continue

                if not binding_line_valid(
                    line, own_node_id, own_label_id,
                    G, [], overflow_candidates, label_candidates, soft=True
                ):
                    blocked_anchors.add(name)
                    continue

                binder_cost = binding_line_valid(
                    line, own_node_id, own_label_id,
                    G, [], overflow_candidates, label_candidates
                )
                # angle_offset from natural direction — penalises drifting far
                angle_offset = abs((off + math.pi) % (2 * math.pi) - math.pi)

                c_dist   = dist ** 1.5  * W_DIST
                c_angle  = angle_offset * W_ANGLE
                c_binder = line.length  * W_BINDER
                c_align  = (1.0 - align) * W_ALIGN
                c_binder_cross = 0.0 if binder_cost else W_BINDER_CROSS
                c_diversity = math.cos(off / 2) ** 2 * W_DIVERSITY_MAX
                cost     = c_dist + c_angle + c_binder + c_align + c_binder_cross + c_diversity
                
                print(f"  label={own_label_id} dist={dist:.1f} "
                    f"| c_dist={c_dist:.1f} c_angle={c_angle:.1f} "
                    f"c_binder={c_binder:.1f} c_align={c_align:.1f} "
                    f"| c_binder_cross={c_binder_cross:.1f} c_diversity={c_diversity:.1f} "
                    f"| total={cost:.1f}")

                scored.append((cost, {
                    'cost':        cost,
                    'cx':          ox,
                    'cy':          oy,
                    'anchor_name': name,
                    'anchor_pt':   pt,
                    'bbox':        tight_bbox,
                    'exp_bbox':    exp_bbox,
                    'binder':      line,
                }))

                found_for_this_ray += 1
                break # best anchor for this position found

            # enough candidates found for this ray, or all anchors blocked, move to next ray
            if found_for_this_ray >= 3 or len(scored_anchors) <= len(blocked_anchors):
                break
                

    scored.sort(key=lambda x: x[0])
    return [c for _, c in scored[:top_k]]

# ── Phase 2: global assignment ────────────────────────────────────────────────
def _build_cost_matrix(label_ids, candidates, top_k):
    n = len(label_ids)
    ncols = n * top_k
    matrix = np.full((n, ncols), W_MISS)
    cand_by_idx = [candidates.get(lid, []) for lid in label_ids]

    for i, lid in enumerate(label_ids):
        for k, cand_i in enumerate(cand_by_idx[i]):
            base_cost = cand_i['cost']
            conflict_penalty = 0.0
            
            # Compare against ALL other labels
            for j in range(n):
                if i == j: continue
                j_cands = cand_by_idx[j]
                if not j_cands: continue
                
                # Check conflict against label j's top 3 options, not just index 0
                # This prevents the "Shadow" effect
                lookahead = min(3, len(j_cands))
                best_j_conflict = min(
                    _conflict_cost(cand_i, j_cands[m]) for m in range(lookahead)
                )
                conflict_penalty += best_j_conflict

            matrix[i, i * top_k + k] = base_cost + conflict_penalty

    return matrix

def _solve_assignment(
    label_ids: List[int],
    candidates: Dict[int, List[dict]],
    top_k: int,
) -> Dict[int, Optional[dict]]:
    """
    Solve the assignment problem globally.

    Strategy:
      1. Hungarian algorithm (scipy) — exact, O(n³)
      2. Greedy best-first — fallback when scipy absent or n > 200

    After an initial assignment, runs one round of pairwise-overlap repair
    where conflicting pairs swap to lower-cost non-overlapping alternatives.
    """
    n = len(label_ids)

    try:
        if n > 200:
            raise ImportError("problem too large for Hungarian; use greedy")
        from scipy.optimize import linear_sum_assignment

        matrix = _build_cost_matrix(label_ids, candidates, top_k)
        row_ind, col_ind = linear_sum_assignment(matrix)

        assignment: Dict[int, Optional[dict]] = {}
        for i, col in zip(row_ind, col_ind):
            lid = label_ids[i]
            k = col - i * top_k   # which candidate within this label's block
            cands = candidates.get(lid, [])
            if 0 <= k < len(cands) and matrix[i, col] < W_MISS:
                assignment[lid] = cands[k]
            else:
                assignment[lid] = None

    except ImportError:
        # greedy best-first fallback
        assignment = {}
        placed_bboxes: List = []
        # sort labels by number of candidates ascending (most constrained first)
        order = sorted(label_ids, key=lambda lid: len(candidates.get(lid, [])))
        for lid in order:
            cands = candidates.get(lid, [])
            chosen = None
            for cand in cands:
                if not any(cand['bbox'].intersects(pb) for pb in placed_bboxes):
                    chosen = cand
                    break
            if chosen is None and cands:
                chosen = cands[0]   # accept overlap rather than skip
            if chosen:
                placed_bboxes.append(chosen['bbox'])
            assignment[lid] = chosen

    # one round of overlap repair (works for both solvers)
    assignment = _repair_overlaps(label_ids, candidates, assignment)
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
        cost += W_BINDER_CROSS
    if cand_b['binder'].intersects(cand_a['bbox']):
        cost += W_BINDER_CROSS

    if cand_a['binder'].crosses(cand_b['binder']):
        cost += W_BINDER_CROSS * 0.5

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

    for node_id, chosen in assignment.items():
        ol = overflow_candidates[node_id]
        if chosen is None:
            print(f"Warning: could not place overflow label for node {node_id}")
            continue

        print(f"Success: placed label for node {node_id} "
              f"(cost={chosen['cost']:.2f})")
        update_overflow_label_position(ol, chosen['cx'], chosen['cy'], chosen['anchor_name'])
        result_map[node_id] = ol

    return result_map
