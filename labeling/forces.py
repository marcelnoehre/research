"""
outer_overflow_forces.py

Force-based refinement pass for outer overflow label positions.

After the Hungarian assignment (outer_overflow_global.py) has produced an
initial placement, this module iteratively nudges each label's (cx, cy) centre
using a sum of physical-analogy forces until the system converges or a maximum
number of steps is reached.

Forces applied every step
─────────────────────────
  F_rep_label   : labels repel each other (inverse-square of gap between bboxes)
  F_att_label   : soft spring that prevents labels from drifting too far apart
                  (only fires when the inter-label gap exceeds a rest length)
  F_rep_binder  : label repelled from its *own* binding line and from all
                  other binding lines
  F_angle       : gradient that nudges the label to widen the angular gap
                  between its binding line and the nearest neighbour binder
  F_crossing    : penalty gradient that pushes a label away from positions
                  where its binder would cross another binder

Validity check (per micro-step)
────────────────────────────────
  After accumulating forces the candidate new position is rejected (label stays)
  if the expanded bbox would
    • re-enter the outer polygon
    • overlap already-placed (inner) label geometry
    • cause binding_line_valid() to return False

Usage
─────
  from outer_overflow_forces import refine_label_positions

  result_map = refine_label_positions(
      overflow_candidates, # {node_id: OverflowLabel}  (already positioned)
      outer_polygon,       # Shapely Polygon
      placed_union,        # Shapely geometry – inner labels already placed
      G,                   # networkx Graph
      label_candidates,    # Dict[int, List[LabelCandidate]]
  )
  # returns updated {node_id: OverflowLabel}
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely import unary_union
import networkx as nx

from label import LabelCandidate, OverflowLabel
from overflow_bounded import (
    _label_wh_expanded,
    _label_bbox_polygon,
    binding_line_valid,
    update_overflow_label_position,
)

# ── tunables ──────────────────────────────────────────────────────────────────

# simulation
MAX_ITERS           = 300
DT                  = 0.08      # base time-step (world units per step)
DAMPING             = 0.55      # velocity damping each step (0 = no inertia)
CONVERGENCE_DELTA   = 1e-3      # stop when max label displacement < this

# repulsion: label ↔ label
REP_LABEL_K         = 3.0       # reduced — gap-centering handles long-range spreading
REP_LABEL_CUTOFF    = 20.0      # wider so labels still feel each other when separated

# attraction: label ↔ label
# disabled — tangential attraction fights gap-centering; set K > 0 to re-enable
ATT_LABEL_K         = 0.0
ATT_LABEL_REST      = 4.0
ATT_LABEL_CUTOFF    = 20.0

# repulsion: label ↔ binder lines
REP_BINDER_K        = 5.0
REP_BINDER_CUTOFF   = 5.0

# angular gap centering — pulls each label to the midpoint of its largest free arc
ANG_CENTER_K        = 3.5
# secondary nearest-neighbour push (only fires when angle is very tight)
ANG_SPREAD_K        = 1.5
ANG_SPREAD_MIN_DEG  = 10.0

# crossing penalty
CROSS_K             = 40.0      # strength of anti-crossing gradient
CROSS_EPSILON       = 0.3       # finite-difference step for crossing gradient

# validity / clamping
MAX_STEP_SIZE       = 1.5       # hard cap on displacement per step (world units)
OUTER_MARGIN        = 0.3       # keep expanded bbox at least this far outside polygon

# ── internal helpers ──────────────────────────────────────────────────────────

def _ink_wh(label: OverflowLabel) -> Tuple[float, float]:
    bl, br, _, tl = label.inner_bbox_corners
    return abs(br[0] - bl[0]), abs(tl[1] - bl[1])


def _expanded_half(label: OverflowLabel) -> Tuple[float, float]:
    ew, eh = _label_wh_expanded(label)
    return ew / 2.0, eh / 2.0

def _bbox_gap(
    cx_a: float, cy_a: float, hw_a: float, hh_a: float,
    cx_b: float, cy_b: float, hw_b: float, hh_b: float,
) -> float:
    '''
    Gap between two axis-aligned bboxes (negative = overlap).
    '''
    dx = abs(cx_a - cx_b) - (hw_a + hw_b)
    dy = abs(cy_a - cy_b) - (hh_a + hh_b)
    return max(dx, dy)   # positive = separated; gap is the smaller axis


def _point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
    '''
    Euclidean distance from point P to segment AB.
    '''
    abx, aby = bx - ax, by - ay
    len_sq   = abx * abx + aby * aby
    if len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / len_sq))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def _closest_point_on_segment(px, py, ax, ay, bx, by) -> Tuple[float, float]:
    abx, aby = bx - ax, by - ay
    len_sq   = abx * abx + aby * aby
    if len_sq < 1e-12:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / len_sq))
    return ax + t * abx, ay + t * aby


def _segments_cross(ax, ay, bx, by, cx2, cy2, dx, dy) -> bool:
    '''
    Check wether segment AB strictly crosses segment CD.
    '''
    def _cross2(ux, uy, vx, vy):
        return ux * vy - uy * vx
    denom = _cross2(bx - ax, by - ay, dx - cx2, dy - cy2)
    if abs(denom) < 1e-10:
        return False
    t = _cross2(cx2 - ax, cy2 - ay, dx - cx2, dy - cy2) / denom
    u = _cross2(cx2 - ax, cy2 - ay, bx - ax, by - ay) / denom
    return 1e-6 < t < 1.0 - 1e-6 and 1e-6 < u < 1.0 - 1e-6

# Forces
def _force_rep_label(
    i: int,
    states: List[dict],
) -> Tuple[float, float]:
    '''
    Repulsion from all other labels (inverse-square on bbox gap).
    '''
    fx = fy = 0.0
    si = states[i]
    for j, sj in enumerate(states):
        if i == j:
            continue
        gap = _bbox_gap(si['cx'], si['cy'], si['hw'], si['hh'],
                        sj['cx'], sj['cy'], sj['hw'], sj['hh'])
        if gap > REP_LABEL_CUTOFF:
            continue
        dx = si['cx'] - sj['cx']
        dy = si['cy'] - sj['cy']
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            dx, dy, dist = 1.0, 0.0, 1.0
        effective = max(-gap, 0.1)
        mag = REP_LABEL_K * effective / (dist * dist)
        fx += mag * dx / dist
        fy += mag * dy / dist
    return fx, fy


def _force_att_label(
    i: int,
    states: List[dict],
) -> Tuple[float, float]:
    '''
    Spring attraction toward neighbours that have drifted too far.
    '''
    fx = fy = 0.0
    si = states[i]
    for j, sj in enumerate(states):
        if i == j:
            continue
        gap = _bbox_gap(si['cx'], si['cy'], si['hw'], si['hh'],
                        sj['cx'], sj['cy'], sj['hw'], sj['hh'])
        if gap < ATT_LABEL_REST or gap > ATT_LABEL_CUTOFF:
            continue
        dx = sj['cx'] - si['cx']
        dy = sj['cy'] - si['cy']
        dist = math.hypot(dx, dy) or 1e-6
        extension = gap - ATT_LABEL_REST
        mag = ATT_LABEL_K * extension
        fx += mag * dx / dist
        fy += mag * dy / dist
    return fx, fy


def _force_rep_binders(
    i: int,
    states: List[dict],
) -> Tuple[float, float]:
    '''
    Repel label i from all binding lines.
    '''
    fx = fy = 0.0
    si = states[i]
    cx, cy = si['cx'], si['cy']

    for j, sj in enumerate(states):
        np_x, np_y = sj['node_pos']
        ax, ay      = sj['cx'], sj['cy']
        dist = _point_to_segment_dist(cx, cy, ax, ay, np_x, np_y)
        if dist > REP_BINDER_CUTOFF or dist < 1e-6:
            continue
        # push direction: away from closest point on binder
        qx, qy = _closest_point_on_segment(cx, cy, ax, ay, np_x, np_y)
        dx, dy  = cx - qx, cy - qy
        d       = math.hypot(dx, dy) or 1e-6
        mag = REP_BINDER_K / (dist * dist)
        fx += mag * dx / d
        fy += mag * dy / d
    return fx, fy

def _force_angular_spread(
    i: int,
    states: List[dict],
    centroid: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Two-part angular force:

    1. Gap-centering (primary): find the largest free angular arc around label i,
       then pull it tangentially toward that arc's midpoint.  This reliably
       centers labels in open space regardless of how far apart they are.

    2. Nearest-neighbour push (secondary): if the smallest neighbouring angle
       is below ANG_SPREAD_MIN_DEG, add a small extra push away from that
       neighbour.  Acts as a last-resort separator for very crowded layouts.
    """
    si = states[i]
    cx, cy     = si['cx'], si['cy']
    np_x, np_y = si['node_pos']

    # Angle of label i's binder as seen from the centroid
    angle_i = math.atan2(cy - centroid[1], cx - centroid[0])

    # Collect all other binder angles, sorted
    other_angles = []
    for j, sj in enumerate(states):
        if i == j:
            continue
        angle_j = math.atan2(sj['cy'] - centroid[1], sj['cx'] - centroid[0])
        other_angles.append(angle_j)

    fx = fy = 0.0

    # gap centering
    if other_angles:
        # Build arcs: gaps between consecutive neighbour angles (sorted circle)
        sorted_others = sorted(other_angles)
        n_others = len(sorted_others)

        # Find which arc label i currently sits in and the midpoint of the
        # largest arc among all arcs (treating label i's slot as part of it).
        arcs = []
        for k in range(n_others):
            a_left  = sorted_others[k]
            a_right = sorted_others[(k + 1) % n_others]
            arc_size = (a_right - a_left) % (2 * math.pi)
            mid      = a_left + arc_size / 2.0
            arcs.append((arc_size, mid))

        # Largest arc midpoint → target angle for label i
        best_arc_size, target_angle = max(arcs, key=lambda x: x[0])

        # Angular error (signed, shortest path)
        error = (target_angle - angle_i + math.pi) % (2 * math.pi) - math.pi

        # Only pull if error is non-trivial (> 2°)
        if abs(error) > math.radians(2.0):
            mag = ANG_CENTER_K * abs(error)
            # Tangential direction at current radial position
            radial_angle = math.atan2(cy - centroid[1], cx - centroid[0])
            sign = 1.0 if error > 0 else -1.0
            tx = -math.sin(radial_angle) * sign
            ty =  math.cos(radial_angle) * sign
            fx += mag * tx
            fy += mag * ty

    # ── part 2: nearest-neighbour push ──────────────────────────────────────
    min_dangle   = math.pi
    tangent_sign = 1.0
    for angle_j in other_angles:
        dangle = (angle_i - angle_j + math.pi) % (2 * math.pi) - math.pi
        if abs(dangle) < abs(min_dangle):
            min_dangle   = dangle
            tangent_sign = 1.0 if dangle > 0 else -1.0

    threshold = math.radians(ANG_SPREAD_MIN_DEG)
    if abs(min_dangle) < threshold:
        deficit = threshold - abs(min_dangle)
        mag     = ANG_SPREAD_K * deficit
        radial_angle = math.atan2(cy - centroid[1], cx - centroid[0])
        tx = -math.sin(radial_angle) * tangent_sign
        ty =  math.cos(radial_angle) * tangent_sign
        fx += mag * tx
        fy += mag * ty

    return fx, fy


def _force_anti_crossing(
    i: int,
    states: List[dict],
) -> Tuple[float, float]:
    """
    Numerical gradient: try small displacements of label i and push away from
    positions that create binder crossings.
    """
    si = states[i]
    cx, cy  = si['cx'], si['cy']
    np_x, np_y = si['node_pos']

    # count crossings for a given (cx, cy)
    def crossings(tcx, tcy) -> int:
        count = 0
        for j, sj in enumerate(states):
            if i == j:
                continue
            jnp_x, jnp_y = sj['node_pos']
            if _segments_cross(tcx, tcy, np_x, np_y,
                               sj['cx'], sj['cy'], jnp_x, jnp_y):
                count += 1
        return count

    base = crossings(cx, cy)
    if base == 0:
        return 0.0, 0.0

    eps = CROSS_EPSILON
    gx = (crossings(cx + eps, cy) - crossings(cx - eps, cy)) / (2 * eps)
    gy = (crossings(cx, cy + eps) - crossings(cx, cy - eps)) / (2 * eps)

    mag = math.hypot(gx, gy)
    if mag < 1e-6:
        return 0.0, 0.0

    return -CROSS_K * gx / mag, -CROSS_K * gy / mag


# validity of a candidate position
def _position_valid(
    cx: float, cy: float,
    hw: float, hh: float,
    node_pos: Tuple[float, float],
    own_node_id: int,
    own_label_id: int,
    outer_polygon: Polygon,
    placed_union,
    G,
    all_overflow: Dict,
    label_candidates,
    ol: OverflowLabel,
) -> bool:
    exp_bbox = _label_bbox_polygon(cx, cy, hw * 2, hh * 2)

    # must stay outside the outer polygon
    if outer_polygon.intersects(exp_bbox):
        return False

    # must not overlap inner placed labels
    if placed_union.intersects(exp_bbox):
        return False

    # binding line validity (checks for ink / label intersections)
    iw, ih  = _ink_wh(ol)
    own_ink = _label_bbox_polygon(cx, cy, iw, ih)
    line    = LineString([(cx, cy), node_pos])
    if own_ink.intersects(line):
        return False
    if not binding_line_valid(
        line, own_node_id, own_label_id,
        G, [], all_overflow, label_candidates, soft=True,
    ):
        return False

    return True


# ── main refinement entry point ───────────────────────────────────────────────

def refine_label_positions(
    overflow_candidates: Dict[int, OverflowLabel],
    outer_nodes: List[int],
    G: nx.Graph,
    label_candidates: Dict[int, List[LabelCandidate]],
    *,
    max_iters: int = MAX_ITERS,
    dt: float = DT,
    verbose: bool = False,
) -> Dict[int, OverflowLabel]:
    """
    Refine the positions of already-placed outer overflow labels using a
    force-directed simulation.

    Parameters
    ----------
    overflow_candidates : {node_id -> OverflowLabel} — already positioned
                          (e.g. result_map from outer_overflow_labels).
                          Objects are deep-copied; originals are not modified.
    outer_polygon       : convex hull polygon of outer nodes
    placed_union        : Shapely geometry of already-placed inner labels
    G                   : the graph
    label_candidates    : inner label candidates (for binding_line_valid)
    max_iters           : simulation step budget
    dt                  : time-step size
    verbose             : print per-iteration diagnostics

    Returns
    -------
    Dict[int, OverflowLabel] with updated positions (same keys as input)
    """
    outer_positions = [G.nodes[n]["pos"] for n in outer_nodes]
    outer_polygon   = Polygon(outer_positions)

    placed_union = unary_union([
        Polygon(cand.expanded_bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]) if label_candidates else Polygon()

    # ── build simulation state ────────────────────────────────────────────────
    placed_ids = list(overflow_candidates.keys())
    if not placed_ids:
        return {}

    ol_map = {nid: copy.deepcopy(ol) for nid, ol in overflow_candidates.items()}

    states: List[dict] = []
    for nid in placed_ids:
        ol     = ol_map[nid]
        hw, hh = _expanded_half(ol)
        # ol.center is the current (cx, cy) after update_overflow_label_position
        cx, cy = ol.center
        states.append({
            'node_id':      nid,
            'own_node_id':  ol.node_id,
            'cx':           cx,
            'cy':           cy,
            'hw':           hw,
            'hh':           hh,
            'node_pos':     G.nodes[ol.node_id]['pos'],
            'anchor_name':  ol.anchor,
            'ol':           ol,
            'vx':           0.0,
            'vy':           0.0,
        })

    n = len(states)
    centroid = (outer_polygon.centroid.x, outer_polygon.centroid.y)

    # ── simulation loop ───────────────────────────────────────────────────────
    for iteration in range(max_iters):
        max_disp = 0.0

        # Compute all forces before moving anything (synchronous update)
        forces = []
        for i in range(n):
            frx, fry = _force_rep_label(i, states)
            fax, fay = _force_att_label(i, states)
            fbx, fby = _force_rep_binders(i, states)
            fasx, fasy = _force_angular_spread(i, states, centroid)
            fcx, fcy = _force_anti_crossing(i, states)

            fx = frx + fax + fbx + fasx + fcx
            fy = fry + fay + fby + fasy + fcy
            forces.append((fx, fy))

        # Integrate and validate
        for i, si in enumerate(states):
            fx, fy = forces[i]

            # Damped velocity integration
            si['vx'] = DAMPING * si['vx'] + fx * dt
            si['vy'] = DAMPING * si['vy'] + fy * dt

            # Clamp step size
            step = math.hypot(si['vx'], si['vy'])
            if step > MAX_STEP_SIZE:
                si['vx'] *= MAX_STEP_SIZE / step
                si['vy'] *= MAX_STEP_SIZE / step

            new_cx = si['cx'] + si['vx']
            new_cy = si['cy'] + si['vy']

            # Reject if the new position is invalid
            if _position_valid(
                new_cx, new_cy, si['hw'], si['hh'],
                si['node_pos'], si['own_node_id'], si['node_id'],
                outer_polygon, placed_union, G,
                ol_map, label_candidates, si['ol'],
            ):
                disp = math.hypot(new_cx - si['cx'], new_cy - si['cy'])
                max_disp = max(max_disp, disp)
                si['cx'], si['cy'] = new_cx, new_cy
            else:
                # Kill velocity so the label doesn't keep trying the same move
                si['vx'] = si['vy'] = 0.0

        if verbose:
            crossings_total = sum(
                1
                for ii in range(n)
                for jj in range(ii + 1, n)
                if _segments_cross(
                    states[ii]['cx'], states[ii]['cy'],
                    *states[ii]['node_pos'],
                    states[jj]['cx'], states[jj]['cy'],
                    *states[jj]['node_pos'],
                )
            )
            print(f"[Forces] iter={iteration+1:3d}  max_disp={max_disp:.4f}  "
                  f"crossings={crossings_total}")

        if max_disp < CONVERGENCE_DELTA:
            if verbose:
                print(f"[Forces] converged after {iteration+1} iterations.")
            break

    # ── write results back ────────────────────────────────────────────────────
    result_map: Dict[int, OverflowLabel] = {}
    for si in states:
        nid = si['node_id']
        ol  = si['ol']
        update_overflow_label_position(ol, si['cx'], si['cy'], si['anchor_name'])
        result_map[nid] = ol

        if verbose:
            orig_cx, orig_cy = overflow_candidates[nid].center
            delta = math.hypot(si['cx'] - orig_cx, si['cy'] - orig_cy)
            print(f"  label={nid}  Δ={delta:.3f}  "
                  f"final=({si['cx']:.2f}, {si['cy']:.2f})")

    return result_map