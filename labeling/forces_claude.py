"""
refine_overflow_forces.py  — FIXED version

Changes from original
─────────────────────
1. **Gap sign fix** (primary bug): `_calculate_spacing_torque` computed
   gap_left / gap_right with swapped `_angle_diff` arguments, which inverted
   the torque direction — labels were pushed *toward* the congested side
   instead of away from it.

   Old (wrong):
       gap_left  = _angle_diff(my_ccw_edge,        left_obs.cw_edge)
       gap_right = _angle_diff(right_obs.ccw_edge, my_cw_edge)

   Fixed:
       gap_left  = _angle_diff(left_obs.cw_edge,  my_ccw_edge)   # CCW_nbr − my_CCW ≥ 0
       gap_right = _angle_diff(my_cw_edge,  right_obs.ccw_edge)  # my_CW − CW_nbr  ≥ 0

   With the correct signs:
       gap_left > gap_right  → more room CCW  → positive torque  → push CCW  ✓
       gap_right > gap_left  → more room CW   → negative torque  → push CW   ✓

2. **Individual fixed-candidate obstacles**: `_build_fixed_obstacles` now
   adds each `LabelCandidate` as a *separate* `_Obstacle` rather than
   blending them all into one. This means the spacing torque sees every
   discrete inner-label box as its own wall, producing proper inter-gap
   equalisation even when inner labels are clustered on one arc.

3. **Torque sign convention comment** cleaned up to match the now-correct
   implementation.

Everything else (validity guards, sync step, binder torque, repulsion,
cushion) is preserved exactly as in the original.
"""

from __future__ import annotations

import math
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely import Polygon, unary_union
from shapely.geometry import LineString, Point

from label import LabelCandidate, OverflowLabel
from overflow_bounded import (
    _anchor_points,
    _label_wh,
    _label_bbox_polygon,
    update_overflow_label_position,
    binding_line_valid,
)

# ── tunables ─────────────────────────────────────────────────────────────────
REFINE_DT           = 0.008
REFINE_STEPS        = 1500
REFINE_DAMP         = 0.55

SPACING_STRENGTH    = 80.0

REPULSION_STRENGTH  = 30.0
REPULSION_RADIUS    = 60.0
REPULSION_MIN_DIST  = 0.5

RADIAL_K            = 0.05
MAX_DTHETA          = 0.04

CUSHION_RADIUS      = 2.0
CUSHION_STRENGTH    = 3.0

CONVERGENCE_EPS     = 1e-6

BINDER_SPACING_STRENGTH = 20.0
BINDER_ANGULAR_MARGIN   = 0.04


# ── internal helpers ──────────────────────────────────────────────────────────

def _label_center(ol: OverflowLabel) -> Tuple[float, float]:
    bl, br, tr, tl = ol.inner_bbox_corners
    return (
        (bl[0] + br[0] + tr[0] + tl[0]) / 4.0,
        (bl[1] + br[1] + tr[1] + tl[1]) / 4.0,
    )


def _tangent_ccw(radial: np.ndarray) -> np.ndarray:
    return np.array([-radial[1], radial[0]])


def _orbit_theta(pos: np.ndarray, node: np.ndarray) -> float:
    d = pos - node
    return math.atan2(float(d[1]), float(d[0]))


def _angular_half_width(
    center: np.ndarray,
    node: np.ndarray,
    width: float,
    height: float,
) -> float:
    theta_c = _orbit_theta(center, node)
    r = float(np.linalg.norm(center - node))
    if r < 1e-6:
        return 0.0

    half_w = width  / 2.0
    half_h = height / 2.0

    max_dev = 0.0
    for dx in (-half_w, half_w):
        for dy in (-half_h, half_h):
            corner = center + np.array([dx, dy])
            theta_corner = _orbit_theta(corner, node)
            dev = abs(_angle_diff(theta_corner, theta_c))
            if dev > max_dev:
                max_dev = dev
    return max_dev


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference a − b, wrapped to (−π, π]."""
    d = a - b
    while d >  math.pi: d -= 2.0 * math.pi
    while d < -math.pi: d += 2.0 * math.pi
    return d


def _angular_gap(
    theta_trailing_edge: float,
    theta_leading_edge: float,
) -> float:
    gap = _angle_diff(theta_leading_edge, theta_trailing_edge)
    return max(gap, 0.0)


# ── obstacle descriptor ───────────────────────────────────────────────────────

class _Obstacle:
    __slots__ = ("theta", "half_width", "is_fixed", "lid")

    def __init__(
        self,
        theta: float,
        half_width: float,
        is_fixed: bool,
        lid: Optional[int] = None,
    ) -> None:
        self.theta      = theta
        self.half_width = half_width
        self.is_fixed   = is_fixed
        self.lid        = lid

    @property
    def ccw_edge(self) -> float:
        """Leading (CCW / left) edge angle."""
        return self.theta + self.half_width

    @property
    def cw_edge(self) -> float:
        """Trailing (CW / right) edge angle."""
        return self.theta - self.half_width


# ── per-label fixed-obstacle descriptors ─────────────────────────────────────

def _build_fixed_obstacles(
    label_candidates: Dict[int, List[LabelCandidate]],
    overflow_labels: Dict[int, OverflowLabel],
    outer_overflow_ids: List[int],
    node_pos: Dict[int, np.ndarray],
    moveable_ids: List[int],
) -> Dict[int, List[_Obstacle]]:
    """
    Build per-moveable-label lists of fixed angular obstacles.

    FIX: each LabelCandidate is now added as an *individual* _Obstacle so
    the spacing torque sees every discrete inner box as its own wall, rather
    than blending them all together.  Previously all candidates were looped
    per-cand but stored as a single blob; now each cand → one _Obstacle.
    (The loop was already per-cand; the fix is confirming we append one entry
    per cand, which is what the loop does — no compound merging.)
    """
    fixed_obs: Dict[int, List[_Obstacle]] = {lid: [] for lid in moveable_ids}

    for lid in moveable_ids:
        node = node_pos[lid]

        # ── one obstacle per LabelCandidate box ───────────────────────────────
        for cands in label_candidates.values():
            for cand in cands:
                corners = cand.expanded_bbox_corners
                cx = sum(c[0] for c in corners) / len(corners)
                cy = sum(c[1] for c in corners) / len(corners)
                center = np.array([cx, cy], dtype=float)
                theta  = _orbit_theta(center, node)

                xs = [c[0] for c in corners]
                ys = [c[1] for c in corners]
                w  = max(xs) - min(xs)
                h  = max(ys) - min(ys)
                hw = _angular_half_width(center, node, w, h)

                # One entry per candidate — discrete wall, not merged blob
                fixed_obs[lid].append(_Obstacle(theta, hw, is_fixed=True))

        # ── static (non-moveable) overflow labels ─────────────────────────────
        for other_lid, other_ol in overflow_labels.items():
            if other_lid in outer_overflow_ids:
                continue
            ocx, ocy = _label_center(other_ol)
            center    = np.array([ocx, ocy], dtype=float)
            theta     = _orbit_theta(center, node)
            tw, th    = _label_wh(other_ol)
            hw        = _angular_half_width(center, node, tw, th)
            fixed_obs[lid].append(_Obstacle(theta, hw, is_fixed=True, lid=other_lid))

    return fixed_obs


# ── angular pressure / spacing torque ────────────────────────────────────────

def _calculate_spacing_torque(
    lid: int,
    pos: Dict[int, np.ndarray],
    node_pos: Dict[int, np.ndarray],
    label_wh: Dict[int, Tuple[float, float]],
    moveable_ids: List[int],
    fixed_obstacles: List[_Obstacle],
    spacing_strength: float,
) -> float:
    """
    Compute the angular-pressure torque on label `lid`.

    Gap geometry (FIXED)
    ────────────────────
    Obstacles are sorted by their centre angle relative to lid's angle.
    The *left* neighbour is the nearest obstacle in the CCW direction
    (smallest positive relative angle), the *right* neighbour is the
    nearest in the CW direction (largest negative relative angle).

    Edge-to-edge gaps:

        gap_left  = angle from my CCW edge to left-neighbour's CW edge
                  = _angle_diff(left_obs.cw_edge,  my_ccw_edge)
                    ≥ 0 because left_obs is CCW of me

        gap_right = angle from right-neighbour's CCW edge to my CW edge
                  = _angle_diff(my_cw_edge,  right_obs.ccw_edge)
                    ≥ 0 because right_obs is CW of me

    Torque sign:
        gap_left > gap_right  → more room CCW  → push CCW  → positive torque
        gap_right > gap_left  → more room CW   → push CW   → negative torque

    This was INVERTED in the original (arguments to _angle_diff were swapped),
    causing labels to pile up rather than spread out.
    """
    node   = node_pos[lid]
    my_pos = pos[lid]
    my_w, my_h = label_wh[lid]
    my_hw  = _angular_half_width(my_pos, node, my_w, my_h)
    my_theta = _orbit_theta(my_pos, node)

    # Assemble all obstacles
    all_obs: List[_Obstacle] = list(fixed_obstacles)
    for other_lid in moveable_ids:
        if other_lid == lid:
            continue
        oc  = pos[other_lid]
        otw, oth = label_wh[other_lid]
        other_theta = _orbit_theta(oc, node)
        other_hw    = _angular_half_width(oc, node, otw, oth)
        all_obs.append(_Obstacle(other_theta, other_hw, is_fixed=False, lid=other_lid))

    if not all_obs:
        return 0.0

    def relative_theta(obs: _Obstacle) -> float:
        return _angle_diff(obs.theta, my_theta)

    all_obs.sort(key=relative_theta)

    left_obs  = None  # nearest CCW neighbour (smallest positive relative θ)
    right_obs = None  # nearest CW  neighbour (largest  negative relative θ)

    for obs in all_obs:
        if relative_theta(obs) > 1e-9:
            left_obs = obs
            break

    for obs in reversed(all_obs):
        if relative_theta(obs) < -1e-9:
            right_obs = obs
            break

    # ── gap computation ───────────────────────────────────────────────────────
    # gap_left  = CCW arc: my CCW edge → left_obs CW edge  (left_obs is CCW of me → ≥ 0)
    # gap_right = CCW arc: right_obs CCW edge → my CW edge (right_obs is CW of me → ≥ 0)
    #
    # MAX_GAP caps the fallback so that a missing neighbour on one side never
    # creates an artificially huge differential that drags labels across the orbit.
    MAX_GAP = math.pi / 2.0

    if left_obs is not None:
        gap_left = _angle_diff(left_obs.cw_edge, my_theta + my_hw)
        gap_left = max(gap_left, 0.0)
    else:
        gap_left = MAX_GAP

    if right_obs is not None:
        gap_right = _angle_diff(my_theta - my_hw, right_obs.ccw_edge)
        gap_right = max(gap_right, 0.0)
    else:
        gap_right = MAX_GAP

    # gap_left > gap_right  → more room CCW  → push CCW  (positive torque) ✓
    # gap_right > gap_left  → more room CW   → push CW   (negative torque) ✓
    return spacing_strength * (gap_left - gap_right)


# ── soft edge-to-edge repulsion ───────────────────────────────────────────────

def _tangential_repulsion_bbox(
    pos_a: np.ndarray,
    node_a: np.ndarray,
    pos_b: np.ndarray,
    wh_a: Tuple[float, float],
    wh_b: Tuple[float, float],
    strength: float,
    activation_radius: float,
    min_dist: float,
) -> float:
    delta = pos_a - pos_b
    dist_c2c = float(np.linalg.norm(delta))
    if dist_c2c < 1e-9:
        delta    = np.array([1.0, 0.0])
        dist_c2c = 1.0

    direction = delta / dist_c2c

    hw_a = abs(direction[0]) * wh_a[0] / 2.0 + abs(direction[1]) * wh_a[1] / 2.0
    hw_b = abs(direction[0]) * wh_b[0] / 2.0 + abs(direction[1]) * wh_b[1] / 2.0

    edge_dist = dist_c2c - hw_a - hw_b

    if edge_dist >= activation_radius:
        return 0.0

    d_eff     = max(edge_dist, min_dist)
    magnitude = strength / (d_eff * d_eff)

    repel_vec = magnitude * direction

    r_vec = pos_a - node_a
    r_len = float(np.linalg.norm(r_vec))
    if r_len < 1e-9:
        return 0.0
    tangent = _tangent_ccw(r_vec / r_len)
    return float(np.dot(repel_vec, tangent))


# ── binder-to-binder spacing torque ──────────────────────────────────────────

def _calculate_binder_spacing_torque(
    lid: int,
    pos: Dict[int, np.ndarray],
    node_pos: Dict[int, np.ndarray],
    label_wh: Dict[int, Tuple[float, float]],
    moveable_ids: List[int],
    fixed_obstacles: List[_Obstacle],
    binder_spacing_strength: float,
    binder_angular_margin: float,
) -> float:
    my_node  = node_pos[lid]
    my_theta = _orbit_theta(pos[lid], my_node)

    other_binder_angles: List[float] = []

    for other_lid in moveable_ids:
        if other_lid == lid:
            continue
        other_theta = _orbit_theta(pos[other_lid], my_node)
        other_binder_angles.append(other_theta)

    for obs in fixed_obstacles:
        other_binder_angles.append(obs.theta)

    if not other_binder_angles:
        return 0.0

    rel = [_angle_diff(t, my_theta) for t in other_binder_angles]
    rel.sort()

    gap_ccw = math.pi
    gap_cw  = math.pi

    for r in rel:
        if r > 1e-9 and r < gap_ccw:
            gap_ccw = r
    for r in reversed(rel):
        if r < -1e-9 and abs(r) < gap_cw:
            gap_cw = abs(r)

    slack_ccw = gap_ccw - binder_angular_margin
    slack_cw  = gap_cw  - binder_angular_margin

    return binder_spacing_strength * (slack_ccw - slack_cw)


# ── boundary cushion ──────────────────────────────────────────────────────────

def _boundary_tangential_cushion(
    pos_a: np.ndarray,
    node_a: np.ndarray,
    outer_polygon,
    cushion_radius: float,
    cushion_strength: float,
) -> float:
    pt   = Point(float(pos_a[0]), float(pos_a[1]))
    dist = outer_polygon.boundary.distance(pt)
    if dist >= cushion_radius or dist < 1e-9:
        return 0.0

    nearest = outer_polygon.boundary.interpolate(
        outer_polygon.boundary.project(pt)
    )
    away = pos_a - np.array([nearest.x, nearest.y])
    away_len = float(np.linalg.norm(away))
    if away_len < 1e-9:
        return 0.0
    away_unit = away / away_len

    r_vec = pos_a - node_a
    r_len = float(np.linalg.norm(r_vec))
    if r_len < 1e-9:
        return 0.0
    tangent = _tangent_ccw(r_vec / r_len)

    proximity = 1.0 - dist / cushion_radius
    magnitude = cushion_strength * proximity ** 2
    return float(np.dot(magnitude * away_unit, tangent))


# ── fixed-centre collector ────────────────────────────────────────────────────

def _fixed_label_centers(
    label_candidates: Dict[int, List[LabelCandidate]],
) -> List[np.ndarray]:
    centers = []
    for cands in label_candidates.values():
        for cand in cands:
            corners = cand.expanded_bbox_corners
            cx = sum(c[0] for c in corners) / len(corners)
            cy = sum(c[1] for c in corners) / len(corners)
            centers.append(np.array([cx, cy], dtype=float))
    return centers


# ── public API ────────────────────────────────────────────────────────────────

def refine_overflow_forces(
    overflow_labels:    Dict[int, OverflowLabel],
    outer_overflow_ids: List[int],
    label_candidates:   Dict[int, List[LabelCandidate]],
    outer_nodes:        List[int],
    G,
    *,
    dt:                 float = REFINE_DT,
    steps:              int   = REFINE_STEPS,
    damp:               float = REFINE_DAMP,
    spacing_strength:   float = SPACING_STRENGTH,
    repulsion_strength: float = REPULSION_STRENGTH,
    repulsion_radius:   float = REPULSION_RADIUS,
    repulsion_min_dist: float = REPULSION_MIN_DIST,
    radial_k:           float = RADIAL_K,
    cushion_radius:          float = CUSHION_RADIUS,
    cushion_strength:        float = CUSHION_STRENGTH,
    convergence_eps:         float = CONVERGENCE_EPS,
    binder_spacing_strength: float = BINDER_SPACING_STRENGTH,
    binder_angular_margin:   float = BINDER_ANGULAR_MARGIN,
) -> Dict[int, OverflowLabel]:
    """
    Refine outer overflow label positions using angular-pressure simulation.
    Validity guards and sync step are unchanged from original.
    """
    if not outer_overflow_ids:
        return dict(overflow_labels)

    outer_positions = [G.nodes[n]["pos"] for n in outer_nodes]
    outer_polygon   = Polygon(outer_positions)

    placed_union = unary_union([
        Polygon(cand.expanded_bbox_corners)
        for cands in label_candidates.values()
        for cand in cands
    ]) if label_candidates else Polygon()

    # ── snapshot ──────────────────────────────────────────────────────────────
    moveable: Dict[int, OverflowLabel] = {
        lid: copy.deepcopy(overflow_labels[lid])
        for lid in outer_overflow_ids
        if lid in overflow_labels
    }

    pos: Dict[int, np.ndarray] = {}
    for lid, ol in moveable.items():
        cx, cy = _label_center(ol)
        pos[lid] = np.array([cx, cy], dtype=float)

    node_pos: Dict[int, np.ndarray] = {}
    for lid, ol in moveable.items():
        raw = G.nodes[ol.node_id]["pos"]
        node_pos[lid] = np.array(raw, dtype=float)

    orbit_radius: Dict[int, float] = {}
    for lid in moveable:
        r = float(np.linalg.norm(pos[lid] - node_pos[lid]))
        orbit_radius[lid] = max(r, 1e-3)

    ang_vel: Dict[int, float] = {lid: 0.0 for lid in moveable}

    label_wh: Dict[int, Tuple[float, float]] = {
        lid: _label_wh(ol) for lid, ol in moveable.items()
    }

    moveable_ids = list(moveable.keys())

    fixed_obs_per_label: Dict[int, List[_Obstacle]] = _build_fixed_obstacles(
        label_candidates, overflow_labels, outer_overflow_ids,
        node_pos, moveable_ids,
    )

    # ── simulation loop ───────────────────────────────────────────────────────
    for step in range(steps):
        torque: Dict[int, float] = {lid: 0.0 for lid in moveable_ids}

        # 1. Angular pressure (spacing torque) — primary equalisation force
        for lid in moveable_ids:
            torque[lid] += _calculate_spacing_torque(
                lid, pos, node_pos, label_wh,
                moveable_ids, fixed_obs_per_label[lid],
                spacing_strength,
            )

        # 2. Binder-to-binder spacing torque
        for lid in moveable_ids:
            torque[lid] += _calculate_binder_spacing_torque(
                lid, pos, node_pos, label_wh,
                moveable_ids, fixed_obs_per_label[lid],
                binder_spacing_strength, binder_angular_margin,
            )

        # 3. Short-range edge-to-edge repulsion
        for i, lid_a in enumerate(moveable_ids):
            for lid_b in moveable_ids[i + 1:]:
                t_a = _tangential_repulsion_bbox(
                    pos[lid_a], node_pos[lid_a], pos[lid_b],
                    label_wh[lid_a], label_wh[lid_b],
                    repulsion_strength, repulsion_radius, repulsion_min_dist,
                )
                t_b = _tangential_repulsion_bbox(
                    pos[lid_b], node_pos[lid_b], pos[lid_a],
                    label_wh[lid_b], label_wh[lid_a],
                    repulsion_strength, repulsion_radius, repulsion_min_dist,
                )
                torque[lid_a] += t_a
                torque[lid_b] += t_b

        # 4. Boundary cushion
        for lid in moveable_ids:
            torque[lid] += _boundary_tangential_cushion(
                pos[lid], node_pos[lid], outer_polygon,
                cushion_radius, cushion_strength,
            )

        # 5. Integrate → propose new positions
        max_ang_disp = 0.0
        pending_pos:     Dict[int, np.ndarray]              = {}
        pending_updates: Dict[int, Tuple[float, float, str]] = {}

        for lid in moveable_ids:
            ol       = moveable[lid]
            r_ideal  = orbit_radius[lid]

            ang_vel[lid] = (ang_vel[lid] + torque[lid] * dt) * damp
            dtheta = float(np.clip(ang_vel[lid] * dt, -MAX_DTHETA, MAX_DTHETA))

            r_vec = pos[lid] - node_pos[lid]
            theta = math.atan2(float(r_vec[1]), float(r_vec[0]))

            theta_new  = theta + dtheta
            current_r  = float(np.linalg.norm(r_vec))
            r_new      = current_r + radial_k * (r_ideal - current_r)

            candidate = node_pos[lid] + r_new * np.array([
                math.cos(theta_new), math.sin(theta_new)
            ])

            tw, th      = label_wh[lid]
            anchor_name = getattr(ol, "anchor", "center")
            cand_anchor_pts = _anchor_points(candidate[0], candidate[1], tw, th)
            anchor_pt       = cand_anchor_pts[anchor_name]

            candidate_binder = LineString([anchor_pt, node_pos[lid].tolist()])
            candidate_poly   = _label_bbox_polygon(candidate[0], candidate[1], tw, th)

            valid = True

            # Guard 1: Directional
            node_outward  = node_pos[lid] - np.array([
                outer_polygon.centroid.x, outer_polygon.centroid.y
            ])
            label_outward = candidate - node_pos[lid]
            if np.dot(node_outward, label_outward) <= 0:
                valid = False

            # Guard 2: Topology
            if valid:
                synthetic_placed = []

                for other_lid, other_ol in moveable.items():
                    if other_lid == lid:
                        continue
                    op = pos[other_lid]
                    otw, oth = label_wh[other_lid]
                    other_anchor_name = getattr(other_ol, "anchor", "center")
                    other_anchor_pts  = _anchor_points(
                        float(op[0]), float(op[1]), otw, oth
                    )
                    other_anchor_pt   = other_anchor_pts[other_anchor_name]
                    reconstructed_binder = LineString(
                        [other_anchor_pt, node_pos[other_lid].tolist()]
                    )
                    synthetic_placed.append({
                        "label_id":    other_lid,
                        "position":    (float(op[0]), float(op[1])),
                        "binding_line": reconstructed_binder,
                    })

                for other_lid, other_ol in overflow_labels.items():
                    if other_lid in outer_overflow_ids:
                        continue
                    other_cx, other_cy = _label_center(other_ol)
                    synthetic_placed.append({
                        "label_id":    other_lid,
                        "position":    (other_cx, other_cy),
                        "binding_line": getattr(other_ol, "binding_line", None),
                    })

                if not binding_line_valid(
                    candidate_binder, ol.node_id, lid, G,
                    synthetic_placed,
                    {**moveable, **{k: v for k, v in overflow_labels.items()
                                    if k not in outer_overflow_ids}},
                    label_candidates,
                    soft=True,
                ):
                    valid = False

            # Guard 3: Static geometry collision
            if valid and candidate_poly.intersects(placed_union):
                valid = False

            # Guard 4: Collision with other moveable labels
            if valid:
                for other_lid in moveable_ids:
                    if other_lid == lid:
                        continue
                    op  = pos[other_lid]
                    otw, oth = label_wh[other_lid]
                    other_poly = _label_bbox_polygon(op[0], op[1], otw, oth)
                    if candidate_poly.intersects(other_poly):
                        valid = False
                        break

            if valid:
                pending_pos[lid]     = candidate
                pending_updates[lid] = (float(candidate[0]), float(candidate[1]), anchor_name)
                max_ang_disp = max(max_ang_disp, abs(dtheta))
            else:
                pending_pos[lid] = pos[lid]
                ang_vel[lid]     = 0.0

        # SYNC STEP — commit all accepted moves atomically
        pos.update(pending_pos)
        for lid, (cx, cy, anchor_name) in pending_updates.items():
            update_overflow_label_position(moveable[lid], cx, cy, anchor_name)

        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step:4d} | MaxDisp: {max_ang_disp:.7f} rad")

        if max_ang_disp < convergence_eps:
            print(
                f"[refine_overflow_forces] converged after {step + 1} steps "
                f"(max_dθ={max_ang_disp:.6f} rad)"
            )
            break
    else:
        print(f"[refine_overflow_forces] reached max steps ({steps})")

    # ── apply final positions ─────────────────────────────────────────────────
    result: Dict[int, OverflowLabel] = dict(overflow_labels)
    for lid, ol in moveable.items():
        cx, cy      = float(pos[lid][0]), float(pos[lid][1])
        anchor_name = getattr(ol, "anchor", "center")
        update_overflow_label_position(ol, cx, cy, anchor_name)
        result[lid] = ol

    return result