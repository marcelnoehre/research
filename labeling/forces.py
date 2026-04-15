"""
refine_overflow_forces.py

Post-assignment physics refinement for outer overflow labels.

After the Hungarian global assignment phase has placed every label, small
residual overlaps can remain. This module runs a lightweight force-directed
simulation to nudge only the labels in `outer_overflow_ids` into cleaner
positions while keeping them outside the boundary polygon.

Physics model
─────────────
  Repulsion   : Every moveable label repels every other moveable label AND
                every fixed label_candidate box. Uses a soft inverse-square
                law so forces remain bounded.

  Elastic binder ("soft spring")
              : Each moveable label is tethered to its source node by a
                nonlinear spring that has *low* tension near an ideal length
                (SPRING_IDEAL ~20 units) but rises steeply beyond
                SPRING_SLACK_MAX to prevent drift. The transition is
                piecewise:
                   |d - ideal| ≤ slack   →  k_soft  * deviation
                   |d - ideal| > slack   →  k_hard  * deviation²
                This keeps the binder "soft" at comfortable distances while
                acting as a hard backstop against infinite drift.

  Boundary cushion
              : Uses outer_polygon.distance() to detect proximity. Within
                CUSHION_RADIUS of the boundary a strong repulsive force
                pushes the label away. Beyond that radius there is no force.

Validity
────────
  After every step the binder is re-checked with binding_line_valid(soft=True)
  so labels are never allowed to drift into a topologically invalid state.
  If a proposed step would invalidate the binder the step is dropped for
  that label this tick (the label stays put).

Tunables (module-level constants)
──────────────────────────────────
  REFINE_DT            time step (world units per iteration)
  REFINE_STEPS         maximum simulation iterations
  REFINE_DAMP          velocity damping factor per step (0–1)
  REPULSION_STRENGTH   scalar for label-label / label-fixed repulsion
  REPULSION_MIN_DIST   soft floor on repulsion denominator (avoids ÷0)
  SPRING_IDEAL         preferred binder length (world units)
  SPRING_SLACK         half-width of the "comfortable zone"
  SPRING_K_SOFT        spring constant inside the comfortable zone
  SPRING_K_HARD        spring constant outside the comfortable zone
  CUSHION_RADIUS       distance from polygon at which cushion force starts
  CUSHION_STRENGTH     scalar for cushion repulsion
  CONVERGENCE_EPS      stop early when max displacement < this value
"""

from __future__ import annotations

import math
import copy
from typing import Dict, List, Tuple

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
# Tangential-orbit model: each label slides around its source node at a
# fixed radius (= its current binder length). Repulsion is projected onto
# the tangent so labels never drift radially away from the drawing.

REFINE_DT           = 0.01   # angular step scale (radians per unit force)
REFINE_STEPS        = 1000    # max iterations
REFINE_DAMP         = 0.7   # angular velocity damping per step

REPULSION_STRENGTH  = 100.0    # label-label tangential repulsion scalar
REPULSION_MIN_DIST  = 1.0    # soft floor on repulsion distance (world units)

# Radial correction: small restoring nudge if the label has drifted off its
# original binder length (e.g. due to bbox snapping). Keep this weak.
RADIAL_K            = 0.10   # spring constant pulling back to original radius
MAX_DTHETA          = 0.05   # roughly 3 degrees per step

# Boundary cushion: pushes label tangentially away from polygon corners.
CUSHION_RADIUS      = 1.0    # world units from boundary that triggers cushion
CUSHION_STRENGTH    = 2.0    # tangential nudge strength near boundary

CONVERGENCE_EPS     = 1e-7  # stop when max angular displacement < this (rad)


# ── internal helpers ──────────────────────────────────────────────────────────

def _label_center(ol: OverflowLabel) -> Tuple[float, float]:
    """Return the geometric centre (cx, cy) of an OverflowLabel."""
    bl, br, tr, tl = ol.inner_bbox_corners
    return (
        (bl[0] + br[0] + tr[0] + tl[0]) / 4.0,
        (bl[1] + br[1] + tr[1] + tl[1]) / 4.0,
    )


def _tangent_ccw(radial: np.ndarray) -> np.ndarray:
    """Unit tangent 90° counter-clockwise from a radial unit vector."""
    return np.array([-radial[1], radial[0]])


def _tangential_repulsion(
    pos_a: np.ndarray,
    node_a: np.ndarray,
    pos_b: np.ndarray,
    strength: float,
    min_dist: float,
) -> float:
    """
    Scalar tangential force on label A caused by label B.

    Projects the repulsion vector (A←B) onto the tangent of A's orbit
    around its own source node. This means B can only push A *sideways*
    around its orbit — never radially outward.

    Returns a signed scalar: positive = push CCW, negative = push CW.
    """
    delta = pos_a - pos_b
    dist  = float(np.linalg.norm(delta))
    if dist < 1e-9:
        delta = np.array([1.0, 0.0])
        dist  = 1.0

    d_eff      = max(dist, min_dist)
    repel_vec  = (strength / d_eff) * (delta / dist)   # 1/d falloff (gentler)

    # radial unit vector from node_a toward pos_a
    r_vec = pos_a - node_a
    r_len = float(np.linalg.norm(r_vec))
    if r_len < 1e-9:
        return 0.0
    radial  = r_vec / r_len
    tangent = _tangent_ccw(radial)

    return float(np.dot(repel_vec, tangent))   # signed tangential component


def _boundary_tangential_cushion(
    pos_a: np.ndarray,
    node_a: np.ndarray,
    outer_polygon,
    cushion_radius: float,
    cushion_strength: float,
) -> float:
    """
    If the label is within `cushion_radius` of the polygon boundary, produce
    a signed tangential nudge that pushes it away from the nearest boundary
    point — projected onto the orbit tangent, so it slides along the radius
    rather than drifting outward.
    """
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


def _fixed_label_centers(
    label_candidates: Dict[int, List[LabelCandidate]],
) -> List[np.ndarray]:
    """
    Collect centre-points of all fixed LabelCandidate boxes so the moveable
    labels are repelled from them.
    """
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
    outer_nodes:    List[int],
    G,
    *,
    dt:                 float = REFINE_DT,
    steps:              int   = REFINE_STEPS,
    damp:               float = REFINE_DAMP,
    repulsion_strength: float = REPULSION_STRENGTH,
    repulsion_min_dist: float = REPULSION_MIN_DIST,
    radial_k:           float = RADIAL_K,
    cushion_radius:     float = CUSHION_RADIUS,
    cushion_strength:   float = CUSHION_STRENGTH,
    convergence_eps:    float = CONVERGENCE_EPS,
) -> Dict[int, OverflowLabel]:
    """
    Refine the positions of `outer_overflow_ids` labels using a tangential
    orbit simulation after the global Hungarian assignment phase.

    Each moveable label orbits its source node at its *current* binder
    radius (locked from Phase 2). Repulsion between labels is projected onto
    the orbit tangent, so labels slide sideways to make room for each other
    rather than drifting radially away from the drawing. A weak radial
    corrector keeps the binder length stable if floating-point drift occurs.

    Parameters
    ----------
    overflow_labels    : full dict of all overflow labels (read-only for fixed ones)
    outer_overflow_ids : the subset of label IDs that are allowed to move
    label_candidates   : fixed internal labels (static obstacles for repulsion)
    outer_polygon      : Shapely Polygon defining the outer boundary
    G                  : NetworkX graph (used to look up node positions)

    Returns
    -------
    Updated dict of overflow_labels with refined positions for outer_overflow_ids.
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

    # Current centre positions — single source of truth for step-start geometry
    pos: Dict[int, np.ndarray] = {}
    for lid, ol in moveable.items():
        cx, cy = _label_center(ol)
        pos[lid] = np.array([cx, cy], dtype=float)

    # Source-node positions
    node_pos: Dict[int, np.ndarray] = {}
    for lid, ol in moveable.items():
        raw = G.nodes[ol.node_id]["pos"]
        node_pos[lid] = np.array(raw, dtype=float)

    # Lock each label's orbit radius to its current binder length
    orbit_radius: Dict[int, float] = {}
    for lid in moveable:
        r = float(np.linalg.norm(pos[lid] - node_pos[lid]))
        orbit_radius[lid] = max(r, 1e-3)   # never zero

    # Angular velocity (signed scalar, radians/step)
    ang_vel: Dict[int, float] = {lid: 0.0 for lid in moveable}

    # Fixed obstacle centres (inner label_candidates + static overflow labels)
    all_fixed_centers: List[np.ndarray] = _fixed_label_centers(label_candidates)
    for lid, ol in overflow_labels.items():
        if lid not in outer_overflow_ids:
            cx, cy = _label_center(ol)
            all_fixed_centers.append(np.array([cx, cy], dtype=float))

    moveable_ids = list(moveable.keys())

    # Pre-compute per-label dimensions once (they don't change between steps).
    # This ensures Guard 4 and the candidate polygon always use the same,
    # mutation-independent size, even before the deferred-update fix below.
    label_wh: Dict[int, Tuple[float, float]] = {
        lid: _label_wh(ol) for lid, ol in moveable.items()
    }

    # ── simulation loop ───────────────────────────────────────────────────────
    for step in range(steps):
        # Accumulate signed tangential torque for each moveable label
        torque: Dict[int, float] = {lid: 0.0 for lid in moveable_ids}

        # 1. Tangential repulsion: moveable ↔ moveable
        for i, lid_a in enumerate(moveable_ids):
            for lid_b in moveable_ids[i + 1:]:
                t_a = _tangential_repulsion(
                    pos[lid_a], node_pos[lid_a], pos[lid_b],
                    repulsion_strength, repulsion_min_dist,
                )
                # Reaction on B: repulsion from A, projected on B's tangent
                t_b = _tangential_repulsion(
                    pos[lid_b], node_pos[lid_b], pos[lid_a],
                    repulsion_strength, repulsion_min_dist,
                )
                torque[lid_a] += t_a
                torque[lid_b] += t_b

        # 2. Tangential repulsion: moveable ← fixed obstacles
        for lid in moveable_ids:
            for fc in all_fixed_centers:
                torque[lid] += _tangential_repulsion(
                    pos[lid], node_pos[lid], fc,
                    repulsion_strength, repulsion_min_dist,
                )

        # 3. Boundary cushion (tangential nudge near polygon edges)
        for lid in moveable_ids:
            torque[lid] += _boundary_tangential_cushion(
                pos[lid], node_pos[lid], outer_polygon,
                cushion_radius, cushion_strength,
            )

        # 4. Integrate angular velocity → new angle → new Cartesian position
        max_ang_disp = 0.0

        # Collect all accepted candidate positions before mutating any object.
        # Keys present   → label accepted the move this step.
        # Keys absent    → label stays put (ang_vel zeroed below).
        pending_pos:     Dict[int, np.ndarray] = {}   # new centre positions
        pending_updates: Dict[int, Tuple[float, float, str]] = {}  # (cx, cy, anchor)

        for lid in moveable_ids:
            ol = moveable[lid]
            r_ideal = orbit_radius[lid]

            # Physics integration
            ang_vel[lid] = (ang_vel[lid] + torque[lid] * dt) * damp
            dtheta = max(-MAX_DTHETA, min(MAX_DTHETA, ang_vel[lid] * dt))

            # Current polar state (always derived from pos[], never from ol)
            r_vec = pos[lid] - node_pos[lid]
            theta = math.atan2(float(r_vec[1]), float(r_vec[0]))

            # Proposed Cartesian candidate
            theta_new = theta + dtheta
            current_r = float(np.linalg.norm(r_vec))
            r_new     = current_r + radial_k * (r_ideal - current_r)

            candidate = node_pos[lid] + r_new * np.array([
                math.cos(theta_new), math.sin(theta_new)
            ])

            # --- Validation Geometry ---
            # Use the pre-computed, immutable dimensions — never read from ol
            # at this point, because ol may have been mutated by a previous
            # step's deferred update.
            tw, th = label_wh[lid]
            anchor_name = getattr(ol, "anchor", "center")
            cand_anchor_pts = _anchor_points(candidate[0], candidate[1], tw, th)
            anchor_pt = cand_anchor_pts[anchor_name]

            candidate_binder = LineString([anchor_pt, node_pos[lid].tolist()])
            candidate_poly   = _label_bbox_polygon(candidate[0], candidate[1], tw, th)

            valid = True

            # Guard 1: Directional (ensure label stays 'outward')
            node_outward = node_pos[lid] - np.array([
                outer_polygon.centroid.x, outer_polygon.centroid.y
            ])
            label_outward = candidate - node_pos[lid]
            if np.dot(node_outward, label_outward) <= 0:
                valid = False

            # Guard 2: Topology (checks against other moveable labels using
            # their step-start state — moveable objects are not yet mutated)
            if valid:
                synthetic_placed = []
                for other_lid, other_ol in moveable.items():
                    if other_lid == lid:
                        continue
                    op = pos[other_lid]   # step-start centre, never the mutated object
                    synthetic_placed.append({
                        "label_id":    other_lid,
                        "position":    (float(op[0]), float(op[1])),
                        "binding_line": getattr(other_ol, "binding_line", None),
                    })

                # Also include fixed overflow labels (non-moveable)
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
                    synthetic_placed,       # ← was [], now has real obstacles
                    {**moveable, **{k: v for k, v in overflow_labels.items()
                                    if k not in outer_overflow_ids}},
                    label_candidates,
                    soft=True
                ):
                    valid = False

            # Guard 3: Collision with static / fixed geometry
            if valid and candidate_poly.intersects(placed_union):
                valid = False

            # Guard 4: Collision with other moveable labels.
            # Use pos[] (step-start positions) together with pre-computed
            # label_wh[] so that no other label's object state is read.
            # This prevents the chimera geometry that occurred when earlier
            # labels in the same step had already been mutated in-place.
            if valid:
                for other_lid in moveable_ids:
                    if other_lid == lid:
                        continue
                    op = pos[other_lid]          # step-start position (stable)
                    otw, oth = label_wh[other_lid]  # immutable dimensions
                    other_poly = _label_bbox_polygon(op[0], op[1], otw, oth)
                    if candidate_poly.intersects(other_poly):
                        valid = False
                        break

            # --- Stage Move ---
            if valid:
                pending_pos[lid]     = candidate
                pending_updates[lid] = (float(candidate[0]), float(candidate[1]), anchor_name)
                max_ang_disp = max(max_ang_disp, abs(dtheta))
            else:
                pending_pos[lid] = pos[lid]   # no movement
                ang_vel[lid]     = 0.0

        # --- SYNC STEP ---
        # Commit all accepted moves at once, *after* every label has been
        # evaluated.  This guarantees that no label's validation saw a
        # neighbour in a partially-updated state (the root cause of the
        # invalid-position bug).
        pos.update(pending_pos)
        for lid, (cx, cy, anchor_name) in pending_updates.items():
            update_overflow_label_position(moveable[lid], cx, cy, anchor_name)

        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step} | MaxDisp: {max_ang_disp:.6f}")

        # 5. Early convergence check
        if max_ang_disp < convergence_eps:
            print(f"[refine_overflow_forces] converged after {step + 1} steps "
                  f"(max_dθ={max_ang_disp:.5f} rad)")
            break
    else:
        print(f"[refine_overflow_forces] reached max steps ({steps})")

    # ── apply final positions ─────────────────────────────────────────────────
    result: Dict[int, OverflowLabel] = dict(overflow_labels)
    for lid, ol in moveable.items():
        cx, cy = float(pos[lid][0]), float(pos[lid][1])
        anchor_name = getattr(ol, "anchor", "center")
        update_overflow_label_position(ol, cx, cy, anchor_name)
        result[lid] = ol

    return result