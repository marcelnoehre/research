"""
overflow.py  –  Force-Balanced Radar Labeling
=====================================================
Improved: Labels are now placed in the widest angular gap between
surrounding obstacles (edges, nodes, existing labels) rather than
greedily taking the first valid slot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases & Constants
# ---------------------------------------------------------------------------
Pt  = Tuple[float, float]
Seg = Tuple[Pt, Pt]
Box = Tuple[float, float, float, float]   # (xmin, ymin, xmax, ymax)

EXTENDED_PAD = 0.25

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def _box_of(cx: float, cy: float, w: float, h: float, pad: float = 0.05) -> Box:
    hw, hh = w / 2 + pad, h / 2 + pad
    return cx - hw, cy - hh, cx + hw, cy + hh

def _boxes_overlap(a: Box, b: Box) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

def _seg_intersects_box(seg: Seg, box: Box) -> bool:
    (x1, y1), (x2, y2) = seg
    xmin, ymin, xmax, ymax = box
    dx, dy = x2 - x1, y2 - y1
    p = (-dx,  dx, -dy,  dy)
    q = (x1 - xmin, xmax - x1, y1 - ymin, ymax - y1)
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0: return False
        elif pi < 0.0:
            t0 = max(t0, qi / pi)
        else:
            t1 = min(t1, qi / pi)
        if t0 > t1: return False
    return True

def _segs_cross(s1: Seg, s2: Seg) -> bool:
    (ax, ay), (bx, by) = s1
    (cx, cy), (dx, dy) = s2
    def _z(ox, oy, px, py, qx, qy) -> float:
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)
    d1 = _z(cx, cy, dx, dy, ax, ay)
    d2 = _z(cx, cy, dx, dy, bx, by)
    d3 = _z(ax, ay, bx, by, cx, cy)
    d4 = _z(ax, ay, bx, by, dx, dy)
    return (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))

# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------

@dataclass
class OverflowCandidate:
    node_id:    int
    label_type: str
    x:      float
    y:      float
    width:  float
    height: float
    angle:  float
    radius: float
    is_overflow: bool = True

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def center(self) -> Pt:
        """Required by plotting.py"""
        return (self.x, self.y)

    @property
    def bbox(self) -> Box:
        return _box_of(self.x, self.y, self.width, self.height)

    @property
    def bbox_corners(self):
        xmin, ymin, xmax, ymax = self.bbox
        return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))

    @property
    def expanded_bbox_corners(self):
        xmin, ymin, xmax, ymax = _box_of(self.x, self.y, self.width, self.height, EXTENDED_PAD)
        return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))

    @property
    def anchor(self) -> str:
        a = self.angle % (2 * math.pi)
        sectors = [
            (0 * math.pi / 4, 'left'), (1 * math.pi / 4, 'bottom_left'),
            (2 * math.pi / 4, 'bottom'), (3 * math.pi / 4, 'bottom_right'),
            (4 * math.pi / 4, 'right'), (5 * math.pi / 4, 'top_right'),
            (6 * math.pi / 4, 'top'), (7 * math.pi / 4, 'top_left'),
        ]
        best = min(sectors, key=lambda s: min(abs(a - s[0]), 2 * math.pi - abs(a - s[0])))
        return best[1]

    def get_anchor_point(self) -> Pt:
        name = self.anchor
        hw, hh = self.width / 2, self.height / 2
        mapping = {
            'left': (self.x - hw, self.y), 'right': (self.x + hw, self.y),
            'bottom': (self.x, self.y - hh), 'top': (self.x, self.y + hh),
            'bottom_left': (self.x - hw, self.y - hh), 'bottom_right': (self.x + hw, self.y - hh),
            'top_left': (self.x - hw, self.y + hh), 'top_right': (self.x + hw, self.y + hh),
        }
        return mapping.get(name, (self.x, self.y))

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _node_pos(G, node_id: int, coords: dict) -> Pt:
    if node_id in G.nodes:
        d = G.nodes[node_id]
        if 'pos' in d: return float(d['pos'][0]), float(d['pos'][1])
        for xk, yk in [('x','y'), ('pos_x','pos_y'), ('px','py')]:
            if xk in d and yk in d: return float(d[xk]), float(d[yk])
    if node_id in coords: return float(coords[node_id][0]), float(coords[node_id][1])
    raise KeyError(f"No position for node {node_id!r}")

def _node_radius(G, node_id: int, default: float) -> float:
    d = G.nodes[node_id]
    return float(d.get('r', d.get('radius', default)))

def _label_dims(node_id: int, label_texts: dict) -> Tuple[float, float]:
    for (nid, _), text in label_texts.items():
        if nid != node_id: continue
        lines = text.split('\n') if text else ['']
        w = max(max(len(l) for l in lines) * 0.18, 0.8)
        h = max(len(lines) * 0.55, 0.55)
        return w, h
    return 1.0, 0.5

def _get_candidate_bbox(c) -> Box:
    if hasattr(c, 'bbox'): return c.bbox
    return _box_of(getattr(c, 'x', 0.0), getattr(c, 'y', 0.0),
                   getattr(c, 'width', 0.0), getattr(c, 'height', 0.0))

# ---------------------------------------------------------------------------
# Angular gap scoring
# ---------------------------------------------------------------------------

def _get_blocker_wedges(
    node_id: int,
    nx: float,
    ny: float,
    G,
    coords: dict,
    nboxes: dict,
    placed_label_centers: List[Pt],
    node_r: float,
    blocker_radius: float = 8.0,
    edge_half_width: float = 0.35,
) -> List[Tuple[float, float]]:
    """
    Return a list of (lo, hi) angular wedges (in radians, [0, 2*pi)) that are
    'occupied' from the perspective of (nx, ny).

    Each blocker contributes a wedge rather than a single ray so that nearby
    wide obstacles block a realistic arc and thin gaps between dense edges
    are not falsely attractive.

    Wedge half-width rules
    ----------------------
    - Edge to a neighbour: half-width = edge_half_width rad (fixed, represents
      the physical width of a typical label that would sit on that bearing)
      plus an angular fattening based on proximity: closer neighbours get
      wider wedges because their edges dominate more of the visual field.
    - Nearby node: half-width proportional to node apparent radius at distance.
    - Placed label center: half-width = 0.4 rad (labels are wide objects).
    """
    TAU = 2 * math.pi
    wedges: List[Tuple[float, float]] = []

    def _add(bearing: float, half_w: float) -> None:
        bearing = bearing % TAU
        lo = (bearing - half_w) % TAU
        hi = (bearing + half_w) % TAU
        wedges.append((lo, hi))

    # --- Edge bearings ---
    if node_id in G.nodes:
        for u, v in G.edges(node_id):
            far = v if u == node_id else u
            try:
                fx, fy = _node_pos(G, far, coords)
            except KeyError:
                continue
            dist = math.hypot(fx - nx, fy - ny)
            bearing = math.atan2(fy - ny, fx - nx)
            # Closer neighbours subtend a larger apparent width; cap at pi/3
            proximity_factor = max(0.5, min(2.0, 3.0 / (dist + 1e-6)))
            half_w = min(math.pi / 3, edge_half_width * proximity_factor)
            _add(bearing, half_w)

    # --- Nearby node bearings ---
    for nid, _nbox in nboxes.items():
        if nid == node_id:
            continue
        try:
            ox, oy = _node_pos(G, nid, coords)
        except KeyError:
            continue
        dist = math.hypot(ox - nx, oy - ny)
        if dist > blocker_radius:
            continue
        bearing = math.atan2(oy - ny, ox - nx)
        apparent_r = _node_radius(G, nid, node_r) if nid in G.nodes else node_r
        half_w = min(math.pi / 2, math.atan2(apparent_r, max(dist, 0.1)))
        _add(bearing, half_w)

    # --- Already-placed label bearings ---
    for lx, ly in placed_label_centers:
        dist = math.hypot(lx - nx, ly - ny)
        if dist > blocker_radius:
            continue
        bearing = math.atan2(ly - ny, lx - nx)
        _add(bearing, 0.4)

    return wedges


def _free_arc_at(candidate_angle: float, wedges: List[Tuple[float, float]]) -> float:
    """
    Compute the total un-blocked arc (in radians) centred on candidate_angle.

    Strategy: build a 1-D coverage map on [0, 2*pi) from the wedges, then
    find the contiguous free interval that contains candidate_angle.
    Returns 0 if the angle is itself inside a blocked wedge.
    """
    TAU = 2 * math.pi
    ca = candidate_angle % TAU

    if not wedges:
        return TAU

    # Collect all coverage events
    events: List[Tuple[float, int]] = []
    for lo, hi in wedges:
        if lo < hi:
            events.append((lo, +1))
            events.append((hi, -1))
        else:
            # Wrap-around wedge: covers [lo, TAU) ∪ [0, hi)
            events.append((lo, +1))
            events.append((TAU, -1))
            events.append((0.0, +1))
            events.append((hi, -1))

    events.sort()

    # Sweep to find free intervals
    depth = 0
    prev_a = 0.0
    free_intervals: List[Tuple[float, float]] = []
    for a, delta in events:
        if depth == 0 and a > prev_a:
            free_intervals.append((prev_a, a))
        depth += delta
        prev_a = a
    # Trailing free interval to TAU
    if depth == 0 and prev_a < TAU:
        free_intervals.append((prev_a, TAU))

    # Find which free interval contains ca
    for lo, hi in free_intervals:
        if lo <= ca < hi:
            return hi - lo

    return 0.0  # ca is inside a blocked wedge


def _score_candidate(
    c: OverflowCandidate,
    wedges: List[Tuple[float, float]],
    preferred_radius: float,
    gap_weight: float = 1.0,
    distance_penalty: float = 0.6,
) -> float:
    """
    Score a placement candidate. Higher is better.

    - gap_score:        free arc the candidate sits in (0 to 2*pi)
    - distance_penalty: strong penalty for radius beyond the minimum —
                        prevents interior nodes from getting long-leader
                        placements in thin but technically-open far directions.
    """
    gap = _free_arc_at(c.angle, wedges)
    # Quadratic distance penalty so distant placements are strongly discouraged
    extra = max(0.0, c.radius - preferred_radius)
    dist_penalty = distance_penalty * extra * (1.0 + extra)
    return gap_weight * gap - dist_penalty

# ---------------------------------------------------------------------------
# Placement Core
# ---------------------------------------------------------------------------

def _is_valid(cx, cy, lw, lh, nx, ny, nboxes, pboxes, pleaders, esegs, pad, own_nid):
    lbox = _box_of(cx, cy, lw, lh, pad)
    if any(_boxes_overlap(lbox, nb) for nb in nboxes.values()): return False
    if any(_boxes_overlap(lbox, pb) for pb in pboxes): return False
    if any(_seg_intersects_box(es, lbox) for es in esegs): return False

    leader = ((nx, ny), (cx, cy))
    for nid, nb in nboxes.items():
        if nid != own_nid and _seg_intersects_box(leader, nb): return False
    if any(_seg_intersects_box(leader, pb) for pb in pboxes): return False
    if any(_segs_cross(leader, pl) for pl in pleaders): return False
    return True


def radar_scan_node(
    node_id, G, coords, label_texts,
    nboxes, pboxes, pleaders, esegs,
    n_angles, radii, node_r, pad,
    placed_label_centers: Optional[List[Pt]] = None,
    gap_weight: float = 1.0,
    distance_penalty: float = 0.6,
    blocker_radius: float = 8.0,
    edge_half_width: float = 0.35,
) -> Optional[OverflowCandidate]:
    """
    Scan all angle x radius combinations for node_id, collect every valid
    candidate, score each by how well it sits in the widest free arc between
    surrounding edge wedges / nodes / labels, and return the best one.

    Key improvements over the original greedy approach
    --------------------------------------------------
    1. Collects ALL valid placements (innermost valid radius per angle).
    2. Scores by *free arc width* — using wedge-based blocking so nearby
       dense edges occlude a realistic angular band, not just a single ray.
       This prevents thin gaps between packed edges from looking attractive.
    3. Quadratic distance penalty strongly discourages long leaders on
       interior nodes where every direction is somewhat obstructed.
    """
    if placed_label_centers is None:
        placed_label_centers = []

    nx, ny = _node_pos(G, node_id, coords)
    lw, lh = _label_dims(node_id, label_texts)
    r_node = _node_radius(G, node_id, node_r)
    half_diag = math.hypot(lw / 2, lh / 2) + pad
    min_r = r_node + half_diag + pad
    preferred_r = min_r

    # Pre-compute blocker wedges once for this node
    wedges = _get_blocker_wedges(
        node_id, nx, ny, G, coords, nboxes,
        placed_label_centers, node_r, blocker_radius, edge_half_width,
    )

    valid_candidates: List[OverflowCandidate] = []
    angles = [2 * math.pi * i / n_angles for i in range(n_angles)]

    for angle in angles:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for r in radii:
            if r < min_r:
                continue
            cx, cy = nx + r * cos_a, ny + r * sin_a
            if _is_valid(cx, cy, lw, lh, nx, ny, nboxes, pboxes, pleaders, esegs, pad, node_id):
                valid_candidates.append(
                    OverflowCandidate(node_id, 'general', cx, cy, lw, lh, angle, r)
                )
                # Keep only the innermost valid radius per angle direction
                break

    if not valid_candidates:
        return None

    best = max(
        valid_candidates,
        key=lambda c: _score_candidate(c, wedges, preferred_r,
                                       gap_weight, distance_penalty),
    )
    return best


def place_overflow_labels(
    G, coords, label_candidates, label_texts, lattice_nodes,
    placed_flags=None, n_angles=32, radii=None, node_r=0.3,
    label_padding=0.2, force_iterations=40,
    gap_weight: float = 2.0,
    distance_penalty: float = 0.1,
    blocker_radius: float = 8.0,
    edge_half_width: float = 0.35,
):
    """
    Place overflow labels using gap-aware radar scanning followed by
    force-balancing refinement.

    Keyword arguments
    -----------------
    gap_weight : float
        Relative importance of the free-arc score vs. distance penalty.
        Increase (e.g. 2.0) to bias even more strongly toward open space.
    distance_penalty : float
        Quadratic cost per unit of extra radius beyond the minimum.
        The default (0.6) strongly discourages long leaders; increase
        further if interior nodes are still escaping to distant positions.
    blocker_radius : float
        How far away (graph units) to search for nearby nodes / labels
        when computing blocker wedges.
    edge_half_width : float
        Base half-width (radians) of the angular wedge each edge occupies.
        Larger values make edges block more of the angular field;
        proximity scaling further widens wedges for near neighbours.
    """
    if radii is None:
        radii = [k * 0.4 for k in range(1, 30)]

    all_nboxes = {
        nid: _box_of(*_node_pos(G, nid, coords),
                     _node_radius(G, nid, node_r) * 2,
                     _node_radius(G, nid, node_r) * 2)
        for nid in lattice_nodes if nid in G.nodes
    }
    all_esegs = [
        (_node_pos(G, u, coords), _node_pos(G, v, coords))
        for u, v in G.edges()
        if u in G.nodes and v in G.nodes
    ]

    overflow_map: Dict[int, OverflowCandidate] = {}
    todo = [
        nid for nid in lattice_nodes
        if (not placed_flags.get(nid) if placed_flags else not label_candidates.get(nid))
    ]

    for nid in todo:
        pboxes = [_get_candidate_bbox(c) for cands in label_candidates.values() for c in cands]
        pboxes += [c.bbox for c in overflow_map.values()]

        # Pass already-placed label centers so they count as angular blockers
        placed_centers = [c.center for c in overflow_map.values()]

        cand = radar_scan_node(
            nid, G, coords, label_texts,
            all_nboxes, pboxes, [],
            all_esegs, n_angles, radii, node_r, label_padding,
            placed_label_centers=placed_centers,
            gap_weight=gap_weight,
            distance_penalty=distance_penalty,
            blocker_radius=blocker_radius,
            edge_half_width=edge_half_width,
        )
        if cand:
            overflow_map[nid] = cand

    # Force Balancing
    dt = 0.08
    for _ in range(force_iterations):
        deltas = {}
        for nid, c in overflow_map.items():
            fx, fy = 0.0, 0.0
            nx, ny = _node_pos(G, nid, coords)

            # Spring (Attraction to origin node)
            dx, dy = nx - c.x, ny - c.y
            dist = math.hypot(dx, dy)
            if dist > 0:
                fx += (dx / dist) * (dist * 0.4)
                fy += (dy / dist) * (dist * 0.4)

            # Repulsion (Even spacing from other overflow labels)
            for oid, oc in overflow_map.items():
                if nid == oid:
                    continue
                rx, ry = c.x - oc.x, c.y - oc.y
                r_dist_sq = rx ** 2 + ry ** 2 + 0.1
                fx += (rx / r_dist_sq) * 0.15
                fy += (ry / r_dist_sq) * 0.15
            deltas[nid] = (fx * dt, fy * dt)

        for nid, (dx, dy) in deltas.items():
            c = overflow_map[nid]
            nx, ny = _node_pos(G, nid, coords)
            new_x, new_y = c.x + dx, c.y + dy
            others = [oc.bbox for oid, oc in overflow_map.items() if oid != nid]
            if _is_valid(new_x, new_y, c.width, c.height, nx, ny, all_nboxes,
                         others, [], all_esegs, label_padding, nid):
                c.x, c.y = new_x, new_y
                c.angle = math.atan2(new_y - ny, new_x - nx)
                c.radius = math.hypot(new_y - ny, new_x - nx)

    binding_lines = {}
    for nid, c in overflow_map.items():
        label_candidates[nid] = [c]
        nx, ny = _node_pos(G, nid, coords)
        binding_lines[nid] = ((nx, ny), c.get_anchor_point())

    return label_candidates, binding_lines


def draw_binding_lines(ax, binding_lines, style=None):
    s = {'color': '#555555', 'lw': 0.6, 'ls': (0, (4, 3)), 'dot_size': 3}
    if style: s.update(style)
    for seg in binding_lines.values():
        if not seg: continue
        (x0, y0), (ax_x, ax_y) = seg
        ax.plot([x0, ax_x], [y0, ax_y], color=s['color'], lw=s['lw'], linestyle=s['ls'], zorder=2)
        ax.plot(ax_x, ax_y, 'o', color=s['color'], ms=s['dot_size'], zorder=3)