"""
placement.py
------------
For each concept node, computes 4 candidate label positions using a
fixed-position approach.

Each candidate is defined by anchoring one corner of the label's outer
bounding box to the node's position. The 4 corners are:
    top-left, top-right, bottom-left, bottom-right

All coordinates are in normalised graph units. Conversion to mm is done
via the mm_per_unit factor derived from the physical drawing height.

Requires normalize_positions() to have been called on G first.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from shapely.geometry import LineString, Point, box

from label import measure_ink_mm


@dataclass
class LabelCandidate:
    """
    One candidate placement for a node label.

    Attributes
    ----------
    anchor               : which corner of the outer bbox is placed at the node position
    label_type           : 'extent' (objects, below node) or 'intent' (attributes, above node)
    bbox_corners         : outer bbox (BL, BR, TR, TL) in graph units
    inner_bbox_corners   : inner (ink) bbox (BL, BR, TR, TL) in graph units
    expanded_bbox_corners: outer bbox expanded on the two free sides for node exclusion check
    center               : center of the outer bbox in graph units
    """
    anchor: str
    label_type: str                                           # 'extent' | 'intent'
    bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]
    inner_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]
    expanded_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]
    center: Tuple[float, float]


def compute_label_candidates(
        G: nx.Graph,
        concepts: List[int],
        label_text: str,
        physical_height_mm: float,
        label_type: str = 'extent',
        padding_x_mm: float = 3.0,
        padding_y_mm: float = 2.0,
        fontsize_pt: float = 10.0,
) -> Dict[int, List[LabelCandidate]]:
    """
    For every concept node, return candidate label placements.

    Parameters
    ----------
    G                  : graph with normalised 'pos' attributes
    concepts           : list of concept node ids
    label_text         : the label string (used to measure ink size)
    physical_height_mm : actual drawing height in mm (sets the unit scale)
    label_type         : 'extent' → only bottom anchors (objects, below node)
                         'intent' → only top anchors    (attributes, above node)
    padding_x_mm       : horizontal padding around the ink box
    padding_y_mm       : vertical padding around the ink box
    fontsize_pt        : font size in points

    Returns
    -------
    dict mapping concept node id → list of LabelCandidate objects
    """
    if 'normalized_height' not in G.graph:
        raise ValueError("Call normalize_positions(G) before compute_label_candidates().")

    # Restrict anchors by label type
    if label_type == 'extent':
        allowed_anchors = {'top_left', 'top_right'}      # objects above node, anchor at bottom
    elif label_type == 'intent':
        allowed_anchors = {'bottom_left', 'bottom_right'} # attributes below node, anchor at top
    else:
        raise ValueError(f"label_type must be 'extent' or 'intent', got '{label_type}'")

    mm_per_unit = physical_height_mm / G.graph['normalized_height']
    units_per_mm = 1.0 / mm_per_unit

    # Measure label in mm, convert to graph units
    ink_w_mm, ink_h_mm = measure_ink_mm(label_text, fontsize_pt)

    outer_w_mm = ink_w_mm + 2 * padding_x_mm
    outer_h_mm = ink_h_mm + 2 * padding_y_mm

    half_w  = (outer_w_mm * units_per_mm) / 2.0
    half_h  = (outer_h_mm * units_per_mm) / 2.0
    # Inner half-extents (ink only, no padding)
    half_iw = (ink_w_mm  * units_per_mm) / 2.0
    half_ih = (ink_h_mm  * units_per_mm) / 2.0

    anchor_offsets: Dict[str, Tuple[float, float]] = {
        'top_left':     ( half_w, -half_h),
        'top_right':    (-half_w, -half_h),
        'bottom_left':  ( half_w,  half_h),
        'bottom_right': (-half_w,  half_h),
    }

    # Expansion = 1 * padding in each free direction, doubling the total gap
    exp_x = (padding_x_mm * units_per_mm)
    exp_y = (padding_y_mm * units_per_mm)

    _expand_factors: Dict[str, Tuple[float, float, float, float]] = {
        'bottom_right': (exp_x, 0,     0,     exp_y),
        'bottom_left':  (0,     exp_x, 0,     exp_y),
        'top_right':    (exp_x, 0,     exp_y, 0    ),
        'top_left':     (0,     exp_x, exp_y, 0    ),
    }

    candidates: Dict[int, List[LabelCandidate]] = {}

    for node in concepts:
        node_x, node_y = G.nodes[node]['pos']

        node_candidates: List[LabelCandidate] = []
        for anchor, (dx, dy) in anchor_offsets.items():
            if anchor not in allowed_anchors:
                continue
            cx = node_x + dx
            cy = node_y + dy

            bl = (cx - half_w,  cy - half_h)
            br = (cx + half_w,  cy - half_h)
            tr = (cx + half_w,  cy + half_h)
            tl = (cx - half_w,  cy + half_h)

            ibl = (cx - half_iw, cy - half_ih)
            ibr = (cx + half_iw, cy - half_ih)
            itr = (cx + half_iw, cy + half_ih)
            itl = (cx - half_iw, cy + half_ih)

            dl, dr, db, dt = _expand_factors[anchor]
            ebl = (bl[0] - dl, bl[1] - db)
            etr = (tr[0] + dr, tr[1] + dt)
            ebr = (etr[0], ebl[1])
            etl = (ebl[0], etr[1])

            node_candidates.append(LabelCandidate(
                anchor=anchor,
                label_type=label_type,
                bbox_corners=(bl, br, tr, tl),
                inner_bbox_corners=(ibl, ibr, itr, itl),
                expanded_bbox_corners=(ebl, ebr, etr, etl),
                center=(cx, cy),
            ))

        candidates[node] = node_candidates

    return candidates


def restrict_outer_node_candidates(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        outer_nodes: List,
        top_node: int,
        bottom_node: int,
) -> Dict[int, List[LabelCandidate]]:
    """
    For nodes on the outer face, restrict candidates to those pointing
    away from the interior of the diagram:

      - leftmost outer nodes  → keep only right-side anchors (top_right, bottom_right)
      - rightmost outer nodes → keep only left-side anchors  (top_left,  bottom_left)
      - top node (id=top_node)    → keep only top anchors    (top_left,  top_right)
      - bottom node (id=bottom_node) → keep only bottom anchors (bottom_left, bottom_right)
      - corner nodes get the intersection of both rules

    Non-outer nodes are left unchanged.

    Parameters
    ----------
    G           : graph with normalised 'pos' attributes
    candidates  : output of compute_label_candidates
    outer_nodes : ordered node-list of the outer face
    top_node    : node id of the top of the lattice (e.g. 0)
    bottom_node : node id of the bottom of the lattice (e.g. max id)
    """
    if not outer_nodes:
        return candidates

    outer_set = set(outer_nodes)

    # Determine left/right split from x-coordinates of outer nodes
    xs = [G.nodes[n]['pos'][0] for n in outer_nodes if isinstance(n, int)]
    if not xs:
        return candidates
    x_mid = (min(xs) + max(xs)) / 2.0

    restricted = dict(candidates)  # shallow copy — we replace lists, not mutate

    for node, node_candidates in candidates.items():
        if node not in outer_set:
            continue

        allowed: set[str] = {'top_left', 'top_right', 'bottom_left', 'bottom_right'}

        if node == top_node:
            allowed &= {'bottom_left', 'bottom_right'}
        elif node == bottom_node:
            allowed &= {'top_left', 'top_right'}
        else:
            x = G.nodes[node]['pos'][0]
            if x <= x_mid:
                allowed &= {'top_right', 'bottom_right'}   # left side → point right
            else:
                allowed &= {'top_left', 'bottom_left'}     # right side → point left

        restricted[node] = [c for c in node_candidates if c.anchor in allowed]

    return restricted


def filter_candidates_by_nodes(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        concepts: List[int],
) -> Dict[int, List[LabelCandidate]]:
    """
    Remove candidates whose expanded outer bbox contains any other concept node.
    The expanded bbox grows the two free sides (away from anchor) by 2x the bbox size.
    """
    filtered: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        surviving = []
        for candidate in node_candidates:
            ebl, ebr, etr, etl = candidate.expanded_bbox_corners
            expanded = box(ebl[0], ebl[1], etr[0], etr[1])
            occupied = any(
                expanded.contains(Point(G.nodes[other]['pos']))
                for other in concepts
                if other != node and other in G.nodes
            )
            if not occupied:
                surviving.append(candidate)
        filtered[node] = surviving

    return filtered


def _face_edges(face: List) -> Set[frozenset]:
    """Return the set of undirected edges for a face."""
    n = len(face)
    return {frozenset((face[i], face[(i + 1) % n])) for i in range(n)}


def _node_to_face_edges(
        node: int,
        bounded_faces: List[List],
        outer_nodes: List,
) -> List[Tuple]:
    """
    Collect all unique edges from faces that contain `node`,
    including the outer face. Returns a list of (u, v) tuples.
    """
    all_faces = bounded_faces + [outer_nodes]
    seen: Set[frozenset] = set()
    edges = []
    for face in all_faces:
        if node not in face:
            continue
        for e in _face_edges(face):
            if e not in seen:
                seen.add(e)
                u, v = tuple(e)
                edges.append((u, v))
    return edges


def filter_candidates_by_neighbor_direction(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        lattice,
        top_node: int,
        bottom_node: int,
) -> Dict[int, List[LabelCandidate]]:
    """
    For nodes that still have 2 candidates of the same label_type, use the
    direction of cover edges to keep only the non-blocked one.

    Extent (top_* anchors, label above node):
        - check children (lower neighbors)
        - only right child  → keep top_right  (anchor on right, label extends left — clear)
        - only left child   → keep top_left   (anchor on left,  label extends right — clear)
        - both sides        → skip (keep both)

    Intent (bottom_* anchors, label below node):
        - check parents (upper neighbors)
        - only right parent → keep bottom_right
        - only left parent  → keep bottom_left
        - both sides        → skip (keep both)

    Groups with != 2 candidates are left unchanged.
    """
    result: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        if not node_candidates:
            result[node] = node_candidates
            continue

        node_x = G.nodes[node]['pos'][0]

        by_type: Dict[str, List[LabelCandidate]] = {}
        for c in node_candidates:
            by_type.setdefault(c.label_type, []).append(c)

        kept = []
        for ltype, group in by_type.items():
            if len(group) != 2:
                kept.extend(group)
                continue

            if ltype == 'extent' and node != top_node:
                neighbors = lattice.children(node)
            elif ltype == 'intent' and node != bottom_node:
                neighbors = lattice.parents(node)
            else:
                kept.extend(group)
                continue

            has_left  = any(G.nodes[nb]['pos'][0] < node_x for nb in neighbors)
            has_right = any(G.nodes[nb]['pos'][0] > node_x for nb in neighbors)

            if has_left and not has_right:
                # edge goes left → keep anchor on left (label extends right, away from edge)
                kept.extend(c for c in group if c.anchor.endswith('left'))
            elif has_right and not has_left:
                # edge goes right → keep anchor on right (label extends left, away from edge)
                kept.extend(c for c in group if c.anchor.endswith('right'))
            else:
                # both or neither — skip
                kept.extend(group)

        result[node] = kept

    return result


def filter_candidates_by_edges(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        bounded_faces: List[List],
        outer_nodes: List,
        skip_nodes: set = None,
) -> Dict[int, List[LabelCandidate]]:
    """
    Remove candidates whose inner bounding box (ink area, no padding)
    is crossed by any edge of the faces the node belongs to.

    Parameters
    ----------
    G             : graph with 'pos' attributes on every node
    candidates    : output of compute_label_candidates
    bounded_faces : bounded face node-lists from extract_faces
    outer_nodes   : outer face node-list from extract_faces
    skip_nodes    : node ids to skip edge filtering entirely (e.g. top/bottom)

    Returns
    -------
    dict with same keys, each value being the surviving candidates
    """
    skip_nodes = skip_nodes or set()
    filtered: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        if node in skip_nodes:
            filtered[node] = node_candidates
            continue

        incident_edges = _node_to_face_edges(node, bounded_faces, outer_nodes)
        lines = [
            LineString([G.nodes[u]['pos'], G.nodes[v]['pos']])
            for u, v in incident_edges
        ]

        surviving = []
        for candidate in node_candidates:
            ibl, ibr, itr, itl = candidate.inner_bbox_corners
            inner_shape = box(ibl[0], ibl[1], itr[0], itr[1])
            if not any(inner_shape.intersects(line) for line in lines):
                surviving.append(candidate)

        filtered[node] = surviving

    return filtered


# ---------------------------------------------------------------------------
# Hybrid label-placement  (Wolff 1999, §3.2.1)
# ---------------------------------------------------------------------------
#
# Design note — two independent label types per node
# ---------------------------------------------------
# Each concept node can carry two completely separate labels:
#   - 'extent'  : object names, anchored at top  (label sits above node)
#   - 'intent'  : attribute names, anchored at bottom (label sits below node)
#
# These two types never conflict with each other geometrically (one is always
# above the node, the other always below), so they must be treated as
# independent features.  The unit of resolution is therefore the pair
#   feature_key = (node_id, label_type)   e.g. (3, 'extent')
#
# The algorithm is run once over ALL candidates of ALL nodes and types
# simultaneously — conflicts can still occur between candidates of the same
# type on different nodes (e.g. two 'extent' labels that overlap), and the
# conflict index below captures exactly those.

def _inner_boxes_overlap(a: LabelCandidate, b: LabelCandidate) -> bool:
    """True iff the ink (inner) bounding boxes of two candidates intersect."""
    (ax0, ay0), _, (ax1, ay1), _ = a.inner_bbox_corners   # BL, BR, TR, TL
    (bx0, by0), _, (bx1, by1), _ = b.inner_bbox_corners
    # strict: touching is not a conflict
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


# A candidate is identified by (node_id, label_type, cand_idx_within_type).
# We use the shorter alias CKey = Tuple[int, str, int].
CKey = Tuple[int, str, int]

# A feature is (node_id, label_type) — the thing we want to assign exactly
# one candidate to.
FKey = Tuple[int, str]


def _split_by_type(
        candidates: Dict[int, List[LabelCandidate]],
) -> Dict[FKey, List[LabelCandidate]]:
    """
    Re-index candidates by (node_id, label_type).

    Returns
    -------
    typed : (node_id, label_type) → ordered list of LabelCandidate
            (index within this list is the cand_idx used in CKey)
    """
    typed: Dict[FKey, List[LabelCandidate]] = {}
    for nid, cands in candidates.items():
        for c in cands:
            fk = (nid, c.label_type)
            typed.setdefault(fk, []).append(c)
    return typed


def _build_conflict_index(
        typed: Dict[FKey, List[LabelCandidate]],
) -> Dict[CKey, List[CKey]]:
    """
    For each candidate CKey return all CKeys whose ink box overlaps it.

    Two candidates conflict iff:
      - they belong to different features (different (node_id, label_type) pairs)
      - their inner bounding boxes strictly intersect

    Same-node, same-type candidates can conflict with each other only
    if they somehow geometrically overlap, which is geometrically impossible
    for the four-position model (they're at opposite sides of the node).
    The check is kept general for safety.
    """
    flat: List[Tuple[CKey, LabelCandidate]] = [
        ((nid, lt, ci), c)
        for (nid, lt), cands in typed.items()
        for ci, c in enumerate(cands)
    ]
    conflicts: Dict[CKey, List[CKey]] = {ck: [] for ck, _ in flat}

    for i in range(len(flat)):
        cki, ca = flat[i]
        ni, lti, _ = cki
        for j in range(i + 1, len(flat)):
            ckj, cb = flat[j]
            nj, ltj, _ = ckj
            # same feature → never a conflict
            if (ni, lti) == (nj, ltj):
                continue
            if _inner_boxes_overlap(ca, cb):
                conflicts[cki].append(ckj)
                conflicts[ckj].append(cki)

    return conflicts


# ── Phase I rules ─────────────────────────────────────────────────────────────

def _live_conflicts(
        ck: CKey,
        conflicts: Dict[CKey, List[CKey]],
        active: Dict[CKey, bool],
        chosen: Dict[FKey, Optional[int]],
) -> List[CKey]:
    """Conflicts of ck that are still active and whose feature is unresolved."""
    return [
        qk for qk in conflicts[ck]
        if active[qk] and chosen[(qk[0], qk[1])] is None
    ]


def _apply_L1(typed, active, chosen, conflicts, log) -> bool:
    """L1: candidate with zero live conflicts → choose it immediately."""
    changed = False
    for fk, cands in typed.items():
        if chosen[fk] is not None:
            continue
        nid, lt = fk
        for ci, _ in enumerate(cands):
            ck = (nid, lt, ci)
            if not active[ck]:
                continue
            if len(_live_conflicts(ck, conflicts, active, chosen)) == 0:
                chosen[fk] = ci
                # deactivate all other candidates of this feature
                for k in range(len(cands)):
                    if k != ci:
                        active[(nid, lt, k)] = False
                log.append(f"  L1  ({nid},{lt}) cand {ci} → chosen (no conflicts)")
                changed = True
                break
    return changed


def _apply_L2(typed, active, chosen, conflicts, log) -> bool:
    """
    L2: candidate cₐ conflicts only with cᵦ of feature q, and q has another
    candidate cᵧ that conflicts only with candidates of feature p → choose both.
    """
    changed = False
    for fk, cands in typed.items():
        if chosen[fk] is not None:
            continue
        nid, lt = fk
        for ci, _ in enumerate(cands):
            ck = (nid, lt, ci)
            if not active[ck]:
                continue
            live = _live_conflicts(ck, conflicts, active, chosen)
            if len(live) != 1:
                continue
            qk = live[0]
            qfk = (qk[0], qk[1])
            if chosen[qfk] is not None:
                continue
            # look for another candidate of qfk that only conflicts with fk
            for kj, _ in enumerate(typed[qfk]):
                qkj = (qfk[0], qfk[1], kj)
                if qkj == qk or not active[qkj]:
                    continue
                kj_live = _live_conflicts(qkj, conflicts, active, chosen)
                if all((pk[0], pk[1]) == fk for pk in kj_live):
                    # choose fk→ci, qfk→kj
                    chosen[fk] = ci
                    for k in range(len(cands)):
                        if k != ci:
                            active[(nid, lt, k)] = False
                    if chosen[qfk] is None:
                        chosen[qfk] = kj
                        for k in range(len(typed[qfk])):
                            if k != kj:
                                active[(qfk[0], qfk[1], k)] = False
                    log.append(
                        f"  L2  ({nid},{lt}) cand {ci}"
                        f" + ({qfk[0]},{qfk[1]}) cand {kj} → chosen"
                    )
                    changed = True
                    break
            if chosen[fk] is not None:
                break
    return changed


def _apply_L3(typed, active, chosen, conflicts, log) -> bool:
    """
    L3: feature has exactly one active candidate and its conflicting
    candidates form a clique → choose it, deactivate the clique members.
    """
    changed = False
    for fk, cands in typed.items():
        if chosen[fk] is not None:
            continue
        nid, lt = fk
        active_here = [ci for ci in range(len(cands)) if active[(nid, lt, ci)]]
        if len(active_here) != 1:
            continue
        ci = active_here[0]
        ck = (nid, lt, ci)
        live = _live_conflicts(ck, conflicts, active, chosen)
        is_clique = all(
            _inner_boxes_overlap(
                typed[(live[a][0], live[a][1])][live[a][2]],
                typed[(live[b][0], live[b][1])][live[b][2]],
            )
            for a in range(len(live))
            for b in range(a + 1, len(live))
        )
        if is_clique:
            chosen[fk] = ci
            for k in range(len(cands)):
                if k != ci:
                    active[(nid, lt, k)] = False
            for qk in live:
                active[qk] = False
            log.append(
                f"  L3  ({nid},{lt}) cand {ci} → chosen; "
                f"deactivated {len(live)} clique member(s)"
            )
            changed = True
    return changed


# ── Phase II helpers ──────────────────────────────────────────────────────────

def _connected_components_keys(
        keys: List[CKey],
        conflicts: Dict[CKey, List[CKey]],
) -> List[List[CKey]]:
    key_set = set(keys)
    visited: Set[CKey] = set()
    comps = []
    for k in keys:
        if k in visited:
            continue
        comp, queue = [], [k]
        visited.add(k)
        while queue:
            cur = queue.pop()
            comp.append(cur)
            for nb in conflicts.get(cur, []):
                if nb in key_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        comps.append(comp)
    return comps


def _is_clique_keys(
        keys: List[CKey],
        typed: Dict[FKey, List[LabelCandidate]],
) -> bool:
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ni, lti, ci = keys[i]
            nj, ltj, cj = keys[j]
            if not _inner_boxes_overlap(typed[(ni, lti)][ci], typed[(nj, ltj)][cj]):
                return False
    return True


def _kt_reduce(
        comp: List[CKey],
        typed: Dict[FKey, List[LabelCandidate]],
        conflicts: Dict[CKey, List[CKey]],
        active: Dict[CKey, bool],
        depth: int = 0,
) -> List[List[CKey]]:
    """Kakoulis-Tollis heuristic: recursively reduce a component to cliques."""
    if not comp:
        return []
    if _is_clique_keys(comp, typed):
        return [comp]

    subset = set(comp)

    def is_important(k: CKey) -> bool:
        nid, lt, _ = k
        return sum(1 for ci in range(len(typed[(nid, lt)])) if active[(nid, lt, ci)]) == 1

    def deg(k: CKey) -> int:
        return sum(1 for nb in conflicts.get(k, []) if nb in subset)

    best = max(comp, key=lambda k: (deg(k), not is_important(k)))
    if is_important(best):
        best_deg = deg(best)
        alt = next(
            (k for k in comp
             if k != best and not is_important(k) and deg(k) >= best_deg - 1),
            None,
        )
        if alt is not None:
            best = alt

    remaining = [k for k in comp if k != best]
    if not remaining or depth >= 200:
        return [remaining] if remaining else []

    sub_comps = _connected_components_keys(remaining, conflicts)
    result = []
    for sc in sub_comps:
        result.extend(_kt_reduce(sc, typed, conflicts, active, depth + 1))
    return result


def _maximum_bipartite_matching(
        features: List[FKey],
        cliques: List[List[CKey]],
) -> Dict[FKey, int]:
    """Augmenting-path bipartite matching: features ↔ cliques."""
    fadj: List[List[int]] = [
        [ci for ci, cl in enumerate(cliques)
         if any((k[0], k[1]) == fk for k in cl)]
        for fk in features
    ]
    match_l = [-1] * len(features)
    match_r = [-1] * len(cliques)

    def dfs(u: int, vis: List[bool]) -> bool:
        for v in fadj[u]:
            if vis[v]:
                continue
            vis[v] = True
            if match_r[v] == -1 or dfs(match_r[v], vis):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in range(len(features)):
        dfs(u, [False] * len(cliques))

    return {features[u]: match_l[u] for u in range(len(features)) if match_l[u] != -1}


# ── Public entry point ────────────────────────────────────────────────────────

def hybrid_label_placement(
        candidates: Dict[int, List[LabelCandidate]],
        verbose: bool = True,
) -> Tuple[Dict[int, List[LabelCandidate]], List[str]]:
    """
    Resolve ink-box conflicts among remaining label candidates using the
    Hybrid algorithm from Wolff 1999, §3.2.1.

    Extent (object) and intent (attribute) labels are treated as fully
    independent features: a node can win both, one, or neither.
    Conflicts are detected only between candidates of different features
    whose ink boxes strictly intersect.

    Phase I  — exhaustive L1 / L2 / L3
    Phase II — Kakoulis-Tollis clique reduction + max bipartite matching

    Parameters
    ----------
    candidates : node_id → list of LabelCandidate
                 (output of the upstream filter pipeline in placement.py;
                 may contain both 'extent' and 'intent' entries per node)
    verbose    : print the execution log to stdout

    Returns
    -------
    chosen_map : node_id → list of chosen LabelCandidates
                 (0, 1, or 2 entries depending on how many types were resolved)
    log        : list of decision strings for inspection / debugging
    """
    log: List[str] = []

    # Split candidates by (node_id, label_type)
    typed = _split_by_type(candidates)

    # active[(node_id, label_type, cand_idx)] = True/False
    active: Dict[CKey, bool] = {
        (nid, lt, ci): True
        for (nid, lt), cands in typed.items()
        for ci in range(len(cands))
    }
    # chosen[(node_id, label_type)] = cand_idx or None
    chosen: Dict[FKey, Optional[int]] = {fk: None for fk in typed}

    conflicts = _build_conflict_index(typed)

    total_cands = sum(len(v) for v in typed.values())
    log.append(
        f"Features: {len(typed)}  "
        f"({sum(1 for fk in typed if fk[1]=='extent')} extent, "
        f"{sum(1 for fk in typed if fk[1]=='intent')} intent)  "
        f"Candidates: {total_cands}"
    )

    # ── Phase I ───────────────────────────────────────────────────────────
    log.append("=== Phase I: L1 / L2 / L3 ===")
    iteration, changed = 0, True
    while changed and iteration < 500:
        changed = False
        if _apply_L1(typed, active, chosen, conflicts, log):
            changed = True
        if _apply_L2(typed, active, chosen, conflicts, log):
            changed = True
        if _apply_L3(typed, active, chosen, conflicts, log):
            changed = True
        iteration += 1
    log.append(f"Phase I done after {iteration} pass(es)")

    # ── Phase II ──────────────────────────────────────────────────────────
    log.append("=== Phase II: KT heuristic + bipartite matching ===")

    unresolved_keys: List[CKey] = [
        (nid, lt, ci)
        for (nid, lt), cands in typed.items()
        if chosen[(nid, lt)] is None
        for ci in range(len(cands))
        if active[(nid, lt, ci)]
    ]

    if not unresolved_keys:
        log.append("  No unresolved candidates — Phase II skipped")
    else:
        comps = _connected_components_keys(unresolved_keys, conflicts)
        log.append(f"  Connected components: {len(comps)}")

        all_cliques: List[List[CKey]] = []
        for comp in comps:
            all_cliques.extend(_kt_reduce(comp, typed, conflicts, active))
        log.append(f"  Cliques after KT reduction: {len(all_cliques)}")

        unresolved_features: List[FKey] = [
            fk for fk, cands in typed.items()
            if chosen[fk] is None
            and any(active[(fk[0], fk[1], ci)] for ci in range(len(cands)))
        ]
        feature_to_clique = _maximum_bipartite_matching(unresolved_features, all_cliques)
        log.append(
            f"  Matched: {len(feature_to_clique)} / {len(unresolved_features)} features"
        )

        for fk, clique_idx in feature_to_clique.items():
            clique = all_cliques[clique_idx]
            matches = [
                ci for (ni, lt, ci) in clique
                if (ni, lt) == fk and active[(ni, lt, ci)]
            ]
            if matches:
                chosen[fk] = matches[0]
                log.append(f"  ({fk[0]},{fk[1]}) → cand {matches[0]} (matched)")

        for fk in unresolved_features:
            if fk not in feature_to_clique:
                log.append(f"  ({fk[0]},{fk[1]}) → UNLABELED (unmatched)")

    # ── Build result map ──────────────────────────────────────────────────
    # Return node_id → list of chosen candidates (one per label_type that won)
    chosen_map: Dict[int, List[LabelCandidate]] = {
        nid: [] for nid in candidates
    }
    for (nid, lt), cands in typed.items():
        ci = chosen[(nid, lt)]
        if ci is not None and ci < len(cands):
            chosen_map[nid].append(cands[ci])
        else:
            # fallback: single remaining active candidate with no live conflicts
            remaining = [
                c for i, c in enumerate(cands)
                if active[(nid, lt, i)]
                and len(_live_conflicts((nid, lt, i), conflicts, active, chosen)) == 0
            ]
            if len(remaining) == 1:
                chosen_map[nid].append(remaining[0])

    n_extent = sum(1 for lst in chosen_map.values() for c in lst if c.label_type == 'extent')
    n_intent = sum(1 for lst in chosen_map.values() for c in lst if c.label_type == 'intent')
    log.append(f"Result: {n_extent} extent labels, {n_intent} intent labels placed")

    if verbose:
        for line in log:
            print(line)

    return chosen_map, log

# ---------------------------------------------------------------------------
# Overflow label placement — labels that were not placed inside the diagram
# ---------------------------------------------------------------------------

@dataclass
class OverflowLabel:
    """
    A label that could not be placed inside the diagram and is instead
    positioned outside the bounding box of all nodes.

    Attributes
    ----------
    node_id    : concept node this label belongs to
    label_type : 'extent' or 'intent'
    center     : (x, y) center of the label box in graph units
    half_w     : half-width  of the ink box in graph units
    half_h     : half-height of the ink box in graph units
    node_pos   : (x, y) of the corresponding concept node
    """
    node_id:    int
    label_type: str
    center:     Tuple[float, float]
    half_w:     float
    half_h:     float
    node_pos:   Tuple[float, float]


def place_overflow_labels(
        G: nx.Graph,
        label_candidates: Dict[int, List[LabelCandidate]],
        chosen_labels:    Dict[int, List[LabelCandidate]],
        label_texts:      Dict,
        physical_height_mm: float,
        fontsize_pt:      float,
        padding_x_mm:     float = 3.0,
        padding_y_mm:     float = 2.0,
        margin_units:     float = 0.2,
) -> List['OverflowLabel']:
    """
    Place unresolved labels outside the diagram.

    Strategy
    --------
    1. Each overflow label gets an angle = midpoint of the gap between its two
       neighbouring chosen labels (or evenly subdivided if multiple labels share
       a gap).  This is purely angular — no preferred-angle bias.
    2. The initial radius for each label is interpolated from its two
       neighbouring chosen labels' radii.
    3. A collision post-pass pushes labels outward radially (one at a time,
       smallest-radius first) until no two ink boxes overlap and no overflow
       label's ink box overlaps any chosen label's ink box.
    """
    import math

    def _normalise_into(a: float, base: float) -> float:
        while a < base:            a += 2 * math.pi
        while a >= base + 2*math.pi: a -= 2 * math.pi
        return a

    def _ink_box(cx, cy, hw, hh):
        """Return (x0, y0, x1, y1) of the ink box."""
        return cx - hw, cy - hh, cx + hw, cy + hh

    def _boxes_overlap(b1, b2):
        """True iff two (x0,y0,x1,y1) boxes strictly overlap (touching is fine)."""
        return b1[0] < b2[2] and b1[2] > b2[0] and b1[1] < b2[3] and b1[3] > b2[1]

    mm_per_unit  = physical_height_mm / G.graph['normalized_height']
    units_per_mm = 1.0 / mm_per_unit

    concept_nodes = [n for n in G.nodes if isinstance(n, int)]
    xs = [G.nodes[n]['pos'][0] for n in concept_nodes]
    ys = [G.nodes[n]['pos'][1] for n in concept_nodes]
    cx_diag = (min(xs) + max(xs)) / 2
    cy_diag = (min(ys) + max(ys)) / 2

    # ── Collect unplaced labels ───────────────────────────────────────────
    chosen_types: Dict[int, set] = {}
    for nid, clist in chosen_labels.items():
        chosen_types[nid] = {c.label_type for c in clist}

    unplaced: List[Tuple[int, str]] = []
    for (nid, lt), text in label_texts.items():
        if text and lt not in chosen_types.get(nid, set()):
            unplaced.append((nid, lt))

    if not unplaced:
        return []

    # ── Measure all unplaced labels ───────────────────────────────────────
    # half_w/half_h here are the INK half-extents (no padding)
    measured = []
    for nid, lt in unplaced:
        text = label_texts[(nid, lt)]
        ink_w_mm, ink_h_mm = measure_ink_mm(text, fontsize_pt)
        half_w = (ink_w_mm / 2) * units_per_mm
        half_h = (ink_h_mm / 2) * units_per_mm
        node_pos = G.nodes[nid]['pos']
        pref_angle = math.atan2(node_pos[1] - cy_diag, node_pos[0] - cx_diag)
        measured.append((nid, lt, half_w, half_h, node_pos, pref_angle))

    # ── Build chosen-label angle index ───────────────────────────────────
    chosen_info = []   # (angle, center, half_w_ink, half_h_ink) for each chosen label
    for clist in chosen_labels.values():
        for c in clist:
            a = math.atan2(c.center[1] - cy_diag, c.center[0] - cx_diag)
            r = math.hypot(c.center[0] - cx_diag, c.center[1] - cy_diag)
            ibl, _, itr, _ = c.inner_bbox_corners
            ihw = (itr[0] - ibl[0]) / 2
            ihh = (itr[1] - ibl[1]) / 2
            chosen_info.append({'angle': a, 'radius': r, 'center': c.center,
                                 'half_w': ihw, 'half_h': ihh})

    # ── Assign angles: subdivide each gap between chosen labels ───────────
    if not chosen_info:
        # No chosen labels — evenly space around ring at outermost node radius
        concept_positions = [G.nodes[n]['pos'] for n in concept_nodes]
        n_total = len(measured)
        result = []
        for idx, (nid, lt, hw, hh, node_pos, _) in enumerate(measured):
            angle = -math.pi + 2 * math.pi * idx / n_total
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            projs = [(p[0]-cx_diag)*cos_a + (p[1]-cy_diag)*sin_a
                     for p in concept_positions]
            r = max(projs) + hw + hh
            result.append(OverflowLabel(
                node_id=nid, label_type=lt,
                center=(cx_diag + cos_a*r, cy_diag + sin_a*r),
                half_w=hw, half_h=hh, node_pos=node_pos,
            ))
        return result

    sorted_chosen = sorted(chosen_info, key=lambda x: x['angle'])
    n_ch = len(sorted_chosen)

    # gaps[i] = (a_start, a_end, r_start, r_end) — angle and radius at each boundary
    gaps = []
    for i in range(n_ch):
        ci0 = sorted_chosen[i]
        ci1 = sorted_chosen[(i + 1) % n_ch]
        a0, a1 = ci0['angle'], ci1['angle']
        if i == n_ch - 1:
            a1 += 2 * math.pi
        gaps.append((a0, a1, ci0['radius'], ci1['radius']))

    # Assign each overflow label to the gap whose midpoint is angularly closest
    # to the label's preferred angle.
    gap_buckets: Dict[int, List[Tuple[int, float]]] = {gi: [] for gi in range(len(gaps))}
    for idx, (nid, lt, hw, hh, node_pos, pref_angle) in enumerate(measured):
        best_gi, best_dist = 0, math.inf
        for gi, (g0, g1, _, _) in enumerate(gaps):
            mid = math.atan2(math.sin((g0+g1)/2), math.cos((g0+g1)/2))
            d = abs(math.atan2(math.sin(pref_angle - mid),
                                math.cos(pref_angle - mid)))
            if d < best_dist:
                best_dist = d
                best_gi = gi
        gap_buckets[best_gi].append((idx, _normalise_into(pref_angle, gaps[best_gi][0])))

    # Within each gap, place labels at evenly-spaced angles (n+1 slots, interior only)
    assigned: Dict[int, Tuple[float, float, float]] = {}  # idx → (angle, r_interp, gi)
    for gi, (g0, g1, r0, r1) in enumerate(gaps):
        bucket = gap_buckets[gi]
        if not bucket:
            continue
        bucket.sort(key=lambda x: x[1])
        n = len(bucket)
        step = (g1 - g0) / (n + 1)
        for k, (idx, _) in enumerate(bucket):
            angle = g0 + step * (k + 1)
            t = (angle - g0) / (g1 - g0)
            r_interp = r0 + t * (r1 - r0)
            assigned[idx] = (math.atan2(math.sin(angle), math.cos(angle)),
                             r_interp, gi)

    # ── Build initial positions ───────────────────────────────────────────
    # Each label center starts at (cx + cos*r, cy + sin*r)
    entries = []   # mutable list of dicts for the push-out pass
    for idx, (nid, lt, hw, hh, node_pos, pref_angle) in enumerate(measured):
        angle, r, gi = assigned.get(idx, (pref_angle, 2.0, 0))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        cx = cx_diag + cos_a * r
        cy = cy_diag + sin_a * r
        entries.append({
            'idx': idx, 'nid': nid, 'lt': lt,
            'hw': hw, 'hh': hh,
            'node_pos': node_pos,
            'angle': angle,
            'r': r,
            'cx': cx, 'cy': cy,
        })

    # ── Collision post-pass: push outward radially ────────────────────────
    # Build ink boxes of chosen labels for overlap testing
    chosen_boxes = [
        _ink_box(ci['center'][0], ci['center'][1], ci['half_w'], ci['half_h'])
        for ci in chosen_info
    ]

    # Repeat until no overlaps or max iterations reached
    for _iter in range(60):
        moved = False
        # Process in order of increasing radius (innermost first)
        entries.sort(key=lambda e: e['r'])
        for i, e in enumerate(entries):
            box_i = _ink_box(e['cx'], e['cy'], e['hw'], e['hh'])
            needs_push = False

            # Check against chosen labels
            for cb in chosen_boxes:
                if _boxes_overlap(box_i, cb):
                    needs_push = True
                    break

            # Check against other overflow labels already processed
            if not needs_push:
                for j, other in enumerate(entries):
                    if j == i:
                        continue
                    box_j = _ink_box(other['cx'], other['cy'], other['hw'], other['hh'])
                    if _boxes_overlap(box_i, box_j):
                        needs_push = True
                        break

            if needs_push:
                # Push outward by the overlap amount + tiny clearance
                # Compute minimum push needed along the radial direction
                cos_a, sin_a = math.cos(e['angle']), math.sin(e['angle'])
                push = max(e['hw'] * 2, e['hh'] * 2) * 0.5 + 0.05
                e['r']  += push
                e['cx'] = cx_diag + cos_a * e['r']
                e['cy'] = cy_diag + sin_a * e['r']
                moved = True

        if not moved:
            break

    # ── Build result ──────────────────────────────────────────────────────
    result = []
    for e in sorted(entries, key=lambda x: x['idx']):
        result.append(OverflowLabel(
            node_id=e['nid'], label_type=e['lt'],
            center=(e['cx'], e['cy']),
            half_w=e['hw'], half_h=e['hh'],
            node_pos=e['node_pos'],
        ))
    return result