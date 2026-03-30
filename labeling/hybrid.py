from label import LabelCandidate
from typing import Dict, List, Optional, Set, Tuple

def _inner_boxes_overlap(a: LabelCandidate, b: LabelCandidate) -> bool:
    '''
    True iff the ink (inner) bounding boxes of two candidates intersect.
    '''
    (ax0, ay0), _, (ax1, ay1), _ = a.inner_bbox_corners   # BL, BR, TR, TL
    (bx0, by0), _, (bx1, by1), _ = b.inner_bbox_corners
    # touching is not a conflict
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


# candidate = (node_id, label_type, cand_idx_within_type).
# shorter alias CKey = Tuple[int, str, int].
CKey = Tuple[int, str, int]

# feature = (node_id, label_type)
FKey = Tuple[int, str]

def _split_by_type(
        candidates: Dict[int, List[LabelCandidate]],
) -> Dict[FKey, List[LabelCandidate]]:
    '''
    Re-index candidates by (node_id, label_type).

    Returns
    -------
    typed : (node_id, label_type)
        ordered list of LabelCandidate (index within this list is the cand_idx used in CKey)
    '''
    typed: Dict[FKey, List[LabelCandidate]] = {}
    for nid, cands in candidates.items():
        for c in cands:
            fk = (nid, c.label_type)
            typed.setdefault(fk, []).append(c)
    return typed


def _build_conflict_index(
        typed: Dict[FKey, List[LabelCandidate]],
) -> Dict[CKey, List[CKey]]:
    '''
    For each candidate CKey return all CKeys whose ink box overlaps it.

    Two candidates conflict iff:
      - they belong to different features (different (node_id, label_type) pairs)
      - their inner bounding boxes strictly intersect
    '''
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
            # same feature - never a conflict
            if (ni, lti) == (nj, ltj):
                continue
            if _inner_boxes_overlap(ca, cb):
                conflicts[cki].append(ckj)
                conflicts[ckj].append(cki)

    return conflicts

################################################################################
# Phase I rules
################################################################################
def _live_conflicts(
        ck: CKey,
        conflicts: Dict[CKey, List[CKey]],
        active: Dict[CKey, bool],
        chosen: Dict[FKey, Optional[int]],
) -> List[CKey]:
    '''
    Conflicts of ck that are still active and whose feature is unresolved.
    '''
    return [
        qk for qk in conflicts[ck]
        if active[qk] and chosen[(qk[0], qk[1])] is None
    ]

def _apply_L1(typed, active, chosen, conflicts, log) -> bool:
    '''
    L1: candidate with zero live conflicts = choose it immediately.
    '''
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
    '''
    L2: candidate ca conflicts only with cb of feature q, and q has another
    candidate cy that conflicts only with candidates of feature p = choose both.
    '''
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
                    # choose fk-ci, qfk-kj
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
    '''
    L3: feature has exactly one active candidate and its conflicting
    candidates form a clique = choose it.
    '''
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
            # deactivate only the other candidates of THIS feature
            for k in range(len(cands)):
                if k != ci:
                    active[(nid, lt, k)] = False
            log.append(
                f"  L3  ({nid},{lt}) cand {ci} → chosen "
                f"({len(live)} live conflict(s) form a clique)"
            )
            changed = True
    return changed


################################################################################
# Phase II helpers
################################################################################
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
    '''
    Kakoulis-Tollis heuristic: recursively reduce a component to cliques.
    '''
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
    '''
    Augmenting-path bipartite matching: features <-> cliques.
    '''
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

def hybrid_label_placement(
        candidates: Dict[int, List[LabelCandidate]],
        verbose: bool = True,
) -> Tuple[Dict[int, List[LabelCandidate]], List[str]]:
    '''
    Resolve ink-box conflicts among remaining label candidates using the
    Hybrid algorithm from Wolff 1999, §3.2.1.

    General (general), Extent (object), and intent (attribute) labels are treated as fully
    independent features. Conflicts are detected only between candidates of different
    features whose ink boxes strictly intersect.

    Phase I  — exhaustive L1 / L2 / L3
    Phase II — Kakoulis-Tollis clique reduction + max bipartite matching

    Parameters
    ----------
    candidates : List
        node_id, list of LabelCandidate
    verbose : bool 
        print the execution log to stdout

    Returns
    -------
    chosen_map : List
        node_id, list of chosen LabelCandidates
    log : List
        list of decision strings for debugging
    '''
    log: List[str] = []

    # Split candidates by (node_id, label_type)
    typed = _split_by_type(candidates)

    # active[(node_id, label_type, cand_idx)] = bool
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
        f"{sum(1 for fk in typed if fk[1]=='intent')} intent, "
        f"{sum(1 for fk in typed if fk[1]=='general')} general)  "
        f"Candidates: {total_cands}"
    )

    # Phase I
    log.append('=== Phase I: L1 / L2 / L3 ===')
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
    log.append(f'Phase I done after {iteration} pass(es)')

    # Phase II
    log.append('=== Phase II: KT heuristic + bipartite matching ===')

    unresolved_keys: List[CKey] = [
        (nid, lt, ci)
        for (nid, lt), cands in typed.items()
        if chosen[(nid, lt)] is None
        for ci in range(len(cands))
        if active[(nid, lt, ci)]
    ]

    if not unresolved_keys:
        log.append('  No unresolved candidates — Phase II skipped')
    else:
        comps = _connected_components_keys(unresolved_keys, conflicts)
        log.append(f'  Connected components: {len(comps)}')

        all_cliques: List[List[CKey]] = []
        for comp in comps:
            all_cliques.extend(_kt_reduce(comp, typed, conflicts, active))
        log.append(f'  Cliques after KT reduction: {len(all_cliques)}')

        unresolved_features: List[FKey] = [
            fk for fk, cands in typed.items()
            if chosen[fk] is None
            and any(active[(fk[0], fk[1], ci)] for ci in range(len(cands)))
        ]
        feature_to_clique = _maximum_bipartite_matching(unresolved_features, all_cliques)
        log.append(
            f'  Matched: {len(feature_to_clique)} / {len(unresolved_features)} features'
        )

        for fk, clique_idx in feature_to_clique.items():
            clique = all_cliques[clique_idx]
            matches = [
                ci for (ni, lt, ci) in clique
                if (ni, lt) == fk and active[(ni, lt, ci)]
            ]
            if matches:
                chosen[fk] = matches[0]
                log.append(f'  ({fk[0]},{fk[1]}) - cand {matches[0]} (matched)')

        for fk in unresolved_features:
            if fk not in feature_to_clique:
                log.append(f'  ({fk[0]},{fk[1]}) - UNLABELED (unmatched)')

    # Build result map (node_id - list of chosen candidates
    chosen_map: Dict[int, List[LabelCandidate]] = {
        nid: [] for nid in candidates
    }
    for (nid, lt), cands in typed.items():
        ci = chosen[(nid, lt)]
        if ci is not None and ci < len(cands):
            chosen_map[nid].append(cands[ci])
        else:
            # if exactly one active candidate remains and it doesn't overlap any *chosen* candidate
            active_cands = [
                (i, c) for i, c in enumerate(cands) if active[(nid, lt, i)]
            ]
            if len(active_cands) == 1:
                sole_i, sole_c = active_cands[0]
                # Build the set of chosen candidates from other features
                chosen_candidates = [
                    typed[ofk][oci]
                    for ofk, oci in chosen.items()
                    if oci is not None and ofk != (nid, lt)
                ]
                overlaps_chosen = any(
                    _inner_boxes_overlap(sole_c, other)
                    for other in chosen_candidates
                )
                if not overlaps_chosen:
                    chosen_map[nid].append(sole_c)
                    log.append(
                        f'  fallback ({nid},{lt}) cand {sole_i} → chosen '
                        f'(sole active, no overlap with chosen labels)'
                    )

    n_extent  = sum(1 for lst in chosen_map.values() for c in lst if c.label_type == 'extent')
    n_intent  = sum(1 for lst in chosen_map.values() for c in lst if c.label_type == 'intent')
    n_general = sum(1 for lst in chosen_map.values() for c in lst if c.label_type == 'general')
    log.append(f'Result: {n_extent} extent labels, {n_intent} intent labels, {n_general} general labels placed')

    if verbose:
        for line in log:
            print(line)

    return chosen_map, log