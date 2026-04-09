from label import LabelCandidate
from typing import Dict, List, Optional, Set, Tuple

def _boxes_overlap(a: LabelCandidate, b: LabelCandidate) -> bool:
    '''
    True iff the ink (inner) bounding boxes of two candidates intersect.
    '''
    (ax0, ay0), _, (ax1, ay1), _ = a.bbox_corners   # BL, BR, TR, TL
    (bx0, by0), _, (bx1, by1), _ = b.bbox_corners
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
            if _boxes_overlap(ca, cb):
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
                log.append(f'  L1  ({nid},{lt}) cand {ci} → chosen (no conflicts)')
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
                        f'  L2  ({nid},{lt}) cand {ci}'
                        f' + ({qfk[0]},{qfk[1]}) cand {kj} → chosen'
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
            _boxes_overlap(
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
                f'  L3  ({nid},{lt}) cand {ci} -> chosen '
                f'({len(live)} live conflict(s) form a clique)'
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
            if not _boxes_overlap(typed[(ni, lti)][ci], typed[(nj, ltj)][cj]):
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
    Modified Hybrid algorithm: Preserves alternatives. 
    Candidates of the same feature are allowed to overlap each other, 
    but will still block candidates from DIFFERENT features.
    '''
    log: List[str] = []
    typed = _split_by_type(candidates)

    active: Dict[CKey, bool] = {
        (nid, lt, ci): True
        for (nid, lt), cands in typed.items()
        for ci in range(len(cands))
    }
    
    chosen: Dict[FKey, Optional[int]] = {fk: None for fk in typed}
    conflicts = _build_conflict_index(typed)

    # Phase I: Pruning
    def _apply_L1_keep_alts(typed, active, chosen, conflicts, log) -> bool:
        changed = False
        for fk, cands in typed.items():
            if chosen[fk] is not None: continue
            for ci, _ in enumerate(cands):
                ck = (fk[0], fk[1], ci)
                if not active[ck]: continue
                if len(_live_conflicts(ck, conflicts, active, chosen)) == 0:
                    chosen[fk] = ci
                    log.append(f'  L1  {fk} cand {ci} is safe')
                    changed = True
                    break
        return changed

    iteration, changed = 0, True
    while changed and iteration < 100:
        changed = _apply_L1_keep_alts(typed, active, chosen, conflicts, log)
        iteration += 1

    # Phase II: Clique reduction
    unresolved_keys = [k for k, v in active.items() if v and chosen[(k[0], k[1])] is None]
    if unresolved_keys:
        comps = _connected_components_keys(unresolved_keys, conflicts)
        all_cliques = []
        for comp in comps:
            all_cliques.extend(_kt_reduce(comp, typed, conflicts, active))
        
        unresolved_features = [fk for fk, c in chosen.items() if c is None]
        match = _maximum_bipartite_matching(unresolved_features, all_cliques)
        for fk, clique_idx in match.items():
            clique = all_cliques[clique_idx]
            for (ni, lt, ci) in clique:
                if (ni, lt) == fk:
                    chosen[fk] = ci
                    break

    # FINAL CONSTRUCTION: Greedy Collector with Feature-Awareness
    chosen_map: Dict[int, List[LabelCandidate]] = {nid: [] for nid in candidates}
    
    # Store tuples of (LabelCandidate, FKey) to know where each box came from
    placed_entries: List[Tuple[LabelCandidate, FKey]] = []

    def is_blocked(cand: LabelCandidate, current_fk: FKey) -> bool:
        for p_cand, p_fk in placed_entries:
            # If they are from DIFFERENT features, check for overlap
            if current_fk != p_fk:
                if _boxes_overlap(cand, p_cand):
                    return True
        return False

    # 1. Place "Chosen" candidates first
    for fk, ci in chosen.items():
        if ci is not None:
            cand = typed[fk][ci]
            if not is_blocked(cand, fk):
                chosen_map[fk[0]].append(cand)
                placed_entries.append((cand, fk))

    # 2. Place all other active alternatives
    for (nid, lt, ci), is_active in active.items():
        if not is_active: continue
        fk = (nid, lt)
        cand = typed[fk][ci]
        
        # Don't add the same object twice
        if cand in chosen_map[nid]: continue
        
        if not is_blocked(cand, fk):
            chosen_map[nid].append(cand)
            placed_entries.append((cand, fk))
            log.append(f'  Added alternative: {fk} index {ci}')

    if verbose:
        for line in log: print(line)
        print(f'Total labels placed: {len(placed_entries)}')

    return chosen_map, log