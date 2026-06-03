#!/usr/bin/env python3
"""
poset.py  —  Eurovision voting-bloc succession poset
=====================================================

Pipeline
--------
1. Divide the full dataset into fixed-width time windows (WINDOW_SIZE years).
2. For each window, aggregate votes and cluster countries with Ward's linkage
   (single largest-gap cut).
3. Build a succession relation across consecutive windows:
     (A, t) → (B, t+1)  iff  |A ∩ B| / min(|A|,|B|)  ≥  OVERLAP_THRESHOLD
4. Render the resulting DAG as a Hasse line diagram (eras = columns,
   left → right = earlier → later).  Nodes = blocs; edges = nation carry-over.

Tune WINDOW_SIZE, OVERLAP_THRESHOLD, MIN_BLOC_SIZE, MAX_BLOC_SIZE at the top.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import graphviz

from topology import fetch_all, YEARS

# ── Tuneable parameters ───────────────────────────────────────────────────────
WINDOW_SIZE          = 5     # years per era window
OVERLAP_THRESHOLD    = 0.4   # min overlap coefficient for a succession edge
MIN_BLOC_SIZE        = 3     # smallest bloc kept (no upper limit — natural size)
MIN_ERA_APPEARANCES  = 2     # min years a country must have competed in the window
MAX_SINGLETON_FRAC   = 0.15  # retry cut if singletons exceed this fraction of countries
OUTPUT               = "eurovision_succession_poset"


# ── Step 1: slice year data into windows ─────────────────────────────────────
def make_windows(
    year_data: Dict[int, pd.DataFrame]
) -> "OrderedDict[str, Tuple[pd.DataFrame, Dict[str, int]]]":
    """
    Return an ordered dict  era_label → (aggregated DataFrame, year_counts)
    for each WINDOW_SIZE-year slice that contains at least one year of data.
    year_counts maps country → number of years it competed (appeared in 'to').
    """
    all_years = sorted(year_data.keys())
    start0 = (all_years[0] // WINDOW_SIZE) * WINDOW_SIZE   # align to grid
    end0   = all_years[-1]

    windows: "OrderedDict[str, Tuple[pd.DataFrame, Dict[str, int]]]" = OrderedDict()
    for start in range(start0, end0 + 1, WINDOW_SIZE):
        end    = start + WINDOW_SIZE - 1
        frames = [year_data[y] for y in range(start, end + 1) if y in year_data]
        if not frames:
            continue
        agg = (pd.concat(frames)
               .groupby(["from", "to"], as_index=False)["points"].sum())
        year_counts: Dict[str, int] = {}
        for f in frames:
            for c in f["to"].unique():
                year_counts[c] = year_counts.get(c, 0) + 1
        windows[f"{start}–{end}"] = (agg, year_counts)
    return windows


# ── Step 2: cluster one window ────────────────────────────────────────────────
def _cluster_window(
    agg: pd.DataFrame, year_counts: Dict[str, int]
) -> List[Tuple[frozenset, float]]:
    """
    Ward hierarchical clustering on the mutual-vote affinity matrix.
    Only countries with >= MIN_ERA_APPEARANCES in the window are included.
    Cluster sizes are uncapped — natural Ward structure determines them.

    Post-processing:
      1. Undersized clusters (< MIN_BLOC_SIZE) are absorbed into the nearest bloc.
      2. Any remaining countries form a new bloc if >= MIN_BLOC_SIZE.
      3. Local search (greedy swap/move) maximises total within-bloc affinity.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    countries = sorted(
        c for c in set(agg["from"]) | set(agg["to"])
        if year_counts.get(c, 0) >= MIN_ERA_APPEARANCES
    )
    n = len(countries)
    if n < 2:
        return [(frozenset([c]), 0.0) for c in countries]

    idx = {c: i for i, c in enumerate(countries)}
    aff = np.zeros((n, n))
    for _, row in agg.iterrows():
        fi, ti = idx.get(row["from"]), idx.get(row["to"])
        if fi is not None and ti is not None:
            aff[fi, ti] += row["points"]
    aff = aff + aff.T
    max_aff = aff.max() if aff.max() > 0 else 1.0
    dist = max_aff - aff
    np.fill_diagonal(dist, 0.0)

    def _strength(members: List[str]) -> float:
        pairs = [(idx[a], idx[b])
                 for ei, a in enumerate(members) for b in members[ei + 1:]]
        return float(np.mean([aff[pi, pj] for pi, pj in pairs])) / max_aff if pairs else 0.0

    # ── dynamic cut: try gap positions largest-first ──────────────────────────
    Z        = linkage(squareform(dist, checks=False), method="ward")
    sorted_h = np.sort(Z[:, 2])
    gaps     = np.diff(sorted_h)
    n_skip   = max(1, len(gaps) // 3)
    candidates = (sorted(range(n_skip, len(gaps)), key=lambda k: gaps[k], reverse=True)
                  if n_skip < len(gaps) else [0])

    best_cov, best_groups = -1, {0: set(countries)}
    for pos in candidates:
        cut_h = (sorted_h[pos] + sorted_h[pos + 1]) / 2
        labels = fcluster(Z, t=cut_h, criterion="distance")
        grps: Dict[int, set] = defaultdict(set)
        for ci, lab in enumerate(labels):
            grps[lab].add(countries[ci])
        cov = sum(len(g) for g in grps.values() if len(g) >= MIN_BLOC_SIZE)
        if cov > best_cov:
            best_cov, best_groups = cov, dict(grps)
        if cov / n >= (1 - MAX_SINGLETON_FRAC):
            break

    # ── pass 1: accept qualifying blocs; collect undersized ──────────────────
    blocs: List[Tuple[frozenset, float]] = []
    covered: set = set()
    small: List[str] = []

    for g in best_groups.values():
        members = sorted(g)
        if len(g) >= MIN_BLOC_SIZE:
            fs = frozenset(members)
            blocs.append((fs, _strength(members)))
            covered |= fs
        else:
            small.extend(members)

    # ── pass 2: absorb undersized into affinity-nearest bloc ─────────────────
    for c in small:
        ci = idx[c]
        best_bi, best_avg = -1, -1.0
        for bi, (fs, _) in enumerate(blocs):
            avg = float(np.mean([aff[ci, idx[m]] for m in fs]))
            if avg > best_avg:
                best_avg, best_bi = avg, bi
        if best_bi >= 0:
            new_members = sorted(blocs[best_bi][0] | {c})
            blocs[best_bi] = (frozenset(new_members), _strength(new_members))
            covered.add(c)

    # ── pass 3: group remaining countries into a new bloc if enough exist ─────
    remaining = [c for c in countries if c not in covered]
    if len(remaining) >= MIN_BLOC_SIZE:
        fs = frozenset(remaining)
        blocs.append((fs, _strength(remaining)))
        covered.update(remaining)

    # ── pass 4: local search — swap / move to maximise within-bloc affinity ───
    def _aff_sum(members: List[str]) -> float:
        return sum(aff[idx[a], idx[b]]
                   for ei, a in enumerate(members) for b in members[ei + 1:])

    improved = True
    while improved:
        improved = False
        best_delta, best_op = 0.0, None
        qual = [(i, fs) for i, (fs, _) in enumerate(blocs)
                if len(fs) >= MIN_BLOC_SIZE]

        # moves: send one country from source (keeps >= MIN) to any other bloc
        for i, fs_i in qual:
            if len(fs_i) <= MIN_BLOC_SIZE:
                continue
            base_i = _aff_sum(sorted(fs_i))
            for j, fs_j in qual:
                if j == i:
                    continue
                base_j = _aff_sum(sorted(fs_j))
                for c in sorted(fs_i):
                    delta = (_aff_sum(sorted(fs_i - {c})) +
                             _aff_sum(sorted(fs_j | {c})) - base_i - base_j)
                    if delta > best_delta:
                        best_delta, best_op = delta, ('move', i, j, c)

        # swaps: exchange one country between two blocs (sizes stay the same)
        for ii, (i, fs_i) in enumerate(qual):
            base_i = _aff_sum(sorted(fs_i))
            for j, fs_j in qual[ii + 1:]:
                base_j = _aff_sum(sorted(fs_j))
                for a in sorted(fs_i):
                    for b in sorted(fs_j):
                        delta = (_aff_sum(sorted((fs_i - {a}) | {b})) +
                                 _aff_sum(sorted((fs_j - {b}) | {a})) - base_i - base_j)
                        if delta > best_delta:
                            best_delta, best_op = delta, ('swap', i, j, a, b)

        if best_op:
            improved = True
            if best_op[0] == 'move':
                _, i, j, c = best_op
                new_i = blocs[i][0] - {c}
                new_j = blocs[j][0] | {c}
                blocs[i] = (new_i, _strength(sorted(new_i)))
                blocs[j] = (new_j, _strength(sorted(new_j)))
            else:
                _, i, j, a, b = best_op
                new_i = (blocs[i][0] - {a}) | {b}
                new_j = (blocs[j][0] - {b}) | {a}
                blocs[i] = (new_i, _strength(sorted(new_i)))
                blocs[j] = (new_j, _strength(sorted(new_j)))

    for c in countries:
        if c not in covered:
            blocs.append((frozenset([c]), 0.0))

    return blocs


# ── Step 3: succession edges ──────────────────────────────────────────────────
def _overlap(a: frozenset, b: frozenset) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def build_succession(
    era_blocs: "OrderedDict[str, List[Tuple[frozenset, float]]]"
) -> List[Tuple[str, int, str, int, int]]:
    """
    Return  (era_src, idx_src, era_dst, idx_dst, n_shared)  for every pair of
    blocs in *consecutive* eras whose overlap coefficient ≥ OVERLAP_THRESHOLD.
    """
    era_list = list(era_blocs.items())
    edges = []
    for (era_a, blocs_a), (era_b, blocs_b) in zip(era_list, era_list[1:]):
        for i, (fa, _) in enumerate(blocs_a):
            for j, (fb, _) in enumerate(blocs_b):
                ov = _overlap(fa, fb)
                if ov >= OVERLAP_THRESHOLD:
                    edges.append((era_a, i, era_b, j, len(fa & fb)))
    return edges


# ── Step 4: render ────────────────────────────────────────────────────────────
# Era colour palette — one hue per window, cycling through a warm→cool gradient
_ERA_PALETTE = [
    "#fde68a", "#fca5a5", "#86efac", "#93c5fd",
    "#c4b5fd", "#f9a8d4", "#6ee7b7", "#fcd34d",
    "#a5b4fc", "#fdba74", "#67e8f9", "#d9f99d",
]


def _node_id(era: str, idx: int) -> str:
    return f"{era.replace('–', '_')}_{idx}"


def render_succession(
    era_blocs: "OrderedDict[str, List[Tuple[frozenset, float]]]",
    edges: List[Tuple[str, int, str, int, int]],
    basename: str = OUTPUT,
) -> str:
    dot = graphviz.Digraph("succession_poset", format="svg")
    dot.attr(rankdir="LR", bgcolor="white",
             nodesep="0.4", ranksep="1.2", splines="ortho")
    dot.attr("edge", color="#9aa5b1", penwidth="1.5",
             arrowhead="normal", arrowsize="0.6")

    for era_idx, (era, blocs) in enumerate(era_blocs.items()):
        fill = _ERA_PALETTE[era_idx % len(_ERA_PALETTE)]
        with dot.subgraph(name=f"cluster_{era_idx}") as sg:
            sg.attr(rank="same", label=era, style="rounded",
                    color="#cccccc", bgcolor="#fafafa",
                    fontname="Helvetica-Bold", fontsize="11")
            for bloc_idx, (fs, strength) in enumerate(blocs):
                nid   = _node_id(era, bloc_idx)
                label = " · ".join(sorted(fs))
                sg.node(nid, label=label, shape="box",
                        style="rounded,filled", fillcolor=fill,
                        color="#888888", fontname="Helvetica",
                        fontsize="9", margin="0.12,0.06")

    # edge weight = number of shared nations → line thickness
    for era_a, i, era_b, j, n_shared in edges:
        src = _node_id(era_a, i)
        dst = _node_id(era_b, j)
        pw  = str(0.8 + n_shared * 0.35)
        dot.edge(src, dst, penwidth=pw,
                 tooltip=f"{n_shared} shared nations")

    dot.render(basename, format="svg", cleanup=True)
    dot.render(basename, format="png", cleanup=True)
    print(f"[render] {basename}.svg / .png")
    return f"{basename}.svg"


# ── Orchestration ─────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print(f"Eurovision bloc succession poset  (window = {WINDOW_SIZE} yr)")
    print("=" * 60)

    year_data = fetch_all(YEARS)
    if not year_data:
        print("No data fetched.")
        return

    windows   = make_windows(year_data)
    era_blocs: OrderedDict[str, List[Tuple[frozenset, float]]] = OrderedDict()

    for era, (agg, year_counts) in windows.items():
        blocs = _cluster_window(agg, year_counts)
        era_blocs[era] = blocs
        names = [" ".join(sorted(fs)[:3]) + ("…" if len(fs) > 3 else "")
                 for fs, _ in blocs]
        print(f"  {era}: {len(blocs)} blocs  {names}")

    edges = build_succession(era_blocs)
    print(f"\n{len(edges)} succession edges  (overlap ≥ {OVERLAP_THRESHOLD})")
    for era_a, i, era_b, j, n in edges:
        a = sorted(era_blocs[era_a][i][0])
        b = sorted(era_blocs[era_b][j][0])
        print(f"  {era_a}[{i}] {a} → {era_b}[{j}] {b}  ({n} shared)")

    render_succession(era_blocs, edges)
    print(f"\nDone.  Open {OUTPUT}.svg")


if __name__ == "__main__":
    main()
