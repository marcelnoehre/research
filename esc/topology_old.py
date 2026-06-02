#!/usr/bin/env python3
"""
eurovision_eras.py
==================

Map the *structural evolution* of Eurovision voting into discrete "eras" by chaining
three techniques:

    raw votes  ->  (TDA)  persistent voting structures per year
               ->  (binary matrix)  Years x Blocs incidence
               ->  (FCA)  formal concept lattice  ->  Hasse diagram

Pipeline
--------
1. Data acquisition   : fetch (FromCountry, ToCountry, Points) per year from the API.
2. TDA per year       : build a distance matrix from mutual points, run `ripser`
                        (H1) to find persistent cyclic voting structures, and read
                        the countries on each persistent loop as a candidate "bloc".
3. Cross-year merge   : unify near-identical blocs across years (Jaccard) so each
                        bloc becomes one stable column.
4. FCA                : feed the Years x Blocs binary matrix to `concepts` to build
                        the concept lattice; each concept node = an "era".
5. Visualisation      : render the Hasse diagram with `graphviz`.

------------------------------------------------------------------------------------
A HONEST METHODOLOGICAL NOTE (read this before trusting the output)
------------------------------------------------------------------------------------
The brief asked for "1-D persistent homology to identify voting blocs (cliques)".
Two things are worth being precise about, because they change how you read results:

  * H1 (1-dimensional homology) detects *loops / cycles*, NOT cliques. A persistent
    H1 feature here is a chain of countries A->B->C->...->A that reciprocally trade
    high points around a ring. That is a real and interesting "structural" object
    (mutual-support rings), but it is NOT the same as a densely-connected clique.
    If you literally want cliques/clusters, H0 components (see `blocs_from_h0`) or a
    graph clique finder are the correct tool. Both detectors are provided; switch
    with DETECTOR below.

  * Topological features have no built-in identity across years. A loop in 2004 and a
    loop in 2018 are only "the same bloc" if we *define* them to be. We do that here
    by canonicalising each bloc to its set of member countries and merging blocs whose
    membership overlaps strongly (Jaccard >= MERGE_JACCARD). This is a modelling
    choice, not a fact discovered by the data — tune it and see how the lattice moves.

Treat the lattice as an *exploratory* map of structural regimes, not a statistical
claim. The interesting output is the SHAPE of the hierarchy, not any single node.
------------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
import itertools
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

import requests
import numpy as np
import pandas as pd
from ripser import ripser
from concepts import Context
import graphviz


# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
BASE_URL      = "https://eurovisionapi.runasp.net/api/senior/contests"
YEARS         = range(1975, 2026)          # inclusive of 2000, exclusive of 2026
SKIP_YEARS    = {2020}                     # no contest held (COVID)
REQUEST_PAUSE = 0.4                        # seconds between calls, be polite to the API
TIMEOUT       = 20                         # per-request timeout (s)
MAX_RETRIES   = 3                          # retries per request on transient failure

# --- TDA / bloc-extraction knobs ---
DETECTOR        = "h0"     # "h0" (clusters/blocs) recommended; "h1" (cycles) is noisy
PERSIST_QUANTILE = 0.65    # keep only features more persistent than this quantile
MIN_BLOC_SIZE    = 3       # ignore blocs smaller than this many countries
MERGE_JACCARD    = 0.40    # cross-year blocs merge if membership overlap >= this
MIN_YEARS_ACTIVE = 3       # drop blocs that appear in fewer than this many years

# --- output ---
OUTPUT_BASENAME = "eurovision_era_lattice"


# ==================================================================================
# 1. DATA ACQUISITION
# ==================================================================================
#
# Endpoint : GET {BASE_URL}/{year}
# Schema (confirmed from working reference script):
#   data['rounds']         -> list of round dicts; pick name == 'final'
#   round['performances']  -> one per contestant; has 'contestantId', 'scores'
#   performance['scores']  -> list of {name, votes}
#       name = 'total' (pre-2016) | 'jury' / 'public' (2016+)
#       votes = {voter_country_code: points_int}
#   data['contestants']    -> list of {id, country, ...}
#
# For every performance we extract (voter -> performer, points) triples.
# Pre-2016 we use the 'total' score; from 2016 we take max(jury, public) per
# voter to collapse split scoring into a single salience signal.
# ==================================================================================

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json",
                      "User-Agent": "eurovision-era-analysis/1.0"})
    return s


def _get_json(sess: requests.Session, url: str) -> Optional[dict]:
    """GET with retries + exponential backoff. Returns parsed JSON or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = sess.get(url, timeout=TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            wait = 2 ** (attempt - 1)
            print(f"    [warn] {url} attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)
    print(f"    [error] giving up on {url}")
    return None


def _extract_votes(data: dict, year: int) -> List[dict]:
    """
    Parse the contest JSON for a single year and return a flat list of
    {'from': voter_code, 'to': performer_code, 'points': float} dicts
    covering ALL performances in the Grand Final.
    """
    # --- contestant id -> country code lookup ---
    contestants = {c['id']: c['country'] for c in data.get('contestants', [])}

    # --- find the Grand Final round ---
    rounds = data.get('rounds', [])
    final = next((r for r in rounds if r.get('name') == 'final'), None)
    if final is None:
        # some early years label it differently
        final = next(
            (r for r in rounds if 'final' in str(r.get('name', '')).lower()),
            None,
        )
    if final is None and len(rounds) == 1:
        final = rounds[0]                        # single-round contest
    if final is None:
        return []

    rows: List[dict] = []

    for perf in final.get('performances', []):
        to_country = contestants.get(perf.get('contestantId'))
        if not to_country:
            continue

        scores = {
            s['name']: s.get('votes', {})
            for s in perf.get('scores', [])
            if isinstance(s, dict) and 'name' in s
        }

        if year >= 2016 and ('jury' in scores or 'public' in scores):
            # split voting era: collapse jury + televote by taking the max
            jury  = scores.get('jury', {})
            tele  = scores.get('public', {})
            voters = set(jury) | set(tele)
            votes = {v: max(jury.get(v, 0), tele.get(v, 0)) for v in voters}
        elif 'total' in scores:
            votes = scores['total']
        else:
            # fallback: grab whichever score dict exists
            votes = next(iter(scores.values()), {})

        for voter, pts in votes.items():
            try:
                pts = float(pts)
            except (TypeError, ValueError):
                continue
            if pts > 0 and voter != to_country:
                rows.append({"from": voter, "to": to_country, "points": pts})

    return rows


def fetch_year(sess: requests.Session, year: int) -> List[dict]:
    """Fetch and parse a single year's Grand Final votes."""
    url = f"{BASE_URL}/{year}"
    data = _get_json(sess, url)
    if data is None:
        return []
    return _extract_votes(data, year)


def fetch_all(years: Iterable[int] = YEARS) -> Dict[int, pd.DataFrame]:
    """
    Returns {year: DataFrame[from, to, points]} for every year that produced data.
    Skipped or empty years are logged and omitted.
    """
    sess = _session()
    out: Dict[int, pd.DataFrame] = {}
    for year in years:
        if year in SKIP_YEARS:
            print(f"[fetch] {year} ... skipped (no contest)")
            continue
        print(f"[fetch] {year} ...")
        rows = fetch_year(sess, year)
        if not rows:
            print(f"    [skip] no usable data for {year}")
        else:
            df = pd.DataFrame(rows)
            # collapse any duplicate voter->performer pairs
            df = df.groupby(["from", "to"], as_index=False)["points"].sum()
            out[year] = df
            print(f"    {len(df)} directed vote pairs, "
                  f"{df['from'].nunique()} voting countries")
        time.sleep(REQUEST_PAUSE)
    return out


# ==================================================================================
# 2. TDA  --  per-year persistent structures
# ==================================================================================
def year_distance_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Build a symmetric distance matrix from a year's votes.

    affinity(i,j) = points i->j + points j->i        (mutual support, symmetric)
    distance(i,j) = max_affinity - affinity(i,j)      (high mutual points => close)

    Countries that never exchange points sit at the maximum distance, so they only
    join the filtration late and never form spurious early structure.
    """
    countries = sorted(set(df["from"]) | set(df["to"]))
    idx = {c: i for i, c in enumerate(countries)}
    n = len(countries)

    aff = np.zeros((n, n))
    for f, t, p in df.itertuples(index=False):
        aff[idx[f], idx[t]] += p
    aff = aff + aff.T                       # symmetrise

    max_aff = aff.max() if aff.max() > 0 else 1.0
    dist = max_aff - aff
    np.fill_diagonal(dist, 0.0)
    return dist, countries


def _persistence_cutoff(dgm: np.ndarray) -> float:
    """Persistence threshold = the chosen quantile of finite feature lifetimes."""
    if dgm.size == 0:
        return np.inf
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life)]
    return np.quantile(life, PERSIST_QUANTILE) if life.size else np.inf


def blocs_from_h1(dist: np.ndarray, countries: List[str]) -> List[frozenset]:
    """
    DETECTOR == "h1": persistent 1-cycles = cyclic mutual-support *rings*.

    We run ripser with do_cocycles=True; each H1 representative cocycle is a set of
    edges [i, j, coeff]. The vertices appearing on a sufficiently-persistent cocycle
    are the countries riding that loop -> one candidate bloc.
    """
    res = ripser(dist, maxdim=1, distance_matrix=True, do_cocycles=True)
    dgm1 = res["dgms"][1]
    cocycles = res["cocycles"][1]
    if len(dgm1) == 0:
        return []

    cutoff = _persistence_cutoff(dgm1)
    blocs = []
    for k, (birth, death) in enumerate(dgm1):
        if not np.isfinite(death) or (death - birth) < cutoff:
            continue
        verts = set()
        for edge in cocycles[k]:
            i, j = int(edge[0]), int(edge[1])
            verts.add(countries[i])
            verts.add(countries[j])
        if len(verts) >= MIN_BLOC_SIZE:
            blocs.append(frozenset(verts))
    return blocs


def blocs_from_h0(dist: np.ndarray, countries: List[str]) -> List[frozenset]:
    """
    DETECTOR == "h0": hierarchical clustering with automatic gap-based cutting.

    Instead of a single persistence-quantile threshold (which tends to produce one
    giant blob or many singletons), we:
      1. Run single-linkage hierarchical clustering on the distance matrix.
      2. Find the largest GAP in the dendrogram merge heights.
      3. Cut at that gap to get the natural cluster partition.

    This reliably separates densely-connected voting blocs (low internal distance)
    from the loose cross-bloc noise (high distance), regardless of the absolute
    scale of the distances.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(countries)
    if n < MIN_BLOC_SIZE:
        return []

    # scipy wants condensed distance vector
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='single')      # single-linkage = same as Rips H0

    # merge heights (column 2 of Z)
    heights = Z[:, 2]

    # find the biggest gap in merge heights -> the natural "jump" between
    # intra-bloc merges (low height) and inter-bloc merges (high height)
    if len(heights) < 2:
        return []
    gaps = np.diff(np.sort(heights))
    # only consider gaps in the upper portion to avoid cutting too early
    # (the bottom gaps are noise merges within blocs)
    n_skip = max(1, len(gaps) // 3)              # skip the bottom third
    candidate_gaps = gaps[n_skip:]
    if len(candidate_gaps) == 0:
        return []
    best_gap_idx = n_skip + np.argmax(candidate_gaps)
    cut_height = (np.sort(heights)[best_gap_idx] + np.sort(heights)[best_gap_idx + 1]) / 2

    labels = fcluster(Z, t=cut_height, criterion='distance')

    groups = defaultdict(set)
    for i, lab in enumerate(labels):
        groups[lab].add(countries[i])
    return [frozenset(g) for g in groups.values() if len(g) >= MIN_BLOC_SIZE]


def extract_blocs(year_data: Dict[int, pd.DataFrame]) -> Dict[int, List[frozenset]]:
    """Run the chosen detector on every year."""
    detect = blocs_from_h1 if DETECTOR == "h1" else blocs_from_h0
    per_year: Dict[int, List[frozenset]] = {}
    for year, df in year_data.items():
        dist, countries = year_distance_matrix(df)
        try:
            per_year[year] = detect(dist, countries)
        except Exception as exc:               # ripser can choke on degenerate years
            print(f"    [warn] TDA failed for {year}: {exc}")
            per_year[year] = []
        print(f"[tda] {year}: {len(per_year[year])} significant blocs")
    return per_year


# ==================================================================================
# 3. CROSS-YEAR MERGE  --  give blocs a stable identity
# ==================================================================================
def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _overlap(a: frozenset, b: frozenset) -> float:
    """Overlap coefficient: |A∩B| / min(|A|,|B|). Tolerant of size differences."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def unify_blocs(per_year: Dict[int, List[frozenset]]
                ) -> Tuple[pd.DataFrame, Dict[str, frozenset]]:
    """
    Greedily merge blocs across all years using the OVERLAP coefficient (not
    Jaccard) so that a 4-country Nordic bloc still matches a slightly-shifted
    5-country Nordic bloc from the next year. We keep a stable SEED for each
    canonical bloc (the first instance that created it) and compare new blocs
    against seeds — NOT against the ever-growing union, which would cause
    Jaccard to collapse and fragment what should be one stable identity.

    Returns:
      * binary DataFrame  rows=year, cols=bloc_label, cell=1 if active that year
      * dict bloc_label -> canonical frozenset of member countries
    """
    seeds:  List[frozenset] = []                 # first instance (stable comparator)
    unions: List[set]       = []                 # running union (for final label)
    active: List[set]       = []                 # set of years each canon is active

    for year, blocs in sorted(per_year.items()):
        for b in blocs:
            best, best_o = -1, 0.0
            for k, seed in enumerate(seeds):
                o = _overlap(b, seed)
                if o > best_o:
                    best, best_o = k, o
            if best_o >= MERGE_JACCARD:          # threshold now on overlap coefficient
                unions[best] |= set(b)
                active[best].add(year)
            else:
                seeds.append(b)
                unions.append(set(b))
                active.append({year})

    # human-readable label: based on the SEED (stable first instance), not the
    # ever-growing union which accumulates noise over decades
    def label(members: frozenset) -> str:
        head = "+".join(sorted(members)[:3])
        return f"{head}…({len(members)})" if len(members) > 3 else head

    labels = [label(s) for s in seeds]
    # de-duplicate identical labels
    seen = defaultdict(int)
    uniq = []
    for lab in labels:
        seen[lab] += 1
        uniq.append(lab if seen[lab] == 1 else f"{lab}#{seen[lab]}")

    years = sorted(per_year.keys())
    mat = pd.DataFrame(0, index=[str(y) for y in years], columns=uniq, dtype=int)
    for k, yrs in enumerate(active):
        if len(yrs) < MIN_YEARS_ACTIVE:
            continue
        for y in yrs:
            mat.loc[str(y), uniq[k]] = 1

    # drop all-zero columns (blocs that failed MIN_YEARS_ACTIVE) and all-zero rows
    mat = mat.loc[:, (mat.sum(axis=0) > 0)]
    mat = mat.loc[(mat.sum(axis=1) > 0), :]
    bloc_members = {uniq[k]: seeds[k] for k in range(len(uniq))
                    if uniq[k] in mat.columns}
    return mat, bloc_members


# ==================================================================================
# 4. FCA  --  build the concept lattice
# ==================================================================================
def build_lattice(binary: pd.DataFrame) -> Context:
    """
    Objects   = years, Attributes = blocs.
    A concept (Extent, Intent) is a maximal pairing: a set of years that ALL share
    exactly the same set of active blocs. Those concepts are the "eras".
    """
    if binary.empty or binary.shape[1] == 0:
        raise ValueError("Empty incidence matrix — no blocs survived filtering. "
                         "Loosen PERSIST_QUANTILE / MERGE_JACCARD / MIN_YEARS_ACTIVE.")
    objects = list(binary.index)
    properties = list(binary.columns)
    bools = [tuple(bool(v) for v in row) for row in binary.values]
    return Context(objects, properties, bools)


# ==================================================================================
# 5. VISUALISATION
# ==================================================================================
def _wrap(items: List[str], per_line: int = 4) -> str:
    """Comma-join with a line break every `per_line` items (keeps nodes narrow)."""
    chunks = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return "\\n".join(chunks)


def render_lattice(ctx: Context, basename: str = OUTPUT_BASENAME) -> str:
    """
    Render the Hasse diagram with REDUCED LABELLING (the standard FCA convention),
    so every year/bloc is written exactly once instead of being repeated on every
    node it belongs to. This is what keeps the diagram readable.

      * a BLOC is written at the highest node where it becomes mandatory
        (intent minus the intents of upper neighbours)  -> shown above the dot, blue
      * a YEAR is written at the lowest node it reaches
        (extent minus the extents of lower neighbours)   -> shown below the dot, grey

    Read top->bottom: the top node is the era shared by all years (no constraints);
    descending an edge ADDS a bloc constraint (a structural pivot); the bottom node
    is the most constrained, niche regime.
    """
    lattice = ctx.lattice
    dot = graphviz.Digraph("eras", format="svg")
    dot.attr(rankdir="TB", bgcolor="white", nodesep="0.5", ranksep="0.9",
             splines="true")
    dot.attr("edge", color="#9aa5b1", arrowhead="none", penwidth="1.2")

    for c in lattice:
        own_objs = set(c.extent)
        for lo in c.lower_neighbors:
            own_objs -= set(lo.extent)
        own_attrs = set(c.intent)
        for up in c.upper_neighbors:
            own_attrs -= set(up.intent)

        attr_txt = _wrap(sorted(own_attrs), 1)          # blocs: one per line
        obj_txt  = _wrap(sorted(own_objs), 6)           # years: compact rows
        parts = []
        if attr_txt:
            parts.append(f'<FONT COLOR="#1c4f8f"><B>{attr_txt}</B></FONT>')
        if obj_txt:
            parts.append(f'<FONT COLOR="#555555" POINT-SIZE="9">{obj_txt}</FONT>')
        label = "<" + "<BR/>".join(parts).replace("\\n", "<BR/>") + ">" if parts \
                else "<<FONT COLOR=\"#bbbbbb\">·</FONT>>"

        # nodes that introduce a bloc are the structural pivots -> emphasise them
        if own_attrs:
            dot.node(str(c.index), label=label, shape="box",
                     style="rounded,filled", fillcolor="#eaf2fd",
                     color="#1c4f8f", fontname="Helvetica")
        else:
            dot.node(str(c.index), label=label, shape="ellipse",
                     style="filled", fillcolor="#f3f4f6",
                     color="#9aa5b1", fontname="Helvetica")

    for c in lattice:
        for up in c.upper_neighbors:
            dot.edge(str(up.index), str(c.index))

    dot.render(basename, format="svg", cleanup=True)
    dot.render(basename, format="png", cleanup=True)
    print(f"[render] wrote {basename}.svg and {basename}.png")
    return f"{basename}.svg"


# ==================================================================================
# Orchestration
# ==================================================================================
def main():
    print("=" * 70)
    print("Eurovision voting-era lattice")
    print("=" * 70)

    year_data = fetch_all(YEARS)
    if not year_data:
        print("\nNo data fetched. Open "
              f"{BASE_URL}/2024 in a browser to inspect the JSON schema, then\n"
              "adjust _extract_votes() if needed.")
        return

    per_year = extract_blocs(year_data)
    binary, members = unify_blocs(per_year)

    print("\nYears x Blocs incidence matrix:")
    print(binary)

    print("\nBloc membership:")
    for lab, mem in members.items():
        print(f"  {lab:>20s} : {', '.join(sorted(mem))}")

    ctx = build_lattice(binary)
    print(f"\n[fca] {len(list(ctx.lattice))} concepts (eras) in the lattice")

    # Interpreting concepts as STRUCTURAL PIVOTS ----------------------------------
    # Each lattice node groups years that are structurally indistinguishable (same
    # active blocs). An EDGE between two nodes is a *pivot*: moving down adds a bloc
    # constraint (a new ring/cluster appears); moving up relaxes one (a bloc dies).
    # Years that sit alone in a deep node are structural one-offs; long vertical
    # chains are gradual drift; wide branches are forks into competing regimes.
    for extent, intent in ctx.lattice:
        if extent and intent:
            print(f"  ERA  years={list(extent)}  defined-by={list(intent)}")

    render_lattice(ctx)
    print("\nDone. Open eurovision_era_lattice.svg")


if __name__ == "__main__":
    main()