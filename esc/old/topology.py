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
MERGE_JACCARD    = 0.55    # cross-year blocs merge if Jaccard similarity >= this
                           # kept below 0.5 to prevent fine sub-blocs merging with
                           # coarser parent blocs (fine-4 ⊂ coarse-8 → Jaccard=0.5)
MAX_BLOC_SIZE    = 18      # drop blocs larger than this (giant coarse clusters obscure structure)

# --- period-based context ---
PERIODS = [(1975, 2003), (2004, 2025)]   # split at EU enlargement / diaspora-voting shift

# --- output ---
OUTPUT_BASENAME = "eurovision_country_lattice"


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
    DETECTOR == "h0": hierarchical clustering at TWO scales to create lattice depth.

    Cuts the dendrogram at two levels:
      - FINE   (largest gap in the lower half of merge heights) → tight sub-blocs
      - COARSE (largest gap in the upper half)                  → broader regional
                                                                  groupings

    Both levels feed the incidence matrix as separate columns. Because a fine
    sub-bloc is a strict subset of its coarse parent, FCA builds depth: a year
    where both levels are active sits *lower* (more constrained) in the Hasse
    diagram than a year that only shows the coarser grouping.

    Combined with Jaccard-based cross-year merging (see unify_blocs), fine and
    coarse blocs stay as distinct identities across years instead of collapsing
    into one.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = len(countries)
    if n < MIN_BLOC_SIZE:
        return []

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='single')      # single-linkage = same as Rips H0
    heights = Z[:, 2]

    if len(heights) < 4:
        return []

    sorted_h = np.sort(heights)
    gaps = np.diff(sorted_h)

    # discard bottom quarter – those are low-noise merges within tight groups
    n_skip = max(1, len(gaps) // 4)
    usable = gaps[n_skip:]
    if len(usable) < 2:
        return []

    # split usable range into lower half (fine scale) / upper half (coarse scale)
    mid = len(usable) // 2
    halves = [(usable[:mid], n_skip), (usable[mid:], n_skip + mid)]

    cut_heights = []
    for half, offset in halves:
        if len(half) == 0:
            continue
        best = int(np.argmax(half))
        abs_idx = offset + best
        cut_h = (sorted_h[abs_idx] + sorted_h[abs_idx + 1]) / 2
        cut_heights.append(cut_h)

    if not cut_heights:
        return []

    all_blocs: set = set()
    for cut_h in cut_heights:
        labels = fcluster(Z, t=cut_h, criterion='distance')
        groups: Dict[int, set] = defaultdict(set)
        for i, lab in enumerate(labels):
            groups[lab].add(countries[i])
        for g in groups.values():
            if len(g) >= MIN_BLOC_SIZE:
                all_blocs.add(frozenset(g))

    return list(all_blocs)


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



def unify_blocs(per_year: Dict[int, List[frozenset]]
                ) -> Tuple[pd.DataFrame, Dict[str, frozenset]]:
    """
    Greedily merge blocs across all years using JACCARD similarity against a
    stable SEED (the first instance that created each canonical identity).

    Why Jaccard (not overlap): Jaccard is symmetric and penalises size
    differences. A fine 4-country sub-bloc vs a coarse 8-country parent bloc
    scores Jaccard = 4/8 = 0.50, which stays below MERGE_JACCARD = 0.55, so
    they remain as *separate columns* in the incidence matrix. Overlap would
    score 1.0 and merge them, destroying the hierarchy.

    Why seed (not running union): comparing against the ever-growing union would
    inflate denominators over decades and fragment what should be one identity.

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
                o = _jaccard(b, seed)
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
def _multi_level_clusters(Z: np.ndarray, countries: List[str],
                           n_levels: int = 3) -> List[frozenset]:
    """
    Cut the dendrogram at its n_levels largest gaps and collect all clusters
    from every cut.  Because the cuts are nested in the tree, a tight pair
    {DK, NO} from the fine cut is always a subset of the medium-cut group
    {DK, NO, SE, FI, IS}, which is itself a subset of the coarse coalition.
    These nested attribute sets are what gives FCA the depth to build a
    proper hierarchy: countries in the tight pair share MORE attributes than
    countries that only appear at the coarser level.
    """
    from scipy.cluster.hierarchy import fcluster
    heights = Z[:, 2]
    if len(heights) < 2:
        return []
    sorted_h = np.sort(heights)
    gaps = np.diff(sorted_h)
    # pick the n_levels widest gaps as cut points (sorted low→high so cuts are ordered)
    top_idx = np.sort(np.argsort(gaps)[::-1][:n_levels])
    all_blocs: set = set()
    for idx in top_idx:
        cut_h = (sorted_h[idx] + sorted_h[idx + 1]) / 2
        labels = fcluster(Z, t=cut_h, criterion='distance')
        groups: Dict[int, set] = defaultdict(set)
        for i, lab in enumerate(labels):
            groups[lab].add(countries[i])
        for g in groups.values():
            if 2 <= len(g) <= MAX_BLOC_SIZE:
                all_blocs.add(frozenset(g))
    return list(all_blocs)


def build_period_context(year_data: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Objects = countries.  Attributes = one per (voting cluster, time period).

    For each period in PERIODS, all votes are aggregated across its years and
    a single hierarchical clustering is run on the resulting distance matrix.
    Each cluster becomes one attribute whose name encodes both its core
    members and the era, e.g. "DK+NO+SE… (1990–1999)".

    A country has that attribute iff it was part of that cluster.  Countries
    that stayed in the same voting group across multiple eras share several
    attributes and sit deep in the FCA lattice.  Countries that shifted
    allegiance appear at branching points; newcomers appear high up.  The
    time periods are embedded in the attribute names, so the rendered diagram
    shows WHEN each grouping was stable without needing year-objects.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    attrs: Dict[str, frozenset] = {}

    for (start, end) in PERIODS:
        label = f"{start}–{end}"
        frames = [df for y, df in year_data.items() if start <= y <= end]
        if not frames:
            continue
        agg = (pd.concat(frames)
               .groupby(["from", "to"], as_index=False)["points"].sum())
        if agg.empty:
            continue

        dist, countries = year_distance_matrix(agg)
        if len(countries) < MIN_BLOC_SIZE + 1:
            continue

        # Ward gives compact balanced clusters; single-linkage chains into one
        # giant blob when votes are aggregated over many years.
        Z = linkage(squareform(dist, checks=False), method='ward')
        blocs = _multi_level_clusters(Z, countries, n_levels=2)
        for bloc in blocs:
            core = "+".join(sorted(bloc)[:3])
            suffix = "…" if len(bloc) > 3 else ""
            attrs[f"{core}{suffix} ({label})"] = bloc
        print(f"[period] {label}: {len(blocs)} blocs")

    if not attrs:
        raise ValueError("No blocs found — check PERIODS covers the data range.")

    all_countries = sorted({c for g in attrs.values() for c in g})
    mat = pd.DataFrame(0, index=all_countries, columns=list(attrs.keys()), dtype=int)
    for attr, members in attrs.items():
        for c in members:
            mat.loc[c, attr] = 1

    mat = mat.loc[mat.sum(axis=1) > 0, :]
    mat = mat.loc[:, mat.sum(axis=0) > 0]
    return mat


def build_combined_context(binary: pd.DataFrame,
                           members: Dict[str, frozenset]) -> pd.DataFrame:
    """
    Objects = countries ∪ years, Attributes = stable voting blocs.

    A country has attribute B iff it is a member of bloc B.
    A year    has attribute B iff bloc B was active that year.

    A concept (extent, intent) therefore reads directly as:
        "these countries were voting together as {blocs} in {years}"

    Blocs larger than MAX_BLOC_SIZE are excluded so one giant coarse cluster
    cannot dominate and flatten the lattice.  The dual-scale extraction
    (fine sub-blocs + coarser coalitions) gives depth: a concept that carries
    both a fine and a coarse bloc attribute sits lower than one that only
    carries the coarse one.
    """
    filtered = {lab: mem for lab, mem in members.items()
                if len(mem) <= MAX_BLOC_SIZE and lab in binary.columns}

    country_objects = sorted({c for mem in filtered.values() for c in mem})
    year_objects    = list(binary.index)          # strings like "1975"
    objects = country_objects + year_objects

    attrs = list(filtered.keys())
    mat = pd.DataFrame(0, index=objects, columns=attrs, dtype=int)

    for lab, mem in filtered.items():
        for c in mem:
            mat.loc[c, lab] = 1

    for lab in filtered:
        for y in year_objects:
            if binary.loc[y, lab] == 1:
                mat.loc[y, lab] = 1

    mat = mat.loc[mat.sum(axis=1) > 0, :]
    mat = mat.loc[:, mat.sum(axis=0) > 0]
    return mat


def build_lattice(binary: pd.DataFrame) -> Context:
    """Generic binary incidence matrix → FCA concept lattice."""
    if binary.empty or binary.shape[1] == 0:
        raise ValueError("Empty incidence matrix — no blocs survived filtering. "
                         "Loosen PERSIST_QUANTILE / MERGE_JACCARD / MIN_YEARS_ACTIVE / MAX_BLOC_SIZE.")
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
    Render the Hasse diagram with reduced labelling (standard FCA convention).

    Each object/attribute is written exactly once, at the highest/lowest node
    where it first appears.  Three visual layers per node (top → bottom):

      BLUE BOLD   — bloc names introduced at this node (intent diff)
      GREEN       — country names whose unique position is this node (extent diff)
      GREY ITALIC — year numbers whose unique position is this node (extent diff)

    Read top → bottom: descending an edge ADDS a bloc constraint; ascending
    relaxes one.  A node that carries both countries and years says "these
    countries voted together as {blocs} in {years}."
    """
    lattice = ctx.lattice
    dot = graphviz.Digraph("voting_blocs", format="svg")
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

        parts = []
        if own_attrs:
            parts.append('<FONT COLOR="#1c4f8f"><B>'
                         + _wrap(sorted(own_attrs), 1)
                         + '</B></FONT>')
        if own_objs:
            parts.append('<FONT COLOR="#2d6a4f" POINT-SIZE="9">'
                         + _wrap(sorted(own_objs), 4)
                         + '</FONT>')
        label = ("<" + "<BR/>".join(parts).replace("\\n", "<BR/>") + ">"
                 if parts else '<<FONT COLOR="#bbbbbb">·</FONT>>')

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
    print("Eurovision voting-bloc country lattice")
    print("=" * 70)

    year_data = fetch_all(YEARS)
    if not year_data:
        print("\nNo data fetched. Open "
              f"{BASE_URL}/2024 in a browser to inspect the JSON schema, then\n"
              "adjust _extract_votes() if needed.")
        return

    mat = build_period_context(year_data)
    print(f"\nCountries × period-blocs: "
          f"{mat.shape[0]} countries × {mat.shape[1]} attributes")
    print(mat.to_string())

    ctx = build_lattice(mat)
    n_concepts = len(list(ctx.lattice))
    print(f"\n[fca] {n_concepts} concepts in the lattice")

    for extent, intent in ctx.lattice:
        if extent and intent:
            print(f"  CONCEPT  {sorted(extent)}  defined-by={sorted(intent)}")

    render_lattice(ctx, "eurovision_country_lattice")
    print("\nDone. Open eurovision_country_lattice.svg")


if __name__ == "__main__":
    main()