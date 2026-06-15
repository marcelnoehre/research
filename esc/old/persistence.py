import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import networkx as nx
from collections import defaultdict
from itertools import combinations

THRESHOLD = 8   # distance ≤ 8  ⟺  combined votes ≥ 16
TOP_N     = 30

valid_years = [y for y in range(1975, 2026) if y != 2020]
year_distances    = defaultdict(dict)   # [year][(a, b)] = dist
year_participants = defaultdict(set)    # [year] -> set of countries in the final

for year in valid_years:
    try:
        print(year)
        url = f'https://eurovisionapi.runasp.net/api/senior/contests/{year}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        id_to_cc = {c['id']: c['country'] for c in data.get('contestants', [])}

        final_round = next(
            (r for r in data['rounds'] if r.get('name') == 'final'), None
        )

        vote_matrix = {}
        for performance in final_round['performances']:
            cid      = performance['contestantId']
            receiver = id_to_cc.get(cid)
            if year >= 2016:
                tele_score = next(
                    (s for s in performance['scores'] if s.get('name') == 'public'), None
                )
                votes = tele_score['votes'] if tele_score else {}
            else:
                total_score = next(
                    (s for s in performance['scores'] if s.get('name') == 'total'), None
                )
                votes = total_score['votes'] if total_score else {}

            for giver, points in votes.items():
                vote_matrix.setdefault(giver, {})[receiver] = points

        countries = list({r for pv in vote_matrix.values() for r in pv})
        year_participants[year].update(countries)
        for i, a in enumerate(countries):
            for b in countries[i + 1:]:
                ab   = vote_matrix.get(a, {}).get(b, 0)
                ba   = vote_matrix.get(b, {}).get(a, 0)
                pair = tuple(sorted([a, b]))
                year_distances[year][pair] = 24 - (ab + ba)

    except Exception as e:
        print(f"Error fetching data for year {year}: {e}")

# ── Build per-year alliance graph ─────────────────────────────────────────────
year_graphs = {}
for year, dists in year_distances.items():
    G = nx.Graph()
    for (a, b), dist in dists.items():
        if dist <= THRESHOLD:
            G.add_edge(a, b)
    year_graphs[year] = G

# ── Find all cliques of size ≥ 2 for each year ───────────────────────────────
# Expand every maximal clique into all its sub-cliques (size ≥ 2);
# a sub-clique is "alive" in a year whenever all its pairwise edges are present.
clique_active = defaultdict(set)   # frozenset(countries) -> set of years active
for year, G in year_graphs.items():
    seen = set()
    for maxclique in nx.find_cliques(G):
        for size in range(2, len(maxclique) + 1):
            for combo in combinations(sorted(maxclique), size):
                fs = frozenset(combo)
                if fs not in seen:
                    seen.add(fs)
                clique_active[fs].add(year)


def compute_intervals(years_active, clique):
    """Break an interval only when every member of the clique participated
    but the alliance was not active.  Absent years are skipped silently."""
    intervals, start, end = [], None, None
    for yr in valid_years:
        all_present = clique.issubset(year_participants.get(yr, set()))
        if yr in years_active:
            if start is None:
                start = yr
            end = yr
        elif all_present and start is not None:
            intervals.append((start, end))
            start = None
        # else: at least one member absent — do not break the interval
    if start is not None:
        intervals.append((start, end))
    return intervals


clique_intervals   = {c: compute_intervals(yrs, c) for c, yrs in clique_active.items()}
clique_persistence = {c: sum(d - b + 1 for b, d in ivs) for c, ivs in clique_intervals.items()}

# ── Select top cliques per size tier for the barcode ─────────────────────────
def label(clique):
    return '-'.join(sorted(clique))

sizes = sorted({len(c) for c in clique_active})
size_cmap = {s: cm.tab10(i / max(len(sizes) - 1, 1)) for i, s in enumerate(sizes)}

# For the barcode: top-N by persistence, weighted by size so groups surface
clique_score  = {c: clique_persistence[c] * len(c) for c in clique_active}
top_cliques   = sorted(clique_score.items(), key=lambda x: -x[1])[:TOP_N]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))

# Left – barcode (ranked by size × persistence)
for idx, (clique, _) in enumerate(top_cliques):
    k     = len(clique)
    color = size_cmap[k]
    lw    = 2 + k          # thicker bars for larger alliances
    for birth, death in clique_intervals[clique]:
        ax1.plot([birth, death], [idx, idx],
                 linewidth=lw, color=color, solid_capstyle='round', alpha=0.85)

ax1.set_yticks(range(len(top_cliques)))
ax1.set_yticklabels([label(c) for c, _ in top_cliques], fontsize=7)
ax1.set_xlabel('Year')
ax1.set_title(f'Alliance barcodes  (distance ≤ {THRESHOLD}, ranked by size × persistence)')
ax1.set_xlim(min(valid_years) - 1, max(valid_years) + 1)
ax1.grid(axis='x', alpha=0.3)
ax1.axvline(x=1993, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax1.text(1993.3, TOP_N - 0.5, '1993', fontsize=7, color='gray', va='top')

# Size legend
handles = [plt.Line2D([0], [0], color=size_cmap[s], linewidth=2 + s, label=f'size {s}')
           for s in sizes]
ax1.legend(handles=handles, loc='lower right', fontsize=7)

# Right – persistence diagram
# each interval (birth, death) is one point; marker area ∝ clique size
all_pts = [
    (birth, death, clique, death - birth + 1)
    for clique, ivs in clique_intervals.items()
    for birth, death in ivs
]
births    = np.array([p[0] for p in all_pts])
deaths    = np.array([p[1] for p in all_pts])
durations = np.array([p[3] for p in all_pts])
ksizes    = np.array([len(p[2]) for p in all_pts])

sc = ax2.scatter(births, deaths, c=durations, s=30 * ksizes**1.5,
                 cmap='plasma', alpha=0.65, zorder=3, vmin=1)
plt.colorbar(sc, ax=ax2, label='Duration (years)')

# Label the most persistent large-clique intervals
thresh_label = np.percentile(durations * ksizes, 93)
labeled = set()
for birth, death, clique, dur in all_pts:
    key = label(clique)
    if dur * len(clique) >= thresh_label and key not in labeled:
        ax2.annotate(key, (birth, death),
                     fontsize=6, ha='left', va='bottom',
                     xytext=(3, 3), textcoords='offset points', alpha=0.9)
        labeled.add(key)

yr_min, yr_max = min(valid_years), max(valid_years)
ax2.plot([yr_min, yr_max], [yr_min, yr_max], 'k--', alpha=0.25, linewidth=1)
ax2.set_xlabel('Birth year')
ax2.set_ylabel('Death year')
ax2.set_title('Persistence diagram  (marker area ∝ alliance size, far above diagonal = long-lived)')
ax2.set_aspect('equal')
ax2.grid(alpha=0.3)

# Size legend for scatter
for s in sizes:
    ax2.scatter([], [], s=30 * s**1.5, color='gray', alpha=0.6, label=f'size {s}')
ax2.legend(loc='upper left', fontsize=7)

fig.suptitle('Eurovision voting alliances — persistence analysis', fontsize=14)
plt.tight_layout()
plt.savefig('alliance_persistence.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved alliance_persistence.png")
