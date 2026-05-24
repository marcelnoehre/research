import requests
import pandas as pd

from odis import FormalContext
import networkx as nx
from data import Parser
from fcapy.lattice import ConceptLattice

# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------

CLUSTERS = {
    'Nordic':                ['IS', 'NO', 'SE', 'DK', 'FI'],
    'British Isles':         ['GB', 'GB-WLS', 'IE'],
    'Benelux':               ['BE', 'NL', 'LU'],
    'DACH':                  ['DE', 'AT', 'CH'],
    'Iberia':                ['PT', 'ES', 'AD'],
    'Italian Peninsula':     ['IT', 'SM', 'MC'],
    'Balkans':       ['AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'SI', 'BG', 'RO'],
    'Visegrad':              ['PL', 'CZ', 'SK', 'HU'],
    'Baltics':               ['EE', 'LV', 'LT'],
    'Eastern Europe':        ['RU', 'BY', 'UA', 'MD'],
    'Caucasus':              ['GE', 'AM', 'AZ'],
    'Eastern Mediterranean': ['GR', 'CY', 'TR', 'IL', 'MT'],
    'Non-European':          ['MA', 'KZ', 'AU'],
    'Defunct':               ['YU', 'CS'],
}

THRESHOLD_RATIO = 0.75

# ---------------------------------------------------------------------------
# Voting system: max points a single country can award (to normalise totals)
# ---------------------------------------------------------------------------
# 1957–1961: various jury systems, rough max ~10 per country
# 1962–1963: 3-jury system, each gave 1–3 pts → max ~9; approximate as 10
# 1964–1966: each country had multiple jury members voting 1–5 → ~10
# 1967–1970: 10-point scale jury
# 1971–1974: each country gave a single 10-point vote
# 1975–2015: 1–8, 10, 12 scale → max 12
# 2016+    : jury 12 + televote 12 reported separately → max 24

def max_points_per_voter(year: int) -> float:
    if year <= 1970:
        return 10.0   # approximate for early variable systems
    elif year <= 1974:
        return 10.0
    elif year <= 2015:
        return 12.0
    else:
        return 24.0   # split jury + tele both reported (combined up to 24)

# ---------------------------------------------------------------------------
# Fetch & build matrix
# ---------------------------------------------------------------------------

records = {}

for year in range(1957, 2026):
    if year == 2020:
        continue

    try:
        print(year)
        url = f'https://eurovisionapi.runasp.net/api/senior/contests/{year}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        final_round = next(
            (r for r in data['rounds'] if r.get('name') == 'final'), None
        )
        if final_round is None:
            print(f"  No final round for {year}, skipping.")
            continue

        performance = final_round['performances'][0]

        total_score = next(
            (s for s in performance['scores'] if s.get('name') == 'total'), None
        )
        if total_score is None:
            print(f"  No total score for {year}, skipping.")
            continue

        votes = total_score['votes']  # {country_code: points}
        eligible_voters = set(votes.keys())

        winner_country = next(
            (c for c in data['contestants'] if c.get('id') == performance['contestantId']), None
        )['country']

        max_pts = max_points_per_voter(year)

        # Normalised cluster totals: sum of (points / max_pts) per member
        # → each member contributes a value in [0, 1], so cluster totals
        #   are comparable across voting-system eras regardless of scale
        cluster_totals = {
            cluster: sum(
                votes.get(m, 0) / max_pts
                for m in members
                if m != winner_country and m in eligible_voters  # ← only eligible voters
            )
            for cluster, members in CLUSTERS.items()
        }

        max_cluster_total = max(cluster_totals.values())
        threshold = THRESHOLD_RATIO * max_cluster_total

        records[f'{winner_country}_{year}'] = {
            cluster: 1 if total >= threshold else 0
            for cluster, total in cluster_totals.items()
        }

    except requests.exceptions.RequestException as e:
        print(f"  Request error for {year}: {e}")

# ---------------------------------------------------------------------------
# Assemble DataFrame
# ---------------------------------------------------------------------------

cluster_names = list(CLUSTERS.keys())
df = pd.DataFrame.from_dict(records, orient='index')[cluster_names].sort_index()
df.index.name   = 'year'
df.columns.name = 'cluster'

df = df.sort_index(axis=0)
df = df.sort_index(axis=1)

rows = list(df.index)
cols = list(df.columns)

with open("eurovision_binary_context_clustered.cxt", "w", encoding="utf-8") as f:
    f.write("B\n")
    f.write("\n")
    f.write(f"{len(rows)}\n")
    f.write(f"{len(cols)}\n")
    f.write("\n")
    for row in rows:
        f.write(f"{row}\n")
    for col in cols:
        f.write(f"{col}\n")
    for row in rows:
        line = "".join("x" if value else "." for value in df.loc[row])
        f.write(f"{line}\n")

parser = Parser()
cxt = parser.decode_cxt('eurovision_binary_context_clustered.cxt')
lattice = ConceptLattice.from_context(cxt)
G = lattice.to_networkx()

for node in G.nodes:
    print('songs:', lattice.get_concept_new_extent(node))

print('Formal Concepts:', len(G.nodes))
print('edges:', len(nx.transitive_reduction(G).edges))

ctx = FormalContext.from_file('eurovision_binary_context_clustered.cxt')
svg = ctx.draw_svg("sugiyama", width=800, height=600)
with open("esc_clustered.svg", "w") as f:
    f.write(svg)