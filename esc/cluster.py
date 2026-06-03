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
    'A':   ['CH', 'FI', 'IT'],
    'B':    ['FR', 'GB', 'IE', 'IL', 'LU', 'NL'],
    'C':   ['DE', 'DK', 'ES', 'IL', 'IT'],
    'D':    ['CY', 'GR', 'PT'],
    'E':           ['EE', 'IE', 'IS', 'SE'],
    'F': ['HR', 'MT', 'PT', 'SK', 'TR'],
    'G':    ['BE', 'DE', 'NL'],
    'H':    ['DK', 'NO', 'SE'],
    'I':    ['AM', 'BA', 'BG', 'BY', 'GE', 'GR', 'HU', 'MD', 'MK', 'RO', 'RS', 'RU', 'SI', 'TR', 'UA'],
}

THRESHOLD_RATIO = 0.9

# ---------------------------------------------------------------------------
# Support definition
# ---------------------------------------------------------------------------

def gave_support(country: str, votes: dict) -> bool:
    """Return True if the country awarded 8, 10, or 12 points (highlighted on TV)."""
    pts = votes.get(country, 0)
    return pts in (8, 10, 12)

# ---------------------------------------------------------------------------
# Fetch & build matrix
# ---------------------------------------------------------------------------

records = {}

for year in range(1975, 2026):
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
        performance = final_round['performances'][0] # winner
        total_score = next(
            (s for s in performance['scores'] if s.get('name') == 'total'), None
        )

        if year >= 2016:
            jury_score = next(
                (s for s in performance['scores'] if s.get('name') == 'jury'), None
            )
            tele_score = next(
                (s for s in performance['scores'] if s.get('name') == 'public'), None
            )
            jury_votes = jury_score['votes'] if jury_score else {}
            tele_votes = tele_score['votes'] if tele_score else {}
            all_voters = set(jury_votes) | set(tele_votes)
            votes = {
                country: max(jury_votes.get(country, 0), tele_votes.get(country, 0))
                for country in all_voters
            }
        else:
            votes = total_score['votes']

        eligible_voters = set(votes.keys())

        winner_country = next(
            (c for c in data['contestants'] if c.get('id') == performance['contestantId']), None
        )['country']

        cluster_totals = {}
        for cluster, members in CLUSTERS.items():
            eligible_members = [
                m for m in members
                if m != winner_country and m in eligible_voters
            ]

            # cluster not represented
            if not eligible_members: 
                cluster_totals[cluster] = 0.0
                continue

            supporters = sum(1 for m in eligible_members if gave_support(m, votes))
            cluster_totals[cluster] = supporters / len(eligible_members)

        records[f'{winner_country}_{year}'] = {
            cluster: 1 if total >= THRESHOLD_RATIO else 0
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
with open("esc_sugiyama.svg", "w") as f:
    f.write(svg)