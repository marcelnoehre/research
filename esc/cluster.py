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
    'Northern Europe':   ['IS', 'NO', 'SE', 'DK', 'FI', 'EE', 'LV', 'LT'],
    'Western Europe':    ['GB', 'GB-WLS', 'IE', 'BE', 'NL', 'LU', 'DE', 'AT', 'CH', 'FR'],
    'Southern Europe':   ['PT', 'ES', 'AD', 'IT', 'SM', 'MC', 'GR', 'CY', 'MT'],
    'Eastern Europe':    ['PL', 'CZ', 'SK', 'HU', 'RU', 'BY', 'UA', 'MD'],
    'Balkans':           ['AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'SI', 'BG', 'RO', 'YU', 'CS'],
    'Near East & Other': ['TR', 'IL', 'GE', 'AM', 'AZ', 'MA', 'KZ', 'AU'],
}

THRESHOLD_RATIO = 0.75

# ---------------------------------------------------------------------------
# Support definition
# ---------------------------------------------------------------------------

def gave_support(country: str, votes: dict, year: int) -> bool:
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

        if year >= 2016:
            jury_score = next(
                (s for s in performance['scores'] if s.get('name') == 'jury'), None
            )
            tele_score = next(
                (s for s in performance['scores'] if s.get('name') == 'public'), None
            )
            print(jury_score)
            print(tele_score)
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
            if not eligible_members:
                cluster_totals[cluster] = 0.0
                continue
            supporters = sum(1 for m in eligible_members if gave_support(m, votes, year))
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
with open("esc_clustered.svg", "w") as f:
    f.write(svg)