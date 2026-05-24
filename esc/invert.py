import requests
import pandas as pd

from odis import FormalContext
import networkx as nx
from data import Parser
from fcapy.lattice import ConceptLattice

# ---------------------------------------------------------------------------
# Empirically grounded voting blocs (Gatherer 2006 / Fenn et al.)
# ---------------------------------------------------------------------------
CLUSTERS = {
    'Nordic':                ['IS', 'NO', 'SE', 'DK', 'FI'],
    'British Isles':         ['GB', 'GB-WLS', 'IE'],
    'Benelux':               ['BE', 'NL', 'LU'],
    'DACH':                  ['DE', 'AT', 'CH'],
    'Iberia':                ['PT', 'ES', 'AD'],
    'Italian Peninsula':     ['IT', 'SM', 'MC'],
    'Balkans':               ['AL', 'BA', 'BG', 'HR', 'ME', 'MK', 'RO', 'RS', 'SI'],
    'Visegrad':              ['PL', 'CZ', 'SK', 'HU'],
    'Baltics':               ['EE', 'LV', 'LT'],
    'Eastern Europe':        ['RU', 'BY', 'UA', 'MD'],
    'Caucasus':              ['GE', 'AM', 'AZ'],
    'Eastern Mediterranean': ['GR', 'CY', 'TR', 'IL', 'MT'],
    'Non-European':          ['MA', 'KZ', 'AU'],
    'Defunct':               ['YU', 'CS'],
}

# Reverse lookup: country → cluster
COUNTRY_TO_CLUSTER = {
    country: cluster
    for cluster, members in CLUSTERS.items()
    for country in members
}

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
# Structure we're building:
#   winner_cluster_by_year[year] = cluster name (or 'Other')
#   support_by_year[year]        = {cluster: total_points_received}

winner_cluster_by_year = {}
support_by_year = {}

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
        winner_country = performance.get('country', '')
        winner_cluster = COUNTRY_TO_CLUSTER.get(winner_country, 'Other')
        winner_cluster_by_year[year] = winner_cluster

        total_score = next(
            (s for s in performance['scores'] if s.get('name') == 'total'), None
        )
        if total_score is None:
            print(f"  No total score for {year}, skipping.")
            continue

        votes = total_score['votes']  # {country_code: points}

        # Sum points per cluster, excluding the winner's own country
        cluster_totals = {
            cluster: sum(
                votes.get(m, 0) for m in members
                if m != winner_country
            )
            for cluster, members in CLUSTERS.items()
        }

        support_by_year[year] = cluster_totals

    except requests.exceptions.RequestException as e:
        print(f"  Request error for {year}: {e}")

# ---------------------------------------------------------------------------
# Binarise using within-year rank (top half = 1)
# Era-normalised: unaffected by expanding voter pool over time
# ---------------------------------------------------------------------------
binary_by_year = {}
for year, totals in support_by_year.items():
    sorted_scores = sorted(totals.values(), reverse=True)
    median_score = sorted_scores[len(sorted_scores) // 2]
    binary_by_year[year] = {
        cluster: 1 if score >= median_score else 0
        for cluster, score in totals.items()
    }

# ---------------------------------------------------------------------------
# Build inverted context:
#   objects   = clusters
#   attributes = years  +  winner_origin_<cluster> flags
# ---------------------------------------------------------------------------
cluster_names = list(CLUSTERS.keys())
years = sorted(binary_by_year.keys())

# Winner-origin attributes: one per cluster, true in years that cluster won
winner_attrs = {
    f'won_{c}': {y for y, wc in winner_cluster_by_year.items() if wc == c}
    for c in cluster_names
}
# Drop winner attrs for clusters that never won (keeps the context clean)
winner_attrs = {k: v for k, v in winner_attrs.items() if v}

all_attrs = [str(y) for y in years] + list(winner_attrs.keys())

# Object (cluster) × attribute (year / won_X) incidence
rows_dict = {}
for cluster in cluster_names:
    row = {}
    # Year attributes: did this cluster support the winner that year?
    for year in years:
        row[str(year)] = binary_by_year.get(year, {}).get(cluster, 0)
    # Winner-origin attributes: did this cluster's own act win that year?
    for attr, winning_years in winner_attrs.items():
        row[attr] = 1 if year in winning_years else 0  # note: evaluated per cluster below
    rows_dict[cluster] = row

# Rebuild correctly — winner_attrs are cluster-level not year-level
for cluster in cluster_names:
    for attr, winning_years in winner_attrs.items():
        rows_dict[cluster][attr] = 1 if any(
            winner_cluster_by_year.get(y) == cluster
            for y in winning_years
        ) else 0

df = pd.DataFrame.from_dict(rows_dict, orient='index')[all_attrs].fillna(0).astype(int)
df.index.name   = 'cluster'
df.columns.name = 'attribute'

rows = list(df.index)
cols = list(df.columns)

# ---------------------------------------------------------------------------
# Write Burmeister .cxt
# ---------------------------------------------------------------------------
with open("eurovision_inverted_context.cxt", "w", encoding="utf-8") as f:
    f.write("B\n\n")
    f.write(f"{len(rows)}\n")
    f.write(f"{len(cols)}\n\n")
    for row in rows:
        f.write(f"{row}\n")
    for col in cols:
        f.write(f"{col}\n")
    for row in rows:
        line = "".join("x" if df.loc[row, col] else "." for col in cols)
        f.write(f"{line}\n")

print("Burmeister .cxt file created: eurovision_inverted_context.cxt")
print(f"\n{len(rows)} objects (clusters) × {len(cols)} attributes (years + winner flags)")
print(f"\nWinner origin distribution:")
for cluster, count in pd.Series(winner_cluster_by_year).value_counts().items():
    print(f"  {cluster}: {count} wins")

parser = Parser()
cxt = parser.decode_cxt(f'eurovision_inverted_context.cxt')
lattice = ConceptLattice.from_context(cxt)
G = lattice.to_networkx()

for node in G.nodes:
    print('songs:', lattice.get_concept_new_extent(node))

print('Formal Concepts:', len(G.nodes))
print('edges:', len(nx.transitive_reduction(G).edges))

ctx = FormalContext.from_file('eurovision_inverted_context.cxt')
svg = ctx.draw_svg("sugiyama", width=800, height=600)
with open("esc_interted.svg", "w") as f:
    f.write(svg)
