import requests
import pandas as pd
from odis import FormalContext
import networkx as nx
from data import Parser
from fcapy.lattice import ConceptLattice

POLITICAL_CLUSTERS = {
    'EU Eurozone':           ['AT', 'BE', 'BG', 'HR', 'CY', 'EE', 'FI', 'FR', 'DE', 'GR',
                                'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PT', 'SK', 'SI', 'ES'],
    'EU Non-Eurozone':       ['CZ', 'DK', 'HU', 'PL', 'RO', 'SE'],
    'EU Candidates':           ['AL', 'BA', 'GE', 'MD', 'ME', 'MK', 'RS', 'TR', 'UA'],
    'EFTA / EEA':              ['CH', 'IS', 'NO'],
    'Post-Brexit':             ['GB', 'GB-WLS'],
    'Eurasian Economic Union': ['AM', 'BY', 'KZ', 'RU'],
    'Non-aligned':             ['AD', 'AZ', 'MC', 'SM'],
    'Non-European':            ['AU', 'IL', 'MA'],
    'Defunct States':          ['CS', 'YU'],
}

HISTORICAL_CLUSTERS = {
    'Former Soviet Union':         ['AM', 'AZ', 'BY', 'EE', 'GE', 'KZ', 'LV', 'LT', 'MD', 'RU', 'UA'],
    'Former Yugoslavia':           ['BA', 'CS', 'HR', 'ME', 'MK', 'RS', 'SI', 'YU'],
    'Former Eastern Bloc':         ['AL', 'BG', 'CZ', 'HU', 'PL', 'RO', 'SK'],
    'Western Bloc (NATO/aligned)': ['AU', 'BE', 'DE', 'DK', 'ES', 'FR', 'GB', 'GB-WLS',
                                    'GR', 'IL', 'IS', 'IT', 'LU', 'NL', 'NO', 'PT', 'TR'],
    'Neutral':                     ['AD', 'AT', 'CH', 'FI', 'IE', 'MC', 'SE', 'SM'],
    'Non-Aligned Movement':        ['CY', 'MA', 'MT'],
}

CULTURAL_CLUSTERS = {
    'Anglophone':          ['AU', 'GB', 'GB-WLS', 'IE'],
    'Nordic':              ['DK', 'FI', 'IS', 'NO', 'SE'],
    'Baltic':              ['EE', 'LV', 'LT'],
    'Germanic':            ['AT', 'BE', 'CH', 'DE', 'LU', 'NL'],
    'Romance':             ['AD', 'FR', 'IT', 'MC', 'MD', 'MT', 'PT', 'RO', 'SM', 'ES'],
    'East-Central Europe': ['CZ', 'HU', 'PL', 'SK', 'SI', 'HR'],
    'East Slavic':         ['BY', 'RU', 'UA'],
    'Balkan':              ['AL', 'BA', 'BG', 'CS', 'ME', 'MK', 'RS', 'YU'],
    'Hellenic':            ['CY', 'GR'],
    'Turkic':              ['AZ', 'KZ', 'TR'],
    'Caucasian':           ['AM', 'GE'],
    'Semitic':             ['IL', 'MA'],
}

REGIONAL_CLUSTERS = {
    'British Isles':     ['GB', 'GB-WLS', 'IE'],
    'Scandinavia':       ['DK', 'FI', 'IS', 'NO', 'SE'],
    'Baltic States':     ['EE', 'LV', 'LT'],
    'Benelux':           ['BE', 'NL', 'LU'],
    'Iberian Peninsula': ['AD', 'ES', 'FR', 'PT'],
    'Central Europe':    ['AT', 'CH', 'CZ', 'DE', 'HU', 'PL', 'SK'], 
    'Mediterranean':     ['IT', 'MC', 'MT', 'SM'],
    'Balkans':           ['AL', 'BA', 'CS', 'CY', 'GR', 'HR', 'ME', 'MK', 'RS', 'SI', 'TR', 'YU'],
    'Eastern Europe':    ['BY', 'BG', 'MD', 'RO', 'RU', 'UA'],
    'Caucasus':          ['AM', 'AZ', 'GE'],
    'Non-European':      ['AU', 'IL', 'KZ', 'MA'],
}

COUNTRIES = {
    'AL': 'Albania',
    'AD': 'Andorra',
    'AM': 'Armenia',
    'AU': 'Australia',
    'AT': 'Austria',
    'AZ': 'Azerbaijan',
    'BY': 'Belarus',
    'BE': 'Belgium',
    'BA': 'Bosnia and Herzegovina',
    'BG': 'Bulgaria',
    'HR': 'Croatia',
    'CY': 'Cyprus',
    'CZ': 'Czechia',
    'DK': 'Denmark',
    'EE': 'Estonia',
    'FI': 'Finland',
    'FR': 'France',
    'GE': 'Georgia',
    'DE': 'Germany',
    'GR': 'Greece',
    'HU': 'Hungary',
    'IS': 'Iceland',
    'IE': 'Ireland',
    'IL': 'Israel',
    'IT': 'Italy',
    'KZ': 'Kazakhstan',
    'LV': 'Latvia',
    'LT': 'Lithuania',
    'LU': 'Luxembourg',
    'MT': 'Malta',
    'MD': 'Moldova',
    'MC': 'Monaco',
    'ME': 'Montenegro',
    'MA': 'Morocco',
    'NL': 'Netherlands',
    'MK': 'North Macedonia',
    'NO': 'Norway',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'RU': 'Russia',
    'SM': 'San Marino',
    'RS': 'Serbia',
    'CS': 'Serbia and Montenegro',
    'SK': 'Slovakia',
    'SI': 'Slovenia',
    'ES': 'Spain',
    'SE': 'Sweden',
    'CH': 'Switzerland',
    'TR': 'Turkey',
    'UA': 'Ukraine',
    'GB': 'United Kingdom',
    'GB-WLS': 'Wales',
    'YU': 'Yugoslavia'
}

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
            total_score = next(
                (s for s in performance['scores'] if s.get('name') == 'total'), None
            )
            votes = total_score['votes']

        eligible_voters = set(votes.keys())

        winner_country = next(
            (c for c in data['contestants'] if c.get('id') == performance['contestantId']), None
        )['country']

        print(winner_country)
        regional_cluster = next((k for k, v in REGIONAL_CLUSTERS.items()  if winner_country in v), None)
        cultural_cluster = next((k for k, v in CULTURAL_CLUSTERS.items()  if winner_country in v), None)
        historical_cluster = next((k for k, v in HISTORICAL_CLUSTERS.items() if winner_country in v), None)
        political_cluster = next((k for k, v in POLITICAL_CLUSTERS.items() if winner_country in v), None)
        print(regional_cluster, cultural_cluster, historical_cluster, political_cluster)

        eligible_regional_members = [
            m for m in REGIONAL_CLUSTERS[regional_cluster]
            if m != winner_country and m in eligible_voters
        ]
        avg_regional_pts = 0.0
        if eligible_regional_members:
            avg_regional_pts = sum(votes.get(m, 0) for m in eligible_regional_members) / len(eligible_regional_members)
        regional_support = avg_regional_pts >= 8

        eligible_cultural_members = [
            m for m in CULTURAL_CLUSTERS[cultural_cluster]
            if m != winner_country and m in eligible_voters
        ]
        avg_cultural_pts = 0.0
        if eligible_cultural_members:
            avg_cultural_pts = sum(votes.get(m, 0) for m in eligible_cultural_members) / len(eligible_cultural_members)
        cultural_support = avg_cultural_pts >= 8

        eligible_historical_members = [
            m for m in HISTORICAL_CLUSTERS[historical_cluster]
            if m != winner_country and m in eligible_voters
        ]
        avg_historical_pts = 0.0
        if eligible_historical_members:
            avg_historical_pts = sum(votes.get(m, 0) for m in eligible_historical_members) / len(eligible_historical_members)
        historical_support = avg_historical_pts >= 8

        eligible_political_members = [
            m for m in POLITICAL_CLUSTERS[political_cluster]
            if m != winner_country and m in eligible_voters
        ]
        avg_political_pts = 0.0
        if eligible_political_members:
            avg_political_pts = sum(votes.get(m, 0) for m in eligible_political_members) / len(eligible_political_members)
        political_support = avg_political_pts >= 8

        print(regional_support, cultural_support, historical_support, political_support)

        records[f'{winner_country}_{year}'] = {
            'regional': regional_support,
            'cultural': cultural_support,
            'historical': historical_support,
            'political': political_support
        }

    except requests.exceptions.RequestException as e:
        print(f"  Request error for {year}: {e}")

support = ['regional','cultural','historical','political']
df = pd.DataFrame.from_dict(records, orient='index')[support].sort_index()
df.index.name   = 'year'
df.columns.name = 'support'

df = df.sort_index(axis=0)
df = df.sort_index(axis=1)

rows = list(df.index)
cols = list(df.columns)

with open("eurovision_support.cxt", "w", encoding="utf-8") as f:
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
cxt = parser.decode_cxt('eurovision_support.cxt')
lattice = ConceptLattice.from_context(cxt)
G = lattice.to_networkx()

for node in G.nodes:
    print('songs:', lattice.get_concept_new_extent(node))

print('Formal Concepts:', len(G.nodes))
print('edges:', len(nx.transitive_reduction(G).edges))

ctx = FormalContext.from_file('eurovision_support.cxt')
svg = ctx.draw_svg("dimdraw", width=800, height=600)
with open("esc_dimdraw.svg", "w") as f:
    f.write(svg)