import requests
import pandas as pd
from odis import FormalContext
import networkx as nx
from data import Parser
from fcapy.lattice import ConceptLattice

POLITICAL_CLUSTERS = {
    'European Union':  ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'],
    'EU Candidates':   ['AL', 'BA', 'GE', 'MD', 'ME', 'MK', 'RS', 'TR', 'UA'],
    'EFTA / EEA':      ['CH', 'IS', 'NO'],
    'Post-Brexit':     ['GB', 'GB-WLS'],
    'Authoritarian':   ['AZ', 'BY', 'KZ', 'RU'],
    'Non-aligned':     ['AD', 'AM', 'MC', 'SM'],
    'Non-European':    ['AU', 'IL', 'MA'],
    'Defunct States':  ['CS', 'YU'],
}

HISTORICAL_CLUSTERS = {
    'Former Soviet Union':    ['AM', 'AZ', 'BY', 'EE', 'GE', 'KZ', 'LV', 'LT', 'MD', 'RU', 'UA'],
    'Former Yugoslavia':      ['BA', 'CS', 'HR', 'ME', 'MK', 'RS', 'SI', 'YU'],
    'Former Eastern Bloc':    ['AL', 'BG', 'CZ', 'HU', 'PL', 'RO', 'SK'],
    'Western Bloc':           ['BE', 'DE', 'DK', 'ES', 'FR', 'GB', 'GB-WLS', 'GR', 'IS', 'IT', 'LU', 'NL', 'NO', 'PT', 'TR'],
    'Neutral':                ['AT', 'CH', 'FI', 'IE', 'SE'],
    'Mediterranean & Microstates': ['AD', 'CY', 'MC', 'MT', 'SM'],
    'Non-European':           ['AU', 'IL', 'MA'],
}

CULTURAL_CLUSTERS = {
    'Anglophone':   ['AU', 'GB', 'GB-WLS', 'IE'],
    'Nordic':       ['DK', 'IS', 'NO', 'SE'],
    'Finno-Ugric':  ['EE', 'FI', 'HU'],
    'Germanic':     ['AT', 'BE', 'CH', 'DE', 'LU', 'NL'],
    'Romance':      ['AD', 'FR', 'IT', 'MC', 'MD', 'MT', 'PT', 'RO', 'SM', 'ES'],
    'Baltic':       ['LV', 'LT'],
    'West Slavic':  ['CZ', 'PL', 'SK'],
    'East Slavic':  ['BY', 'RU', 'UA'],
    'Balkan':       ['AL', 'BA', 'BG', 'CS', 'HR', 'ME', 'MK', 'RS', 'SI', 'YU'],
    'Hellenic':     ['CY', 'GR'],
    'Turkic':       ['AZ', 'KZ', 'TR'],
    'Caucasian':    ['AM', 'GE'],
    'Semitic':      ['IL', 'MA'],
}

REGIONAL_CLUSTERS = {
    'British Isles':    ['GB', 'GB-WLS', 'IE'],
    'Scandinavia':      ['DK', 'FI', 'IS', 'NO', 'SE'],
    'Baltic States':    ['EE', 'LV', 'LT'],
    'Benelux':          ['BE', 'NL', 'LU'],
    'Iberian Peninsula':['AD', 'ES', 'PT'],
    'Central Europe':   ['AT', 'CH', 'DE', 'FR', 'MC'],
    'Mediterranean':    ['CY', 'GR', 'IT', 'MT', 'SM'],
    'Balkans':          ['AL', 'BA', 'CS', 'HR', 'ME', 'MK', 'RS', 'SI', 'TR', 'YU'],
    'Eastern Europe':   ['BY', 'BG', 'CZ', 'HU', 'MD', 'PL', 'RO', 'RU', 'SK', 'UA'],
    'Caucasus':         ['AM', 'AZ', 'GE'],
    'Non-European':     ['AU', 'IL', 'KZ', 'MA'],
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



    except requests.exceptions.RequestException as e:
        print(f"  Request error for {year}: {e}")