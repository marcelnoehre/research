import requests
import pandas as pd
from odis import FormalContext
import networkx as nx
from data import Parser
from fcapy.lattice import ConceptLattice

CLUSTERS = [
    ['1979', '1989', '1988', '1975'],
    ['1998', '1976', '1996', '1978'],
    ['1983', '1977', '1995'],
    ['1981', '1985', '2005', '2010'],
    ['1981', '1985', '2005', '2010'],
    ['2007'],
    ['1990', '2011'],
    ['1982'],
    ['1997', '1993', '1992', '2000', '2004'],
    ['2018', '1986', '2022'],
    ['1991', '2021', '1999'],
    ['2025', '2014'],
    ['2001', '1994', '2016', '1980', '1984', '2012', '2023', '2013', '2024', '1987', '2017', '2002', '2009', '2006', '2019', '2008', '2015'],
]


for i, cluster in enumerate(CLUSTERS):
    records = {}
    for year in cluster:
        try:
            print(year)
            url = f'https://eurovisionapi.runasp.net/api/senior/contests/{year}/contestants/0' # winning song
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            is_fast = data['bpm'] > 116
            is_major = 'major' in data['tone']
            has_backings = data['backings'] is not None
            has_dancers = data['dancers'] is not None
            is_english = data['lyrics'][0]['languages'][0] == 'english'
            records[year] = {
                'is_fast': is_fast,
                'is_major': is_major,
                'has_backings': has_backings,
                'has_dancers': has_dancers,
                'is_english': is_english
            }

        except requests.exceptions.RequestException as e:
            print(f"  Request error for {year}: {e}")

    support = ['is_fast','is_major','has_backings','has_dancers', 'is_english']
    df = pd.DataFrame.from_dict(records, orient='index')[support].sort_index()
    df.index.name   = 'year'
    df.columns.name = 'support'

    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1)

    rows = list(df.index)
    cols = list(df.columns)

    with open(f"cluster_{i}.cxt", "w", encoding="utf-8") as f:
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