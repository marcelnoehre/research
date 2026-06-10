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

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLUSTER_COLORS = {
    'British Isles':     '#c0392b',  # crimson
    'Scandinavia':       '#2980b9',  # blue
    'Baltic States':     '#27ae60',  # green
    'Benelux':           '#8e44ad',  # purple
    'Iberian Peninsula': '#e67e22',  # orange
    'Central Europe':    '#f1c40f',  # yellow
    'Mediterranean':     '#16a085',  # teal
    'Balkans':           '#e91e63',  # pink
    'Eastern Europe':    '#795548',  # brown
    'Caucasus':          '#00bcd4',  # cyan
    'Non-European':      '#546e7a',  # slate — was orange, now clearly distinct
}

ISO2_TO_ISO3 = {
    'AL': 'ALB', 'AD': 'AND', 'AM': 'ARM', 'AU': 'AUS', 'AT': 'AUT',
    'AZ': 'AZE', 'BY': 'BLR', 'BE': 'BEL', 'BA': 'BIH', 'BG': 'BGR',
    'HR': 'HRV', 'CY': 'CYP', 'CZ': 'CZE', 'DK': 'DNK', 'EE': 'EST',
    'FI': 'FIN', 'FR': 'FRA', 'GE': 'GEO', 'DE': 'DEU', 'GR': 'GRC',
    'HU': 'HUN', 'IS': 'ISL', 'IE': 'IRL', 'IL': 'ISR', 'IT': 'ITA',
    'KZ': 'KAZ', 'LV': 'LVA', 'LT': 'LTU', 'LU': 'LUX', 'MT': 'MLT',
    'MD': 'MDA', 'MC': 'MCO', 'ME': 'MNE', 'MA': 'MAR', 'NL': 'NLD',
    'MK': 'MKD', 'NO': 'NOR', 'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU',
    'RU': 'RUS', 'SM': 'SMR', 'RS': 'SRB', 'SK': 'SVK', 'SI': 'SVN',
    'ES': 'ESP', 'SE': 'SWE', 'CH': 'CHE', 'TR': 'TUR', 'UA': 'UKR',
    'GB': 'GBR', 'GB-WLS': 'GBR',
}

country_to_cluster = {
    ISO2_TO_ISO3[code]: cluster
    for cluster, codes in REGIONAL_CLUSTERS.items()
    for code in codes
    if code in ISO2_TO_ISO3
}

_shp = '/home/mno/Documents/kde/research/.venv/lib/python3.12/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp'
world = gpd.read_file(_shp)
# clip to ESC-relevant extent — excludes Americas/Pacific overseas territories
world = world.clip([-35, -50, 180, 82])
# fix known -99 iso_a3 entries in this dataset
_name_fixes = {'France': 'FRA', 'Norway': 'NOR'}
world.loc[world['iso_a3'] == '-99', 'iso_a3'] = world.loc[world['iso_a3'] == '-99', 'name'].map(_name_fixes)
world['cluster'] = world['iso_a3'].map(country_to_cluster)
world['color'] = world['cluster'].map(CLUSTER_COLORS).fillna('#dddddd')

from shapely.affinity import translate as _translate
from shapely.geometry import LineString
from shapely.ops import split, unary_union

def _cut_at_antimeridian(geom, lon_0=15):
    geom = geom.buffer(0)  # fix invalid geometry
    for cut_lon, xoff, check in [
        (lon_0 - 180,  360, lambda cx: cx < lon_0 - 180),
        (lon_0 + 180, -360, lambda cx: cx > lon_0 + 180),
    ]:
        cut = LineString([(cut_lon, -91), (cut_lon, 91)])
        try:
            parts = split(geom, cut)
            if len(list(parts.geoms)) > 1:
                geom = unary_union([
                    _translate(p, xoff=xoff) if check(p.centroid.x) else p
                    for p in parts.geoms
                ])
        except Exception:
            pass
    return geom

world = world.explode(index_parts=False).reset_index(drop=True)
world['geometry'] = world['geometry'].apply(_cut_at_antimeridian)
world_proj = world.to_crs('+proj=robin +lon_0=15 +datum=WGS84')
world_proj = world_proj.explode(index_parts=False).reset_index(drop=True)

# drop artifact strips: real geography is never >50:1 wide vs tall
_b = world_proj.geometry.bounds
_xspan = _b['maxx'] - _b['minx']
_yspan = (_b['maxy'] - _b['miny']).clip(lower=1)
world_proj = world_proj[(_xspan / _yspan) < 50].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(18, 9))
world_proj.plot(ax=ax, color=world_proj['color'], edgecolor='white', linewidth=0.4)

patches = [mpatches.Patch(facecolor=c, label=k) for k, c in CLUSTER_COLORS.items()]
ax.legend(handles=patches, loc='lower left', fontsize=9, framealpha=0.8)
ax.set_title('ESC Countries by Regional Cluster', fontsize=14, pad=12)
ax.axis('off')

plt.tight_layout()
plt.savefig('regional_map.svg', bbox_inches='tight')
plt.show()