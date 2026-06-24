POLITICAL_CLUSTERS = {
    'EU Eurozone':           ['AT', 'BE', 'BG', 'HR', 'CY', 'EE', 'FI', 'FR', 'DE', 'GR',
                                'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PT', 'SK', 'SI', 'ES'],
    'EU Non-Eurozone':       ['CZ', 'DK', 'HU', 'PL', 'RO', 'SE'],
    'EU Candidates':         ['AL', 'BA', 'GE', 'MD', 'ME', 'MK', 'RS', 'TR', 'UA'],
    'EFTA / EEA':            ['CH', 'IS', 'NO'],
    'Post-Brexit':           ['GB', 'GB-WLS'],
    'Eurasian Economic Union': ['AM', 'BY', 'KZ', 'RU'],
    'Non-aligned':           ['AD', 'AZ', 'MC', 'SM'],
    'Non-European':          ['AU', 'IL', 'MA'],
    'Defunct States':        ['CS', 'YU'],
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

import os, zipfile, urllib.request, io as _io
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as _Rect
from shapely.affinity import translate as _translate
from shapely.geometry import LineString, box as _box
from shapely.ops import split, unary_union

# ── color themes ──────────────────────────────────────────────────────────────

# Regional — blue/teal/indigo family, maximising lightness & sub-hue spread
REGIONAL_COLORS = {
    'British Isles':     '#001529',  # near-black navy
    'Scandinavia':       '#1a237e',  # deep indigo
    'Baltic States':     '#004d40',  # dark teal
    'Benelux':           '#1565c0',  # royal blue
    'Iberian Peninsula': '#00838f',  # medium-dark teal
    'Central Europe':    '#5c6bc0',  # medium indigo
    'Mediterranean':     '#039be5',  # bright ocean blue
    'Balkans':           '#42a5f5',  # light blue
    'Eastern Europe':    '#b3e5fc',  # very light blue
    'Caucasus':          '#80deea',  # pale cyan
    'Non-European':      '#546e7a',  # slate
}

# Political — red/orange/amber family, maximising lightness & hue spread
POLITICAL_COLORS = {
    'EU Eurozone':             '#3e0000',  # near-black crimson
    'EU Non-Eurozone':         '#c62828',  # dark red
    'EU Candidates':           '#ff1744',  # vivid red
    'EFTA / EEA':              '#bf360c',  # dark burnt orange
    'Post-Brexit':             '#ff6d00',  # vivid orange
    'Eurasian Economic Union': '#f9a825',  # golden amber
    'Non-aligned':             '#fff176',  # pale yellow
    'Non-European':            '#546e7a',  # slate
    'Defunct States':          '#9e9e9e',  # grey
}

# Historical — green family, maximising lightness & hue spread (6 clusters)
HISTORICAL_COLORS = {
    'Former Soviet Union':         '#0a1f0a',  # near-black forest
    'Former Yugoslavia':           '#1b5e20',  # dark forest green
    'Former Eastern Bloc':         '#558b2f',  # olive green
    'Western Bloc (NATO/aligned)': '#00c853',  # vivid green
    'Neutral':                     '#c5e1a5',  # pale lime
    'Non-Aligned Movement':        '#00897b',  # dark teal (strong hue contrast)
}

# Cultural — violet/purple/magenta/pink family, maximising lightness & hue spread
CULTURAL_COLORS = {
    'Anglophone':          '#1a0030',  # near-black violet
    'Nordic':              '#4527a0',  # dark violet
    'Baltic':              '#7c4dff',  # vivid blue-violet
    'Germanic':            '#9c27b0',  # medium purple
    'East-Central Europe': '#e1bee7',  # pale lavender
    'Romance':             '#4a0020',  # near-black wine
    'East Slavic':         '#880e4f',  # dark wine
    'Balkan':              '#d500f9',  # vivid magenta
    'Hellenic':            '#e91e63',  # hot pink
    'Turkic':              '#fce4ec',  # pale blush
    'Caucasian':           '#ad1457',  # raspberry
    'Semitic':             '#546e7a',  # slate
}

# ── ISO mappings ──────────────────────────────────────────────────────────────

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

# ── load & project world geometry once ───────────────────────────────────────

_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ne_50m_cache')
_shp = os.path.join(_cache_dir, 'ne_50m_admin_0_countries.shp')
if not os.path.exists(_shp):
    os.makedirs(_cache_dir, exist_ok=True)
    _url = 'https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip'
    with urllib.request.urlopen(_url) as _r:
        with zipfile.ZipFile(_io.BytesIO(_r.read())) as _z:
            _z.extractall(_cache_dir)

_world_raw = gpd.read_file(_shp)
_world_raw = _world_raw[['ISO_A3_EH', 'NAME', 'geometry']].rename(
    columns={'ISO_A3_EH': 'iso_a3', 'NAME': 'name'})
_world_raw = _world_raw.clip([-35, -50, 180, 82])

def _cut_at_antimeridian(geom, lon_0=15):
    geom = geom.buffer(0)
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

_world_raw = _world_raw.explode(index_parts=False).reset_index(drop=True)
_world_raw['geometry'] = _world_raw['geometry'].apply(_cut_at_antimeridian)
_world_proj = _world_raw.to_crs('+proj=robin +lon_0=15 +datum=WGS84')
_world_proj = _world_proj.explode(index_parts=False).reset_index(drop=True)

_b = _world_proj.geometry.bounds
_xspan = _b['maxx'] - _b['minx']
_yspan = (_b['maxy'] - _b['miny']).clip(lower=1)
_world_proj = _world_proj[(_xspan / _yspan) < 50].reset_index(drop=True)

_extent_proj = gpd.GeoDataFrame(
    geometry=[_box(-15, 30, 52, 71)], crs='EPSG:4326'
).to_crs('+proj=robin +lon_0=15 +datum=WGS84')
_xmin, _ymin, _xmax, _ymax = _extent_proj.total_bounds

# ── map function ──────────────────────────────────────────────────────────────

def make_cluster_map(cluster_dict, colors, filename):
    ctc = {
        ISO2_TO_ISO3[code]: cluster
        for cluster, codes in cluster_dict.items()
        for code in codes
        if code in ISO2_TO_ISO3
    }
    wp = _world_proj.copy()
    wp['color'] = wp['iso_a3'].map(ctc).map(colors).fillna('#dddddd')

    fig, ax = plt.subplots(figsize=(12, 9))
    wp.plot(ax=ax, color=wp['color'], edgecolor='white', linewidth=0.4)
    ax.set_xlim(_xmin, _xmax)
    ax.set_ylim(_ymin, _ymax)
    ax.axis('off')
    plt.tight_layout(pad=0.5)

    # Australia inset — top-right corner
    _aus = wp[wp['iso_a3'] == 'AUS']
    ax_inset = fig.add_axes([0.728, 0.753, 0.31, 0.22])
    ax_inset.add_patch(_Rect((0, 0), 1, 1, transform=ax_inset.transAxes,
                              facecolor='white', edgecolor='none', zorder=0))
    if len(_aus) > 0:
        _aus.plot(ax=ax_inset, color=_aus['color'].iloc[0], edgecolor='white', linewidth=0.4)
        _bb = _aus.geometry.iloc[_aus.geometry.area.argmax()].bounds
        _pad = min(_bb[2] - _bb[0], _bb[3] - _bb[1]) * 0.04
        ax_inset.set_xlim(_bb[0] - _pad, _bb[2] + _pad)
        ax_inset.set_ylim(_bb[1] - _pad, _bb[3] + _pad)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for _spine in ax_inset.spines.values():
        _spine.set_linewidth(2.0)
        _spine.set_edgecolor('#555555')

    plt.savefig(filename, bbox_inches='tight')
    plt.show()

# ── generate all four maps ────────────────────────────────────────────────────

make_cluster_map(REGIONAL_CLUSTERS,   REGIONAL_COLORS,   'regional_map.pdf')
make_cluster_map(POLITICAL_CLUSTERS,  POLITICAL_COLORS,  'political_map.pdf')
make_cluster_map(HISTORICAL_CLUSTERS, HISTORICAL_COLORS, 'historical_map.pdf')
make_cluster_map(CULTURAL_CLUSTERS,   CULTURAL_COLORS,   'cultural_map.pdf')
