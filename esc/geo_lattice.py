"""
Geo-Political Concept Lattice — Eurovision Dark Horse Supporter Network
=======================================================================
Lattice is primary structure. Geography anchors the horizontal axis.
The 2-country layer is split into western/eastern sub-rows to reduce
crowding. A clipped Europe+Caucasus map sits behind.

  y-axis  = lattice layer (concept specificity)
  x-axis  = geographic longitude centroid of concept members
  Dotted lines connect country nodes to their true capital positions
  Red dashed arc = Armenia–Azerbaijan armed conflict
"""

import os, io, warnings, urllib.request
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from shapely.geometry import box as sbox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#07070f"
SURFACE = "#0f0f25"
MID     = "#202040"
GREY    = "#5858a0"
WHITE   = "#e8e8f5"
ACCENT1 = "#e8c84a"
ACCENT2 = "#5ec9b8"
ACCENT3 = "#c96aa0"

COUNTRY_COLORS = {
    "Italy":          "#e8c84a",
    "Armenia":        "#d06090",
    "Georgia":        "#50c8b0",
    "Moldova":        "#f07840",
    "Azerbaijan":     "#70c050",
    "Russia":         "#d05050",
    "Montenegro":     "#5090d8",
    "Switzerland":    "#9070d8",
    "Australia":      "#50c8d8",
    "Czech Republic": "#d09050",
    "Greece":         "#d0d050",
}

# True geographic centroids (lon, lat) — capital/centroid of country
GEO = {
    "Italy":          (12.5,  41.9),   # Rome
    "Armenia":        (44.5,  40.2),   # Yerevan
    "Georgia":        (44.8,  41.7),   # Tbilisi
    "Moldova":        (28.9,  47.0),   # Chisinau
    "Azerbaijan":     (49.9,  40.4),   # Baku
    "Russia":         (37.6,  55.8),   # Moscow
    "Montenegro":     (19.3,  42.4),   # Podgorica
    "Switzerland":    ( 7.4,  46.9),   # Bern
    "Australia":      (149.1,-35.3),   # Canberra — off-map, special handling
    "Czech Republic": (14.4,  50.1),   # Prague
    "Greece":         (23.7,  37.9),   # Athens
}

# Map extent
LON0, LON1 = -2,  58
LAT0, LAT1 = 36,  62

# Layer latitude assignments — two sub-rows for the crowded 2-country layer
LAYER_LAT = {
    "top":    66.0,   # ALL SONGS
    1:        59.5,   # 1-country nodes
    "2w":     55.5,   # 2-country, western pairs (centroid lon < 31)
    "2e":     51.5,   # 2-country, eastern pairs
    3:        47.0,   # 3-country concepts
    4:        43.5,   # 4-country concepts
    5:        40.5,   # 5-country concepts
    6:        38.5,   # 6-country concepts
    7:        36.5,   # 7-country concepts
    8:        35.0,   # 8-country concepts
    9:        34.5,   # 9-country concepts
    10:       34.0,   # 10-country concepts
    "bottom": 33.0,   # ∅ no consensus
}

GEO_SPLIT = 31.0   # longitude split for 2-country sub-rows


# ── Data ──────────────────────────────────────────────────────────────────────
SUPPLEMENT = {
    2023: {
        "results": "country,rank\nSweden,1\nFinland,2\nIsrael,3\nNorway,4\nItaly,5\nMoldova,6\nUkraine,7\nBelgium,8\nEstonia,9\nAustralia,10\nAustria,11\nPoland,12\nSlovenia,13\nSerbia,14\nCroatia,15\nFrance,16\nGermany,17\nCyprus,18\nArmenia,19\nAlbania,20\nLithuania,21\nCzech Republic,22\nPortugal,23\nSpain,24\nUnited Kingdom,25\nSwitzerland,26",
        "votes":   "from_country,to_country,points\nAlbania,Moldova,12\nArmenia,Moldova,12\nArmenia,Italy,10\nAzerbaijan,Moldova,12\nCroatia,Moldova,12\nFrance,Italy,12\nFrance,Belgium,10\nGeorgia,Moldova,12\nGeorgia,Ukraine,10\nItaly,France,12\nItaly,Moldova,10\nMalta,Italy,12\nMoldova,Ukraine,12\nMoldova,Armenia,10\nMontenegro,Serbia,12\nMontenegro,Croatia,10\nPortugal,Italy,12\nRomania,Moldova,12\nSan Marino,Italy,12\nSpain,Italy,12\nSpain,France,10\nSwitzerland,Italy,12\nSwitzerland,Moldova,10\nUkraine,Moldova,12"
    },
    2024: {
        "results": "country,rank\nSwitzerland,1\nCroatia,2\nUkraine,3\nFrance,4\nIsrael,5\nLuxembourg,6\nArmenia,7\nIreland,8\nLithuania,9\nItaly,10\nSerbia,11\nGreece,12\nGeorgia,13\nSweden,14\nNorway,15\nFinland,16\nPortugal,17\nEstonia,18\nMalta,19\nAlbania,20\nSlovenia,21\nCyprus,22\nNetherlands,23\nAustria,24\nUnited Kingdom,25\nGermany,26",
        "votes":   "from_country,to_country,points\nAlbania,Serbia,12\nAlbania,Italy,10\nArmenia,France,12\nArmenia,Greece,10\nAzerbaijan,Armenia,12\nAzerbaijan,Georgia,10\nCroatia,Serbia,12\nCyprus,Greece,12\nCyprus,Armenia,10\nCzech Republic,Armenia,10\nFrance,Luxembourg,12\nGeorgia,Armenia,12\nGeorgia,Ukraine,10\nGreece,Armenia,10\nItaly,France,12\nItaly,Armenia,10\nMoldova,Armenia,12\nMoldova,Georgia,10\nMontenegro,Serbia,12\nMontenegro,Croatia,10\nRomania,Moldova,12\nRomania,Armenia,10\nSerbia,Montenegro,12\nSlovenia,Croatia,12\nSwitzerland,France,12\nUkraine,Armenia,12\nUkraine,Georgia,10"
    },
    2025: {
        "results": "country,rank\nAustria,1\nIsrael,2\nEstonia,3\nFinland,4\nMalta,5\nPortugal,6\nItaly,7\nSweden,8\nAlbania,9\nLatvia,10\nAustralia,11\nUkraine,12\nGreece,13\nUnited Kingdom,14\nLithuania,15\nGermany,16\nFrance,17\nSan Marino,18\nNorway,19\nPoland,20\nDenmark,21\nIceland,22\nSlovenia,23\nCroatia,24\nGeorgia,25\nArmenia,26",
        "votes":   "from_country,to_country,points\nAlbania,Greece,12\nAlbania,Italy,10\nArmenia,Greece,12\nAzerbaijan,Armenia,12\nAzerbaijan,Georgia,10\nCroatia,Slovenia,12\nCyprus,Greece,12\nFrance,Italy,10\nGeorgia,Armenia,12\nGeorgia,Latvia,10\nGermany,Austria,12\nItaly,France,12\nItaly,Albania,10\nLatvia,Estonia,12\nLithuania,Latvia,12\nMoldova,Romania,12\nMoldova,Ukraine,10\nMontenegro,Serbia,12\nMontenegro,Croatia,10\nPortugal,Italy,12\nRomania,Moldova,12\nSan Marino,Italy,12\nSerbia,Montenegro,12\nSlovenia,Croatia,12\nSwitzerland,Austria,12\nSwitzerland,France,10\nUkraine,Moldova,12"
    }
}


def build_context(n_countries=8, min_championed_by=3):
    """
    Fetch all Grand Final voting data directly from the Eurovision API.
    https://eurovisionapi.runasp.net/api/senior/contests/{year}

    Caches each year as JSON in esc_api_cache/ next to this script.
    Covers 1975–2025 (skips 2020, no contest held).
    """
    import urllib.request, json, time

    CC = {
        # Standard ISO codes
        "AL":"Albania","AD":"Andorra","AM":"Armenia","AU":"Australia",
        "AT":"Austria","AZ":"Azerbaijan","BY":"Belarus","BE":"Belgium",
        "BA":"Bosnia & Herzegovina","BG":"Bulgaria","HR":"Croatia",
        "CY":"Cyprus","CZ":"Czech Republic","DK":"Denmark","EE":"Estonia",
        "FI":"Finland","FR":"France","GE":"Georgia","DE":"Germany",
        "GR":"Greece","HU":"Hungary","IS":"Iceland","IE":"Ireland",
        "IL":"Israel","IT":"Italy","LV":"Latvia","LT":"Lithuania",
        "LU":"Luxembourg","MT":"Malta","MD":"Moldova","MC":"Monaco",
        "ME":"Montenegro","MA":"Morocco","NL":"Netherlands","MK":"North Macedonia",
        "NO":"Norway","PL":"Poland","PT":"Portugal","RO":"Romania",
        "RU":"Russia","SM":"San Marino","RS":"Serbia","SK":"Slovakia",
        "SI":"Slovenia","ES":"Spain","SE":"Sweden","CH":"Switzerland",
        "TR":"Turkey","UA":"Ukraine","GB":"United Kingdom","YU":"Yugoslavia",
        "CS":"Serbia and Montenegro","KZ":"Kazakhstan","XK":"Kosovo",
        # UK sub-national codes (all map to United Kingdom)
        "GB-ENG":"United Kingdom","GB-WLS":"United Kingdom",
        "GB-SCO":"United Kingdom","GB-NIR":"United Kingdom",
        # Historical / alternate codes
        "LI":"Liechtenstein","MK":"North Macedonia",
    }

    BASE      = "https://eurovisionapi.runasp.net/api/senior/contests"
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "esc_api_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def fetch_year(year):
        path = os.path.join(cache_dir, f"{year}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        try:
            with urllib.request.urlopen(f"{BASE}/{year}", timeout=15) as r:
                data = json.loads(r.read().decode())
            with open(path, "w") as f:
                json.dump(data, f)
            time.sleep(0.15)
            return data
        except Exception as e:
            print(f"    Warning: could not fetch {year}: {e}")
            return None

    votes_rows   = []
    results_rows = []

    years = list(range(1975, 2026))
    years = [y for y in years if y != 2020]

    print(f"  Fetching {len(years)} contests from API …")
    for year in years:
        data = fetch_year(year)
        if data is None:
            continue

        id_to_cc = {c["id"]: c["country"] for c in data.get("contestants", [])}

        final = next(
            (r for r in data.get("rounds", []) if r.get("name") == "final"), None
        )
        if final is None:
            continue

        for perf in final.get("performances", []):
            cid   = perf["contestantId"]
            to_cc = id_to_cc.get(cid)
            place = perf.get("place")
            if to_cc is None:
                continue

            if place is not None:
                results_rows.append({
                    "year":    year,
                    "country": CC.get(to_cc, to_cc),
                    "rank":    place,
                })

            # Total score votes
            total = next(
                (s for s in perf.get("scores", []) if s["name"] == "total"), None
            )
            if total is None:
                continue

            for from_cc, pts in total.get("votes", {}).items():
                if pts > 0 and from_cc != to_cc:
                    votes_rows.append({
                        "year":         year,
                        "from_country": CC.get(from_cc, from_cc),
                        "to_country":   CC.get(to_cc, to_cc),
                        "points":       pts,
                    })

    print(f"  {len(votes_rows):,} vote rows, {len(results_rows):,} result rows")
    print(f"  Years: {min(r['year'] for r in votes_rows)}–{max(r['year'] for r in votes_rows)}")

    va          = pd.DataFrame(votes_rows)
    results_raw = pd.DataFrame(results_rows)
    rank_map    = {(int(r.year), r.country): int(r["rank"])
                   for _, r in results_raw.iterrows()}

    va["recipient_rank"] = va.apply(
        lambda r: rank_map.get((int(r.year), r.to_country), np.nan), axis=1)
    va = va.dropna(subset=["recipient_rank"])
    va["recipient_rank"] = va["recipient_rank"].astype(int)

    dh = va[(va["points"] >= 8) &
            (va["recipient_rank"] > 5) &
            (va["from_country"] != va["to_country"])].copy()

    ny = va.groupby("from_country")["year"].nunique().rename("n_years")
    st = dh.groupby("from_country").agg(dh_votes=("points", "count")).join(ny)
    st["dh_rate"] = st["dh_votes"] / st["n_years"]
    st = st[st["n_years"] >= 5].sort_values("dh_rate", ascending=False)

    dhc = st[(st["dh_rate"] >= st["dh_rate"].quantile(0.75)) &
             (st["dh_votes"] >= 5)].index.tolist()

    piv = (dh[dh["from_country"].isin(dhc)]
           .assign(song_id=lambda d: d["year"].astype(str) + " · " + d["to_country"])
           .groupby(["song_id", "from_country"])["points"].sum()
           .unstack(fill_value=0))
    bin_ = (piv >= 8).astype(int)
    bin_ = bin_[bin_.sum(axis=1) >= 2]

    top_c = st[st.index.isin(bin_.columns)].head(n_countries).index.tolist()
    fca   = bin_[top_c]
    fca   = fca[fca.sum(axis=1) >= min_championed_by]
    return fca, dh, st, rank_map

def build_lattice_graph(fca, min_extent=3):
    from concepts import Context
    obj=list(fca.index); prop=list(fca.columns)
    bools=[tuple(bool(v) for v in row) for row in fca.values]
    lat=Context(obj,prop,bools).lattice

    Gf=nx.DiGraph()
    for i,c in enumerate(lat):
        Gf.add_node(i,n_extent=len(list(c.extent)),n_intent=len(list(c.intent)),
                    extent=list(c.extent),intent=list(c.intent))
    for i,c in enumerate(lat):
        for u in c.upper_neighbors: Gf.add_edge(i,u.index)

    me=max(Gf.nodes[n]["n_extent"] for n in Gf)
    keep={n for n in Gf if Gf.nodes[n]["n_extent"]>=min_extent
          or Gf.nodes[n]["n_extent"]==me or Gf.nodes[n]["n_extent"]==0}

    def nka(node):
        vis,q,res=set(),[node],[]
        while q:
            cur=q.pop(0)
            for nx_ in Gf.successors(cur):
                if nx_ in vis: continue
                vis.add(nx_)
                if nx_ in keep: res.append(nx_)
                else: q.append(nx_)
        return res

    G=nx.DiGraph()
    for n in keep: G.add_node(n,**Gf.nodes[n])
    for n in keep:
        for a in nka(n): G.add_edge(n,a)
    G=nx.transitive_reduction(G)
    for n in G.nodes(): G.nodes[n].update(Gf.nodes[n])
    return G, prop


def assign_layer_key(n_intent, lon_centroid, max_intent):
    if n_intent == 0:           return "top"
    if n_intent == max_intent:  return "bottom"
    if n_intent == 1:           return 1
    if n_intent == 2:
        return "2w" if lon_centroid < GEO_SPLIT else "2e"
    if n_intent in (3, 4): return n_intent
    return n_intent


def compute_pos(G, properties):
    max_intent = len(properties)
    AUS_LON = 57.0   # visual placeholder longitude for Australia

    # First pass: assign layer key and raw lon
    for n,d in G.nodes(data=True):
        intent = d["intent"]
        n_int  = d["n_intent"]
        # Longitude centroid (replace Australia with placeholder)
        lons = []
        for c in intent:
            if c == "Australia": lons.append(AUS_LON)
            elif c in GEO:       lons.append(GEO[c][0])
        lon = np.mean(lons) if lons else (LON0+LON1)/2
        key = assign_layer_key(n_int, lon, max_intent)
        G.nodes[n]["_layer_key"] = key
        G.nodes[n]["_raw_lon"]   = lon

    # Assign y from LAYER_LAT
    for n,d in G.nodes(data=True):
        d["_y"] = LAYER_LAT[d["_layer_key"]]

    # Spread nodes within each sub-layer; preserve geographic order
    sub_layers = defaultdict(list)
    for n,d in G.nodes(data=True): sub_layers[d["_layer_key"]].append(n)

    min_gap_deg = 3.0   # degrees longitude minimum gap

    for key, nodes in sub_layers.items():
        if len(nodes) < 2:
            G.nodes[nodes[0]]["_x"] = G.nodes[nodes[0]]["_raw_lon"]
            continue

        sorted_nodes = sorted(nodes, key=lambda n: G.nodes[n]["_raw_lon"])
        xs = [G.nodes[n]["_raw_lon"] for n in sorted_nodes]

        # Push apart any that are too close
        for i in range(1, len(sorted_nodes)):
            if xs[i] - xs[i-1] < min_gap_deg:
                xs[i] = xs[i-1] + min_gap_deg

        # If overflow right, slide left
        if xs[-1] > LON1 - 0.5:
            shift = xs[-1] - (LON1 - 0.5)
            xs = [x - shift for x in xs]

        # Re-centre around the geographic mean of the layer
        geo_mean = np.mean([G.nodes[n]["_raw_lon"] for n in sorted_nodes])
        cur_mean = np.mean(xs)
        xs = [x + (geo_mean - cur_mean) for x in xs]

        # Final clamp
        if xs[0] < LON0 + 0.5:
            shift = LON0 + 0.5 - xs[0]
            xs = [x + shift for x in xs]
        if xs[-1] > LON1 - 0.5:
            shift = xs[-1] - (LON1 - 0.5)
            xs = [x - shift for x in xs]

        for n, x in zip(sorted_nodes, xs):
            G.nodes[n]["_x"] = x

    return {n: np.array([G.nodes[n]["_x"], G.nodes[n]["_y"]]) for n in G.nodes()}


def mix_colors(intent):
    cols = [COUNTRY_COLORS.get(c, ACCENT3) for c in intent]
    rgb  = np.mean([[int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)]
                    for c in cols], axis=0).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def main():
    print("[1] Building context …")
    fca, dark_horse, dh_stats, rank_map = build_context(8, 3)

    print("[2] Building lattice …")
    G, properties = build_lattice_graph(fca, 3)
    max_intent = len(properties)
    max_extent = max(G.nodes[n]["n_extent"] for n in G.nodes())
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"    Countries: {properties}")

    print("[3] Layout …")
    pos = compute_pos(G, properties)

    print("[4] Map data …")
    geojson_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ne_countries.geojson")
    if not os.path.exists(geojson_path):
        fallback = "/tmp/ne_countries.geojson"
        if os.path.exists(fallback):
            geojson_path = fallback
        else:
            print("    Downloading Natural Earth GeoJSON …")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                "master/geojson/ne_50m_admin_0_countries.geojson",
                fallback
            )
            geojson_path = fallback
    world  = gpd.read_file(geojson_path)
    clip   = sbox(LON0-1, LAT0-2, LON1+1, LAT1+2)
    region = world.clip(clip)

    print("[5] Rendering …")
    fig, ax = plt.subplots(figsize=(22, 22), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(LON0-2, LON1+3)
    ax.set_ylim(LAT0-5, LAT1+9)
    ax.axis("off")

    # ── Map ───────────────────────────────────────────────────────────────────
    region.plot(ax=ax, facecolor="#0c0c22", edgecolor="#1c1c3c",
                linewidth=0.3, zorder=1)

    name_fixes = {"Czech Republic":"Czechia"}
    for c in properties:
        if c == "Australia": continue
        row = world[world["NAME"]==name_fixes.get(c,c)]
        if row.empty: continue
        if c == "Russia":
            row = row.clip(sbox(LON0-1, LAT0-2, LON1+1, LAT1+2))
        color = COUNTRY_COLORS.get(c, WHITE)
        row.plot(ax=ax, facecolor=color+"18", edgecolor=color+"55",
                 linewidth=0.8, zorder=2)

    # ── Bloc shading bands ────────────────────────────────────────────────────
    # Post-Soviet: roughly 28°E–57°E, lat 38–58
    ax.fill_between([28, 57], [38,38], [58,58],
                    color="#50c8b0", alpha=0.03, zorder=0)
    # Western Europe: roughly -2°E–20°E, lat 43–58
    ax.fill_between([-2, 20], [43,43], [58,58],
                    color="#e8c84a", alpha=0.03, zorder=0)

    # Bloc labels (faint, in map space)
    ax.text(44, 60.5, "POST-SOVIET BLOC", ha="center", fontsize=7.5,
            color="#50c8b0", alpha=0.35, fontfamily="monospace",
            fontstyle="italic", fontweight="bold", zorder=3)
    ax.text(9, 60.5, "WESTERN EUROPE", ha="center", fontsize=7.5,
            color="#e8c84a", alpha=0.35, fontfamily="monospace",
            fontstyle="italic", fontweight="bold", zorder=3)

    # ── Layer separator lines ─────────────────────────────────────────────────
    for key, lat in LAYER_LAT.items():
        if key in ("top","bottom"): continue
        ax.axhline(lat-1.8, color=MID, lw=0.25, alpha=0.5,
                   zorder=3, linestyle="--", dashes=(6,4))

    # ── Conflict arc ──────────────────────────────────────────────────────────
    arm_p = aze_p = arm_aze_concept = None
    for n,d in G.nodes(data=True):
        if d["intent"]==["Armenia"]:   arm_p = pos[n]
        if d["intent"]==["Azerbaijan"]: aze_p = pos[n]
        if set(d["intent"])=={"Armenia","Azerbaijan"}: arm_aze_concept = (n,d)

    if arm_p is not None and aze_p is not None:
        arc = FancyArrowPatch(
            (arm_p[0], arm_p[1]), (aze_p[0], aze_p[1]),
            arrowstyle="-", connectionstyle="arc3,rad=0.4",
            color="#cc1818", lw=2.0, alpha=0.5, zorder=4,
            linestyle=(0,(4,3))
        )
        ax.add_patch(arc)
        # Label midpoint
        mx = (arm_p[0]+aze_p[0])/2 - 1
        my = (arm_p[1]+aze_p[1])/2 + 4.5
        ax.text(mx, my, "⚔  armed conflict  2004–2020",
                ha="center", fontsize=6.5, color="#e03030",
                fontfamily="monospace", fontstyle="italic", zorder=12,
                path_effects=[pe.withStroke(linewidth=2.0, foreground=BG)])

    # ── True-position dotted connectors for country nodes ─────────────────────
    for n,d in G.nodes(data=True):
        if len(d["intent"]) != 1: continue
        c = d["intent"][0]
        if c == "Australia": continue
        geo = GEO.get(c)
        if geo is None: continue
        px, py = pos[n]
        gx, gy = geo
        color = COUNTRY_COLORS.get(c, GREY)
        if abs(gx-px) > 0.3 or abs(gy-py) > 0.3:
            ax.plot([gx,px],[gy,py], color=color+"40",
                    lw=0.6, linestyle="dotted", zorder=3)
            ax.scatter(gx, gy, s=18, c=color, alpha=0.45,
                       linewidths=0, zorder=4, marker="^")

    # ── Lattice edges ─────────────────────────────────────────────────────────
    for u,v in G.edges():
        p0, p1 = pos[u], pos[v]
        w = ((G.nodes[u]["n_extent"]+G.nodes[v]["n_extent"])/(2*max_extent))**0.45
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]],
                color=MID, lw=0.4+1.1*w, alpha=0.18+0.55*w,
                zorder=5, solid_capstyle="round")

    # ── Nodes ─────────────────────────────────────────────────────────────────
    for n,d in G.nodes(data=True):
        x, y   = pos[n]
        intent = d["intent"]
        extent = d["n_extent"]
        n_int  = d["n_intent"]
        size   = 45 + 380*(extent/max_extent)**0.5

        if n_int == 0:              color = ACCENT2
        elif n_int == max_intent:   color = "#303050"
        elif len(intent)==1:        color = COUNTRY_COLORS.get(intent[0], ACCENT3)
        else:                       color = mix_colors(intent)

        # Glow for high-extent nodes
        if extent > max_extent * 0.3:
            ax.scatter(x,y, s=size*2.8, c=color, alpha=0.12, zorder=6, linewidths=0)

        ax.scatter(x, y, s=size, c=color, zorder=7,
                   linewidths=0.7, edgecolors=WHITE+"44", alpha=0.94)

        # Label — above the dot
        if n_int == 0:
            lbl, fs, fw, dy = "ALL SONGS", 7.0, "bold", 1.4
        elif n_int == max_intent:
            lbl, fs, fw, dy = "∅  no consensus", 6.0, "normal", 1.4
        elif len(intent) == 1:
            lbl, fs, fw, dy = intent[0], 6.5, "bold", 1.6
        elif extent >= max_extent * 0.10:
            lbl = " + ".join(c.split()[0] for c in intent)
            fs, fw, dy = 4.8, "normal", 1.3
        else:
            lbl = ""

        if lbl:
            ax.text(x, y+dy, lbl, ha="center", va="bottom",
                    fontsize=fs, color=WHITE, fontweight=fw,
                    fontfamily="monospace", zorder=9,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=BG+"cc")])

        # Song-count badge — below the dot for multi-country concepts
        if len(intent) >= 2 and extent >= 3:
            ax.text(x, y-1.5, str(extent)+" songs", ha="center", va="top",
                    fontsize=4.2, color=ACCENT1, fontweight="bold",
                    fontfamily="monospace", zorder=9, alpha=0.85,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

    # ── ARM+AZE callout ───────────────────────────────────────────────────────
    if arm_aze_concept is not None:
        n, d = arm_aze_concept
        p = pos[n]
        ax.annotate(
            f"  {d['n_extent']} shared underdog picks\n  despite 20 years of conflict",
            xy=(p[0], p[1]),
            xytext=(p[0]-12, p[1]-2.5),
            fontsize=7.5, color=ACCENT1, fontfamily="monospace",
            fontstyle="italic", zorder=12,
            arrowprops=dict(arrowstyle="->", color=ACCENT1+"99",
                            lw=1.0, connectionstyle="arc3,rad=0.25"),
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    # ── Australia inset box ───────────────────────────────────────────────────
    aus_node = next((n for n,d in G.nodes(data=True)
                     if d["intent"]==["Australia"]), None)
    if aus_node is not None:
        ax.text(LON1+1.5, LAYER_LAT[1], "AUS ↗",
                ha="left", va="center", fontsize=6.5,
                color=COUNTRY_COLORS["Australia"],
                fontfamily="monospace", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
        ax.text(LON1+1.5, LAYER_LAT[1]-1.2, "(off-map)",
                ha="left", va="center", fontsize=5.0, color=GREY,
                fontfamily="monospace")

    # ── Layer labels — right margin ───────────────────────────────────────────
    lbl_map = {
        "top":    "ALL SONGS",
        1:        "1 country",
        "2w":     "2 countries\n(western)",
        "2e":     "2 countries\n(eastern)",
        3:        "3 countries",
        4:        "4 countries",
        5:        "5 countries",
        6:        "6 countries",
        7:        "7 countries",
        8:        "8 countries",
        9:        "9 countries",
        10:       "10 countries",
        "bottom": "NO CONSENSUS",
    }
    sub_layers = defaultdict(list)
    for n,d in G.nodes(data=True): sub_layers[d["_layer_key"]].append(n)
    for key, nodes in sub_layers.items():
        lat = np.mean([pos[n][1] for n in nodes])
        ax.text(LON1+1.0, lat, lbl_map.get(key, str(key)),
                va="center", ha="left", fontsize=5.5,
                color=GREY, fontfamily="monospace", linespacing=1.3)

    # ── Legends ───────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=COUNTRY_COLORS.get(c, GREY), label=c)
               for c in properties]
    leg1 = ax.legend(handles=patches, loc="upper left",
                     facecolor=SURFACE, edgecolor=MID,
                     labelcolor=WHITE, fontsize=6.5,
                     title="Dark-horse voters", title_fontsize=6.5,
                     handlelength=0.9, borderpad=0.7, labelspacing=0.35)
    leg1.get_title().set_color(ACCENT1)
    ax.add_artist(leg1)

    for frac, lbl in [(0.08,"3 songs"),(0.4,"10 songs"),(1.0,"all songs")]:
        s = 45 + 380*frac**0.5
        ax.scatter([],[],s=s,c=GREY,alpha=0.7,label=lbl,edgecolors=WHITE+"44")
    leg2 = ax.legend(loc="lower left", facecolor=SURFACE, edgecolor=MID,
                     labelcolor=WHITE, fontsize=6.5, title="Node size",
                     title_fontsize=6.5, scatterpoints=1,
                     borderpad=0.7, labelspacing=0.35)
    leg2.get_title().set_color(ACCENT1)

    # ── Axis label ────────────────────────────────────────────────────────────
    ax.annotate("", xy=(LON0-1.5, LAT1+7.5), xytext=(LON0-1.5, LAT0-4.0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.text(LON0-1.8, (LAT0+LAT1)/2+3, "← more specific",
            va="center", ha="right", fontsize=6, color=GREY,
            fontfamily="monospace", rotation=90)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        "DARK HORSE SUPPORTER NETWORK\n"
        "Eurovision Song Contest  ·  Geo-Political Concept Lattice  ·  2004–2025",
        fontsize=15, fontweight="bold", color=ACCENT1,
        fontfamily="monospace", pad=16)

    ax.text(0.5, -0.015,
            "x = geographic longitude of concept members  ·  "
            "y = lattice layer (specificity)  ·  "
            "▲ = true capital position  ·  "
            "Data: TidyTuesday + eurovision.tv  ·  Map: Natural Earth",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6, color=GREY, fontfamily="monospace")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esc_output")
    os.makedirs(out_dir, exist_ok=True)
    for ext,dpi in [("pdf",200),("png",150)]:
        path = os.path.join(out_dir, f"geo_lattice.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"✓  {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()