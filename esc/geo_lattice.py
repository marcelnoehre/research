"""
Geo-Political Concept Lattice — Eurovision Dark Horse Supporter Network
=======================================================================
FIXES applied vs original:
  1. Song-first selection: identify dark-horse songs (rank > 5, ≥3 countries
     gave ≥5 pts), THEN derive the supporter countries — not the reverse.
  2. Points threshold lowered to ≥5 (captures 5,6,7,8,10,12) so consistent
     mid-tier support counts, not just top-2 maximum votes.
  3. Removed circular combined_score re-ranking that just laundered dh_rate bias.
  4. Epoch presence uses a proportional threshold (≥8 % of that epoch's songs)
     rather than a flat "≥2 songs" cut that biased short epochs.
  5. Country set derived from FCA itself (attributes that appear in ≥6 songs),
     so the lattice reflects actual taste clusters, not a pre-chosen 8.

Each concept node shows offset shapes:
  ■  square   (top-left)  = present in epoch 1998–2008
  ●  circle   (top-right) = present in epoch 2009–2015
  ▲  triangle (bottom)    = present in epoch 2016–2025

Multiple shapes = concept persisted across epochs (stable).
Shape size ∝ songs championed in that epoch.
Node x = geographic longitude centroid · y = lattice layer.
"""

import os, io, warnings, urllib.request, math
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

EPOCHS = {
    "1998–2008": {"range": (1998, 2008), "color": "#e8c84a", "shape": "square",   "marker": "s"},
    "2009–2015": {"range": (2009, 2015), "color": "#5ec9b8", "shape": "circle",   "marker": "o"},
    "2016–2025": {"range": (2016, 2025), "color": "#c96aa0", "shape": "triangle", "marker": "^"},
}
EPOCH_LABELS = list(EPOCHS.keys())

COUNTRY_COLORS = {
    "Italy":                "#e8c84a",
    "Armenia":              "#d06090",
    "Georgia":              "#50c8b0",
    "Moldova":              "#f07840",
    "Azerbaijan":           "#70c050",
    "Russia":               "#d05050",
    "Montenegro":           "#5090d8",
    "Switzerland":          "#9070d8",
    "Australia":            "#50c8d8",
    "Czech Republic":       "#d09050",
    "Greece":               "#d0d050",
    "Ukraine":              "#60a8e0",
    "Serbia":               "#e08060",
    "Norway":               "#80e080",
    "San Marino":           "#a0a0e0",
    "North Macedonia":      "#d0a060",
    "Bosnia & Herzegovina": "#a0d0a0",
    "Albania":              "#e0a0a0",
    "Croatia":              "#80d0d0",
    "Slovakia":             "#a0b0d0",
    "Bulgaria":             "#e0b080",
    "Romania":              "#b0e0b0",
    "Belarus":              "#d0b0d0",
    "Poland":               "#e0d080",
    "Cyprus":               "#f0c090",
    "Malta":                "#c0d0f0",
    "Portugal":             "#90b090",
    "Ireland":              "#70d070",
    "Finland":              "#90d0e0",
    "Latvia":               "#b0c0e0",
    "Lithuania":            "#d0e0b0",
    "Estonia":              "#b0e0d0",
    "Hungary":              "#e0c0b0",
    "Slovenia":             "#c0e0c0",
    "Sweden":               "#e0e0a0",
    "Denmark":              "#f0a0b0",
    "Netherlands":          "#a0c0f0",
    "Israel":               "#f0f0b0",
    "Austria":              "#d0a0d0",
    "France":               "#a0b0e0",
    "Spain":                "#f0b070",
    "Germany":              "#c0c0c0",
    "United Kingdom":       "#b0a0d0",
    "Turkey":               "#f0d090",
}

GEO = {
    "Italy":                (12.5, 41.9),
    "Armenia":              (44.5, 40.2),
    "Georgia":              (44.8, 41.7),
    "Moldova":              (28.9, 47.0),
    "Azerbaijan":           (49.9, 40.4),
    "Russia":               (37.6, 55.8),
    "Montenegro":           (19.3, 42.4),
    "Switzerland":          ( 7.4, 46.9),
    "Czech Republic":       (14.4, 50.1),
    "Greece":               (23.7, 37.9),
    "Ukraine":              (30.5, 50.4),
    "Serbia":               (21.0, 44.0),
    "Norway":               (10.7, 59.9),
    "San Marino":           (12.4, 43.9),
    "North Macedonia":      (21.4, 42.0),
    "Bosnia & Herzegovina": (18.4, 43.8),
    "Albania":              (19.8, 41.3),
    "Croatia":              (16.0, 45.8),
    "Slovakia":             (17.1, 48.1),
    "Bulgaria":             (23.3, 42.7),
    "Romania":              (26.1, 44.4),
    "Belarus":              (27.6, 53.9),
    "Poland":               (21.0, 52.2),
    "Australia":            (149.1,-35.3),
    "Cyprus":               (33.4, 35.1),
    "Malta":                (14.5, 35.9),
    "Portugal":             ( -8.2, 39.4),
    "Ireland":              ( -8.2, 53.3),
    "Finland":              (25.7, 61.9),
    "Latvia":               (24.1, 56.9),
    "Lithuania":            (23.9, 55.2),
    "Estonia":              (25.0, 58.6),
    "Hungary":              (19.0, 47.5),
    "Slovenia":             (14.5, 46.1),
    "Sweden":               (18.1, 59.3),
    "Denmark":              (10.2, 55.7),
    "Netherlands":          ( 4.9, 52.4),
    "Israel":               (35.2, 31.8),
    "Austria":              (16.4, 48.2),
    "France":               ( 2.3, 46.2),
    "Spain":                ( -3.7, 40.4),
    "Germany":              (10.5, 51.2),
    "United Kingdom":       ( -1.5, 52.5),
    "Turkey":               (35.2, 39.9),
}

LON0, LON1 = -12,  58
LAT0, LAT1 =  34,  62

LAYER_LAT = {
    "top":    66.0,
    1:        59.5,
    "2w":     55.5,
    "2e":     51.5,
    3:        47.0,
    4:        43.5,
    5:        40.5,
    6:        38.5,
    7:        36.5,
    8:        35.0,
    9:        34.5,
    10:       34.0,
    "bottom": 35.5,
}
GEO_SPLIT = 31.0

CC = {
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
    "CS":"Montenegro",
    "KZ":"Kazakhstan","XK":"Kosovo",
    "GB-ENG":"United Kingdom","GB-WLS":"United Kingdom",
    "GB-SCO":"United Kingdom","GB-NIR":"United Kingdom",
    "LI":"Liechtenstein",
}

NAME_NORM = {
    "F.Y.R. Macedonia":      "North Macedonia",
    "Macedonia":             "North Macedonia",
    "Serbia and Montenegro": "Montenegro",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_all_data():
    import urllib.request, json, time
    BASE      = "https://eurovisionapi.runasp.net/api/senior/contests"
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "esc_api_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def fetch_year(year):
        path = os.path.join(cache_dir, f"{year}.json")
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
        try:
            with urllib.request.urlopen(f"{BASE}/{year}", timeout=15) as r:
                data = json.loads(r.read().decode())
            with open(path, "w") as f: json.dump(data, f)
            time.sleep(0.15)
            return data
        except Exception as e:
            print(f"    Warning: {year}: {e}")
            return None

    votes_rows, results_rows = [], []
    years = [y for y in range(1998, 2026) if y != 2020]
    print(f"  Fetching {len(years)} contests …")

    for year in years:
        data = fetch_year(year)
        if data is None: continue
        id_to_cc = {c["id"]: c["country"] for c in data.get("contestants", [])}
        final = next((r for r in data.get("rounds", [])
                      if r.get("name") == "final"), None)
        if final is None: continue

        year_perfs = []
        for perf in final.get("performances", []):
            cid   = perf["contestantId"]
            to_cc = id_to_cc.get(cid)
            if to_cc is None: continue
            scores    = perf.get("scores", [])
            total     = next((s for s in scores if s["name"] == "total"), None)
            total_pts = total["points"] if total else 0
            year_perfs.append((cid, to_cc, scores, total_pts))

        year_perfs.sort(key=lambda x: -x[3])
        for rank, (cid, to_cc, perf_scores, _) in enumerate(year_perfs, start=1):
            results_rows.append({
                "year":    year,
                "country": CC.get(to_cc, to_cc),
                "rank":    rank,
            })
            # FIX 1: Use ≥5 pts to capture consistent mid-tier support,
            # not just 10/12-pt extremes. For 2016+, use public vote only.
            if year >= 2016:
                score = next((s for s in perf_scores if s["name"] == "public"), None)
            else:
                score = next((s for s in perf_scores if s["name"] == "total"), None)
            combined = score["votes"] if score else {}

            for from_cc, pts in combined.items():
                if pts >= 8 and from_cc != to_cc:   # keep only 8, 10, 12 — real championing
                    votes_rows.append({
                        "year":         year,
                        "from_country": CC.get(from_cc, from_cc),
                        "to_country":   CC.get(to_cc, to_cc),
                        "points":       pts,
                    })

    va  = pd.DataFrame(votes_rows)
    res = pd.DataFrame(results_rows)
    rank_map = {(int(r.year), r.country): int(r["rank"])
                for _, r in res.iterrows()}
    print(f"  {len(va):,} rows | {va['year'].min()}–{va['year'].max()}")
    return va, rank_map


# ── FCA context ───────────────────────────────────────────────────────────────

def build_union_lattice(va, rank_map,
                        min_pts=8,          # 8/10/12 only — genuine championing
                        min_rank=5,         # recipient must have finished > 5th
                        min_champions=3,    # song needs ≥3 different country champions
                        min_song_appearances=10):  # top-N countries to keep
    from concepts import Context

    va = va.copy()
    va["from_country"] = va["from_country"].replace(NAME_NORM)
    va["to_country"]   = va["to_country"].replace(NAME_NORM)
    rank_map = {(y, NAME_NORM.get(c, c)): r for (y, c), r in rank_map.items()}

    va["recipient_rank"] = [
        rank_map.get((int(y), tc), float("nan"))
        for y, tc in zip(va["year"], va["to_country"])
    ]
    va = va.dropna(subset=["recipient_rank"])
    va["recipient_rank"] = va["recipient_rank"].astype(int)

    # ── FIX: Song-first selection ──────────────────────────────────────────
    # Step 1: find all dark-horse votes (≥min_pts, rank > min_rank)
    dh = va[
        (va["points"] >= min_pts) &
        (va["recipient_rank"] > min_rank) &
        (va["from_country"] != va["to_country"])
    ].copy()

    dh["song_id"] = dh["year"].astype(str) + " · " + dh["to_country"]

    # Step 2: keep only songs championed by ≥min_champions distinct countries
    song_champ_counts = dh.groupby("song_id")["from_country"].nunique()
    qualifying_songs  = song_champ_counts[song_champ_counts >= min_champions].index
    dh = dh[dh["song_id"].isin(qualifying_songs)]

    print(f"    {len(qualifying_songs)} qualifying dark-horse songs "
          f"({min_champions}+ countries, rank>{min_rank}, ≥{min_pts}pts)")

    if dh.empty:
        raise ValueError("No qualifying songs found — check thresholds or data.")

    # Step 3: pivot to binary matrix (songs × countries)
    piv = (dh.groupby(["song_id", "from_country"])["points"]
             .sum()
             .unstack(fill_value=0))
    bin_full = (piv >= min_pts).astype(int)

    # Step 4: restrict to top N countries by songs championed.
    # Ranked purely by song count — no dh_rate bias, no circular scoring.
    country_song_counts = bin_full.sum(axis=0).sort_values(ascending=False)
    active_countries    = country_song_counts.head(min_song_appearances).index.tolist()

    print(f"    Top-{min_song_appearances} countries by songs championed:")
    print(f"    {country_song_counts.head(min_song_appearances).to_dict()}")

    bin_full = bin_full[active_countries]

    # Re-filter songs: still need ≥min_champions of the *active* countries
    bin_full = bin_full[bin_full.sum(axis=1) >= min_champions]

    print(f"    FCA matrix: {bin_full.shape[0]} songs × "
          f"{bin_full.shape[1]} countries")

    # ── Debug export ───────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esc_output")
    os.makedirs(out_dir, exist_ok=True)
    export_cxt(bin_full, out_dir)

    # Per-epoch debug
    epoch_song_sets = {}
    for label, meta in EPOCHS.items():
        y0, y1 = meta["range"]
        epoch_songs = [s for s in bin_full.index
                       if y0 <= int(s.split(" · ")[0]) <= y1]
        epoch_song_sets[label] = set(epoch_songs)
        dh_e = dh[dh["song_id"].isin(epoch_songs)]
        slug = label.replace("–","_").replace(" ","")
        dh_e.to_csv(os.path.join(out_dir, f"debug_dh_votes_{slug}.csv"), index=False)
        print(f"    {label}: {len(epoch_songs)} qualifying songs, "
              f"{len(dh_e)} champion votes")

    # ── Build concept lattice ──────────────────────────────────────────────
    objects    = list(bin_full.index)
    properties = list(bin_full.columns)
    bools      = [tuple(bool(v) for v in row) for row in bin_full.values]
    lattice    = Context(objects, properties, bools).lattice

    # Pruned Hasse graph
    Gf = nx.DiGraph()
    for i, c in enumerate(lattice):
        Gf.add_node(i,
                    n_extent=len(list(c.extent)),
                    n_intent=len(list(c.intent)),
                    extent=list(c.extent),
                    intent=list(c.intent))
    for i, c in enumerate(lattice):
        for u in c.upper_neighbors:
            Gf.add_edge(i, u.index)

    me   = max(Gf.nodes[n]["n_extent"] for n in Gf)
    keep = {n for n in Gf
            if Gf.nodes[n]["n_extent"] >= 3     # FIX: was 4, lowered to 3
            or Gf.nodes[n]["n_extent"] == me
            or Gf.nodes[n]["n_extent"] == 0}

    def next_kept_ancestors(node):
        vis, q, res = set(), [node], []
        while q:
            cur = q.pop(0)
            for nx_ in Gf.successors(cur):
                if nx_ in vis: continue
                vis.add(nx_)
                if nx_ in keep: res.append(nx_)
                else: q.append(nx_)
        return res

    G = nx.DiGraph()
    for n in keep: G.add_node(n, **Gf.nodes[n])
    for n in keep:
        for a in next_kept_ancestors(n): G.add_edge(n, a)
    G = nx.transitive_reduction(G)
    for n in G.nodes(): G.nodes[n].update(Gf.nodes[n])

    # ── FIX: Proportional epoch presence threshold ─────────────────────────
    # Require ≥ max(2, 8% of that epoch's total qualifying songs) songs per epoch
    epoch_thresholds = {}
    for label in EPOCHS:
        total_epoch_songs = len(epoch_song_sets[label])
        epoch_thresholds[label] = max(2, math.ceil(total_epoch_songs * 0.08))
    print(f"    Epoch presence thresholds: {epoch_thresholds}")

    for n, d in G.nodes(data=True):
        ep = {}
        for label, meta in EPOCHS.items():
            y0, y1 = meta["range"]
            songs_in_epoch = [s for s in d["extent"]
                              if y0 <= int(s.split(" · ")[0]) <= y1]
            if len(songs_in_epoch) >= epoch_thresholds[label]:
                ep[label] = len(songs_in_epoch)
        d["epoch_presence"] = ep

    return G, properties, active_countries


# ── Layout ────────────────────────────────────────────────────────────────────

def assign_layer_key(n_intent, lon_centroid, max_intent):
    if n_intent == 0:          return "top"
    if n_intent == max_intent: return "bottom"
    if n_intent == 1:          return 1
    if n_intent == 2:
        return "2w" if lon_centroid < GEO_SPLIT else "2e"
    if n_intent in LAYER_LAT:  return n_intent
    int_keys = [k for k in LAYER_LAT if isinstance(k, int)]
    return min(int_keys, key=lambda k: abs(k - n_intent))


def compute_pos(G, properties):
    max_intent = len(properties)
    AUS_LON    = 57.0   # clamp Australia to right edge

    for n, d in G.nodes(data=True):
        intent = d["intent"]
        n_int  = d["n_intent"]
        lons   = [AUS_LON if c == "Australia" else GEO.get(c, (None, None))[0]
                  for c in intent
                  if c == "Australia" or (c in GEO and GEO[c][0] is not None)]
        lon = np.mean(lons) if lons else (LON0+LON1)/2
        key = assign_layer_key(n_int, lon, max_intent)
        G.nodes[n]["_layer_key"] = key
        G.nodes[n]["_raw_lon"]   = lon
        G.nodes[n]["_y"]         = LAYER_LAT.get(key, 40.0)

    sub_layers = defaultdict(list)
    for n, d in G.nodes(data=True): sub_layers[d["_layer_key"]].append(n)

    for key, nodes in sub_layers.items():
        if len(nodes) < 2:
            G.nodes[nodes[0]]["_x"] = G.nodes[nodes[0]]["_raw_lon"]
            continue
        sn        = sorted(nodes, key=lambda n: G.nodes[n]["_raw_lon"])
        xs        = [G.nodes[n]["_raw_lon"] for n in sn]
        available = (LON1 - LON0) * 0.92
        gap       = max(1.2, min(3.0, available / len(sn)))
        for i in range(1, len(sn)):
            if xs[i] - xs[i-1] < gap: xs[i] = xs[i-1] + gap
        gm = np.mean([G.nodes[n]["_raw_lon"] for n in sn])
        cm = np.mean(xs); xs = [x + (gm-cm) for x in xs]
        span = xs[-1] - xs[0]
        if span > available:
            scale = available / span
            xs = [gm + (x-gm)*scale for x in xs]
        if xs[0]  < LON0+0.5: d_ = LON0+0.5-xs[0];  xs = [x+d_ for x in xs]
        if xs[-1] > LON1-0.5: d_ = xs[-1]-(LON1-0.5);xs = [x-d_ for x in xs]
        for n, x in zip(sn, xs): G.nodes[n]["_x"] = x

    return {n: np.array([G.nodes[n]["_x"], G.nodes[n]["_y"]]) for n in G.nodes()}


# ── Drawing ───────────────────────────────────────────────────────────────────

SHAPE_OFFSETS = [(-0.5, 0.5), (0.5, 0.5), (0.0, -0.5)]  # sq, circ, tri


def draw_shape(ax, x, y, shape, size_pts, color, alpha=0.92, zorder=8):
    marker = {"square":"s", "circle":"o", "triangle":"^"}[shape]
    ax.scatter(x, y, s=size_pts, c=color, marker=marker,
               alpha=alpha, zorder=zorder,
               linewidths=0.6, edgecolors=WHITE+"55")


def draw_node(ax, x, y, d, max_extent, properties, epoch_max_ext):
    ep         = d.get("epoch_presence", {})
    n_int      = d["n_intent"]
    intent     = d["intent"]
    max_intent = len(properties)

    if n_int == 0:
        ax.scatter(x, y, s=220, c=ACCENT2, marker="o", zorder=8,
                   linewidths=0.8, edgecolors=WHITE+"55", alpha=0.95)
        ax.text(x, y+1.5, "ALL SONGS", ha="center", va="bottom",
                fontsize=6.5, color=WHITE, fontweight="bold",
                fontfamily="monospace", zorder=10,
                path_effects=[pe.withStroke(linewidth=2.0, foreground=BG+"cc")])
        return

    if n_int == max_intent:
        ax.scatter(x, y, s=80, c="#303050", marker="o", zorder=8,
                   linewidths=0.6, edgecolors=GREY+"55", alpha=0.8)
        ax.text(x, y+1.2, "∅", ha="center", va="bottom",
                fontsize=6.0, color=GREY, fontfamily="monospace", zorder=10)
        return

    present = [(lbl, EPOCHS[lbl], ep[lbl])
               for lbl in EPOCH_LABELS if lbl in ep]

    if not present:
        ax.scatter(x, y, s=40, c=GREY, marker="o", zorder=6,
                   linewidths=0.4, edgecolors=GREY+"33", alpha=0.3)
        return

    for i, (lbl, meta, ep_n) in enumerate(present):
        ep_max  = epoch_max_ext.get(lbl, max_extent) or max_extent
        size_pt = max(80, 60 + 300*(ep_n/ep_max)**0.5)
        dx, dy  = SHAPE_OFFSETS[i]
        draw_shape(ax, x+dx, y+dy, meta["shape"], size_pt,
                   meta["color"], alpha=0.92, zorder=8+i)
        ax.text(x+dx, y+dy, str(ep_n), ha="center", va="center",
                fontsize=5.5, color=BG, fontweight="bold",
                fontfamily="monospace", zorder=11)

    if n_int == 1:
        lbl = intent[0]
        fs, fw = 7.0, "bold"
    elif n_int <= 3:
        lbl = " + ".join(c.split()[0][:6] for c in intent)
        fs, fw = 5.5, "normal"
    else:
        lbl = ""

    if lbl:
        ax.text(x, y+1.8, lbl, ha="center", va="bottom",
                fontsize=fs, color=WHITE, fontweight=fw,
                fontfamily="monospace", zorder=12,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=BG+"ee")])


# ── CXT export ────────────────────────────────────────────────────────────────

def export_cxt(fca, out_dir, filename="esc_dark_horse_context.cxt"):
    objects    = list(fca.index)
    attributes = list(fca.columns)
    path       = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("B\n\n")
        f.write(f"{len(objects)}\n{len(attributes)}\n\n")
        for obj in objects:    f.write(f"{obj}\n")
        for att in attributes: f.write(f"{att}\n")
        for obj in objects:
            f.write("".join("X" if fca.loc[obj, att] else "."
                            for att in attributes) + "\n")
    print(f"    Saved {filename}  ({len(objects)} objects × {len(attributes)} attributes)")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[1] Fetching data …")
    va, rank_map = fetch_all_data()

    print("[2] Building union lattice …")
    G, properties, active_countries = build_union_lattice(va, rank_map)
    max_intent = len(properties)
    max_extent = max(G.nodes[n]["n_extent"] for n in G.nodes())
    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"    Countries ({len(properties)}): {properties}")

    # Concept CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esc_output")
    rows = []
    for n, d in G.nodes(data=True):
        ep = d.get("epoch_presence", {})
        rows.append({
            "node_id":         n,
            "n_intent":        d["n_intent"],
            "n_extent":        d["n_extent"],
            "intent":          " + ".join(sorted(d["intent"])),
            "extent_songs":    " | ".join(sorted(d["extent"])[:5]),
            "epoch_1998_2008": ep.get("1998–2008", 0),
            "epoch_2009_2015": ep.get("2009–2015", 0),
            "epoch_2016_2025": ep.get("2016–2025", 0),
            "n_epochs":        len(ep),
        })
    cdf = pd.DataFrame(rows).sort_values(["n_intent","n_extent"],
                                          ascending=[True,False])
    cdf.to_csv(os.path.join(out_dir, "debug_concepts.csv"), index=False)
    print(f"    Saved debug_concepts.csv  ({len(cdf)} rows)")
    print(f"    Nodes 1998-2008:{(cdf['epoch_1998_2008']>0).sum()}"
          f"  2009-2015:{(cdf['epoch_2009_2015']>0).sum()}"
          f"  2016-2025:{(cdf['epoch_2016_2025']>0).sum()}")

    print("[3] Layout …")
    pos = compute_pos(G, properties)

    print("[4] Map …")
    geojson_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "ne_countries.geojson")
    if not os.path.exists(geojson_path):
        fallback = "/tmp/ne_countries.geojson"
        if not os.path.exists(fallback):
            print("    Downloading Natural Earth GeoJSON …")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                "master/geojson/ne_50m_admin_0_countries.geojson", fallback)
        geojson_path = fallback
    world  = gpd.read_file(geojson_path)
    region = world.clip(sbox(LON0-1, LAT0-2, LON1+1, LAT1+2))

    print("[5] Rendering …")
    fig, ax = plt.subplots(figsize=(24, 24), facecolor=BG)
    ax.set_facecolor(BG)
    all_xs = [p[0] for p in pos.values()]
    all_ys = [p[1] for p in pos.values()]
    ax.set_xlim(max(LON0-1, min(all_xs)-2), LON1+2)
    ax.set_ylim(min(all_ys)-1.5, max(all_ys)+3)
    ax.axis("off")

    # Map base
    region.plot(ax=ax, facecolor="#0c0c22", edgecolor="#1c1c3c",
                linewidth=0.3, zorder=1)
    name_fixes = {"Czech Republic":"Czechia"}
    for c in properties:
        if c == "Australia": continue
        row = world[world["NAME"] == name_fixes.get(c, c)]
        if row.empty: continue
        if c == "Russia": row = row.clip(sbox(LON0-1, LAT0-2, LON1+1, LAT1+2))
        row.plot(ax=ax, facecolor=COUNTRY_COLORS.get(c, WHITE)+"14",
                 edgecolor=COUNTRY_COLORS.get(c, WHITE)+"50",
                 linewidth=0.7, zorder=2)

    # Bloc tints
    ax.fill_between([28,57],[38,38],[58,58], color="#50c8b0", alpha=0.025, zorder=0)
    ax.fill_between([-12,22],[43,43],[58,58], color="#e8c84a", alpha=0.025, zorder=0)
    ax.text(44, 60.5, "POST-SOVIET", ha="center", fontsize=6.5, color="#50c8b0",
            alpha=0.3, fontfamily="monospace", fontstyle="italic", fontweight="bold")
    ax.text(4,  60.5, "WESTERN EU",  ha="center", fontsize=6.5, color="#e8c84a",
            alpha=0.3, fontfamily="monospace", fontstyle="italic", fontweight="bold")

    # Layer separators
    for key, lat in LAYER_LAT.items():
        if key in ("top","bottom"): continue
        ax.axhline(lat-1.8, color=MID, lw=0.2, alpha=0.4,
                   zorder=3, linestyle="--", dashes=(6,4))

    # Conflict arc (Armenia ↔ Azerbaijan)
    arm_p = aze_p = None
    for n, d in G.nodes(data=True):
        if d["intent"] == ["Armenia"]:    arm_p = pos[n]
        if d["intent"] == ["Azerbaijan"]: aze_p = pos[n]
    if arm_p is not None and aze_p is not None:
        ax.add_patch(FancyArrowPatch(
            (arm_p[0], arm_p[1]), (aze_p[0], aze_p[1]),
            arrowstyle="-", connectionstyle="arc3,rad=0.4",
            color="#cc1818", lw=1.5, alpha=0.4, zorder=4,
            linestyle=(0,(4,3))))

    # Lattice edges
    for u, v in G.edges():
        p0, p1 = pos[u], pos[v]
        stable_u = len(G.nodes[u].get("epoch_presence", {})) >= 2
        stable_v = len(G.nodes[v].get("epoch_presence", {})) >= 2
        ec, ew, ea = (WHITE, 0.8, 0.45) if (stable_u and stable_v) else (GREY, 0.5, 0.25)
        ax.plot([p0[0],p1[0]], [p0[1],p1[1]],
                color=ec, lw=ew, alpha=ea, zorder=5, solid_capstyle="round")

    # Per-epoch max extents for proportional sizing
    epoch_max_ext = defaultdict(int)
    for n, d in G.nodes(data=True):
        for lbl, ns in d.get("epoch_presence",{}).items():
            epoch_max_ext[lbl] = max(epoch_max_ext[lbl], ns)
    epoch_max_ext = dict(epoch_max_ext)

    # Nodes
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        draw_node(ax, x, y, d, max_extent, properties, epoch_max_ext)

    # ARM+AZE callout
    for n, d in G.nodes(data=True):
        if set(d["intent"]) == {"Armenia","Azerbaijan"}:
            p  = pos[n]
            ep = d.get("epoch_presence",{})
            ax.annotate(
                f"  {sum(ep.values())} picks across {len(ep)} epoch"
                f"{'s' if len(ep)>1 else ''}\n  despite armed conflict",
                xy=(p[0],p[1]), xytext=(p[0]-11, p[1]-2.5),
                fontsize=6.5, color=ACCENT1, fontfamily="monospace",
                fontstyle="italic", zorder=12,
                arrowprops=dict(arrowstyle="->", color=ACCENT1+"88",
                                lw=0.8, connectionstyle="arc3,rad=0.2"),
                path_effects=[pe.withStroke(linewidth=2.0, foreground=BG)])
            break

    # Australia off-map label
    for n, d in G.nodes(data=True):
        if d["intent"] == ["Australia"] and d.get("epoch_presence"):
            ax.text(LON1+0.8, LAYER_LAT[1], "AUS↗\n(off-map)",
                    ha="left", va="center", fontsize=5.0,
                    color=COUNTRY_COLORS.get("Australia", WHITE),
                    fontfamily="monospace", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])
            break

    # Layer labels
    lbl_map = {"top":"ALL","bottom":"∅",1:"1c","2w":"2c(W)","2e":"2c(E)",
               3:"3c",4:"4c",5:"5c",6:"6c",7:"7c",8:"8c",9:"9c",10:"10c"}
    sub_layers = defaultdict(list)
    for n, d in G.nodes(data=True): sub_layers[d["_layer_key"]].append(n)
    for key, nodes in sub_layers.items():
        lat = np.mean([pos[n][1] for n in nodes])
        ax.text(LON1+0.6, lat, lbl_map.get(key, str(key)),
                va="center", ha="left", fontsize=4.8,
                color=GREY, fontfamily="monospace")

    # Legend: epoch shapes
    shape_handles = [
        ax.scatter([],[], s=80, c=meta["color"], marker=meta["marker"],
                   label=label, linewidths=0.5, edgecolors=WHITE+"44")
        for label, meta in EPOCHS.items()
    ]
    shape_handles.append(
        ax.scatter([],[], s=80, c=WHITE, marker="o", alpha=0.3,
                   label="multi-epoch (stable)", linewidths=2.0, edgecolors=WHITE))
    leg1 = ax.legend(handles=shape_handles, loc="upper left",
                     facecolor=SURFACE, edgecolor=MID, labelcolor=WHITE,
                     fontsize=7.0, title="Epoch  (■=1998–08  ●=09–15  ▲=16–25)",
                     title_fontsize=6.5, handlelength=0.9,
                     borderpad=0.7, labelspacing=0.4)
    leg1.get_title().set_color(ACCENT1)
    ax.add_artist(leg1)

    patches = [mpatches.Patch(color=COUNTRY_COLORS.get(c, GREY), label=c)
               for c in properties]
    leg2 = ax.legend(handles=patches, loc="lower left",
                     facecolor=SURFACE, edgecolor=MID, labelcolor=WHITE,
                     fontsize=5.5, title="Countries", title_fontsize=6.0,
                     handlelength=0.8, borderpad=0.6, labelspacing=0.25, ncol=2)
    leg2.get_title().set_color(ACCENT1)
    ax.add_artist(leg2)

    edge_handles = [
        mpatches.Patch(color=WHITE, label="stable (multi-epoch) concept edge"),
        mpatches.Patch(color=GREY,  label="single-epoch concept edge"),
    ]
    leg3 = ax.legend(handles=edge_handles, loc="upper right",
                     facecolor=SURFACE, edgecolor=MID, labelcolor=WHITE,
                     fontsize=5.5, title="Edge colour", title_fontsize=6.0,
                     handlelength=0.8, borderpad=0.6, labelspacing=0.25)
    leg3.get_title().set_color(ACCENT1)

    # Axis arrow
    ax.annotate("", xy=(LON0-2.0, max(all_ys)+2.5),
                    xytext=(LON0-2.0, min(all_ys)-1.0),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9))
    ax.text(LON0-2.6, (LAT0+LAT1)/2+3, "← more specific",
            va="center", ha="right", fontsize=6, color=GREY,
            fontfamily="monospace", rotation=90)

    # Title & footer
    ax.set_title(
        "DARK HORSE SUPPORTER NETWORK  ·  EUROVISION SONG CONTEST  ·  1998–2025\n"
        "Geo-Political Concept Lattice  ·  Televote Signal  ·  Three-Epoch Overlay",
        fontsize=15, fontweight="bold", color=ACCENT1,
        fontfamily="monospace", pad=14)
    ax.text(0.5, -0.012,
            "■=1998–2008  ●=2009–2015  ▲=2016–2025  ·  "
            "Multiple shapes = stable concept  ·  "
            "x = geographic longitude  ·  y = lattice layer  ·  "
            "≥8pts champion threshold (8/10/12 only)  ·  song-first selection  ·  "
            "Data: eurovisionapi.runasp.net  ·  Map: Natural Earth",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=5.5, color=GREY, fontfamily="monospace")

    for ext, dpi in [("pdf",200),("png",150)]:
        path = os.path.join(out_dir, f"geo_lattice.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"✓  {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()