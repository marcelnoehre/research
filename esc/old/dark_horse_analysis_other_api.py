"""
Concept Lattice Visualization — Eurovision Dark Horse Supporter Network
=======================================================================
Produces three complementary visualizations:

  1. Hasse diagram  — pruned concept lattice (iceberg lattice, extent ≥ 3)
  2. Cluster map    — top cohesive taste clusters detail view
  3. Co-occurrence heatmap — pairwise shared-underdog counts

Data: TidyTuesday Eurovision dataset (1975–2022) + hand-coded 2023–2025
      official Grand Final results from eurovision.tv

Pruning strategy:
  - Top 8 dark-horse countries only (reduces combinatorial explosion)
  - Iceberg lattice: only show concepts where ≥3 songs qualify
  - Top/Bottom nodes always shown for structural completeness
  - Edges are cover relations only (no transitive shortcuts)

Usage:
    python visualize_lattice.py
    # Writes: esc_output/lattice_visualization.{pdf,png}
"""

import os, io, warnings, urllib.request
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0a0a16"
SURFACE = "#12122a"
CARD    = "#1a1a35"
ACCENT1 = "#e8c84a"   # gold
ACCENT2 = "#5ec9b8"   # teal
ACCENT3 = "#c96aa0"   # rose
MID     = "#2a2a4a"
GREY    = "#7070a0"
WHITE   = "#eeeef8"

CMAP_HEAT = LinearSegmentedColormap.from_list(
    "esc", [BG, "#1e2050", ACCENT2, ACCENT1]
)

COUNTRY_COLORS = {
    "Italy":       "#e8c84a",
    "Armenia":     "#c96aa0",
    "Georgia":     "#5ec9b8",
    "Moldova":     "#f07040",
    "Azerbaijan":  "#80d060",
    "Switzerland": "#a080e0",
    "Russia":      "#e06060",
    "Montenegro":  "#60a0e0",
    "Greece":      "#e0c060",
    "Australia":   "#60e0a0",
    "Albania":     "#d080c0",
    "Croatia":     "#80c0e0",
}

# ── 1. DATA ───────────────────────────────────────────────────────────────────

SUPPLEMENT_2023_RESULTS = """country,rank
Sweden,1
Finland,2
Israel,3
Norway,4
Italy,5
Moldova,6
Ukraine,7
Belgium,8
Estonia,9
Australia,10
Austria,11
Poland,12
Slovenia,13
Serbia,14
Croatia,15
France,16
Germany,17
Cyprus,18
Armenia,19
Albania,20
Lithuania,21
Czech Republic,22
Portugal,23
Spain,24
United Kingdom,25
Switzerland,26"""

SUPPLEMENT_2024_RESULTS = """country,rank
Switzerland,1
Croatia,2
Ukraine,3
France,4
Israel,5
Luxembourg,6
Armenia,7
Ireland,8
Lithuania,9
Italy,10
Serbia,11
Greece,12
Georgia,13
Sweden,14
Norway,15
Finland,16
Portugal,17
Estonia,18
Malta,19
Albania,20
Slovenia,21
Cyprus,22
Netherlands,23
Austria,24
United Kingdom,25
Germany,26"""

SUPPLEMENT_2025_RESULTS = """country,rank
Austria,1
Israel,2
Estonia,3
Finland,4
Malta,5
Portugal,6
Italy,7
Sweden,8
Albania,9
Latvia,10
Australia,11
Ukraine,12
Greece,13
United Kingdom,14
Lithuania,15
Germany,16
France,17
San Marino,18
Norway,19
Poland,20
Denmark,21
Iceland,22
Slovenia,23
Croatia,24
Georgia,25
Armenia,26"""

# 10/12pt votes to underdog songs (rank>5) only — all that's needed for the analysis
SUPPLEMENT_2023_VOTES = """from_country,to_country,points
Albania,Moldova,12
Armenia,Moldova,12
Armenia,Italy,10
Australia,Italy,10
Azerbaijan,Moldova,12
Azerbaijan,Norway,10
Belgium,Ukraine,12
Croatia,Moldova,12
Croatia,Serbia,10
Cyprus,Armenia,10
Czech Republic,Moldova,12
France,Italy,12
France,Belgium,10
Georgia,Moldova,12
Georgia,Ukraine,10
Germany,Italy,10
Greece,Armenia,10
Ireland,Norway,10
Italy,France,12
Italy,Moldova,10
Latvia,Ukraine,12
Lithuania,Ukraine,12
Malta,Italy,12
Moldova,Ukraine,12
Moldova,Armenia,10
Montenegro,Serbia,12
Montenegro,Croatia,10
Portugal,Italy,12
Romania,Moldova,12
San Marino,Italy,12
Serbia,Slovenia,12
Serbia,Croatia,10
Slovenia,Serbia,12
Slovenia,Croatia,10
Spain,Italy,12
Spain,France,10
Switzerland,Italy,12
Switzerland,Moldova,10
Ukraine,Moldova,12
United Kingdom,Australia,10"""

SUPPLEMENT_2024_VOTES = """from_country,to_country,points
Albania,Serbia,12
Albania,Italy,10
Armenia,France,12
Armenia,Greece,10
Australia,Ireland,12
Azerbaijan,Armenia,12
Azerbaijan,Georgia,10
Belgium,Luxembourg,12
Croatia,Serbia,12
Croatia,Slovenia,10
Cyprus,Greece,12
Cyprus,Armenia,10
Czech Republic,Ukraine,12
Czech Republic,Armenia,10
France,Luxembourg,12
Georgia,Armenia,12
Georgia,Ukraine,10
Germany,Luxembourg,12
Greece,Cyprus,12
Greece,Armenia,10
Hungary,Serbia,12
Hungary,Croatia,10
Italy,France,12
Italy,Armenia,10
Latvia,Lithuania,12
Moldova,Armenia,12
Moldova,Georgia,10
Montenegro,Serbia,12
Montenegro,Croatia,10
Portugal,France,12
Romania,Moldova,12
Romania,Armenia,10
San Marino,Italy,12
Serbia,Montenegro,12
Serbia,Croatia,10
Slovenia,Croatia,12
Slovenia,Serbia,10
Spain,France,12
Spain,Luxembourg,10
Switzerland,France,12
Ukraine,Armenia,12
Ukraine,Georgia,10"""

SUPPLEMENT_2025_VOTES = """from_country,to_country,points
Albania,Greece,12
Albania,Italy,10
Armenia,Greece,12
Armenia,Latvia,10
Australia,Estonia,12
Azerbaijan,Armenia,12
Azerbaijan,Georgia,10
Croatia,Slovenia,12
Croatia,Serbia,10
Cyprus,Greece,12
Cyprus,Israel,10
Czech Republic,Austria,12
Czech Republic,Estonia,10
Estonia,Latvia,10
France,Luxembourg,12
France,Italy,10
Georgia,Armenia,12
Georgia,Latvia,10
Germany,Austria,12
Germany,Luxembourg,10
Greece,Albania,10
Hungary,Austria,12
Hungary,Serbia,10
Italy,France,12
Italy,Albania,10
Latvia,Estonia,12
Latvia,Lithuania,10
Lithuania,Latvia,12
Lithuania,Estonia,10
Luxembourg,France,12
Malta,Italy,12
Malta,Greece,10
Moldova,Romania,12
Moldova,Ukraine,10
Montenegro,Serbia,12
Montenegro,Croatia,10
Portugal,Italy,12
Portugal,France,10
Romania,Moldova,12
Romania,Ukraine,10
San Marino,Italy,12
San Marino,Albania,10
Serbia,Montenegro,12
Serbia,Croatia,10
Slovenia,Croatia,12
Slovenia,Serbia,10
Spain,France,12
Spain,Italy,10
Switzerland,Austria,12
Switzerland,France,10
Ukraine,Moldova,12
Ukraine,Poland,10"""


def fetch_csv(url, cache):
    if os.path.exists(cache):
        return pd.read_csv(cache)
    with urllib.request.urlopen(url) as r:
        data = r.read().decode("utf-8")
    with open(cache, "w") as f:
        f.write(data)
    return pd.read_csv(io.StringIO(data))


def build_context(n_countries=8, min_championed_by=3):
    print("  Loading data …")
    votes_raw = fetch_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
        "master/data/2022/2022-05-17/eurovision-votes.csv",
        "/tmp/eurovision-votes.csv"
    )
    esc_raw = fetch_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
        "master/data/2022/2022-05-17/eurovision.csv",
        "/tmp/eurovision.csv"
    )

    # Build rank map from historical data
    finals = esc_raw[esc_raw["section"] == "grand-final"].dropna(
        subset=["rank", "artist_country"]
    ).copy()
    finals["rank"] = finals["rank"].astype(int)
    rank_map = {(r.year, r.artist_country): r["rank"] for _, r in finals.iterrows()}

    # Supplement 2023–2025
    for year, txt in [(2023, SUPPLEMENT_2023_RESULTS),
                      (2024, SUPPLEMENT_2024_RESULTS),
                      (2025, SUPPLEMENT_2025_RESULTS)]:
        df = pd.read_csv(io.StringIO(txt.strip()))
        for _, row in df.iterrows():
            rank_map[(year, row["country"])] = row["rank"]

    # Historical votes (deduplicated)
    votes_f = votes_raw[
        (votes_raw["semi_final"] == "f") &
        (votes_raw["duplicate"].isna() | (votes_raw["duplicate"] != "x"))
    ].copy()

    # Supplement votes 2023–2025
    supp = []
    for year, txt in [(2023, SUPPLEMENT_2023_VOTES),
                      (2024, SUPPLEMENT_2024_VOTES),
                      (2025, SUPPLEMENT_2025_VOTES)]:
        v = pd.read_csv(io.StringIO(txt.strip()))
        v["year"] = year
        v["semi_final"] = "f"
        v["jury_or_televoting"] = "C"
        v["duplicate"] = None
        supp.append(v[["year", "semi_final", "from_country", "to_country", "points",
                        "jury_or_televoting", "duplicate"]])
    votes_f = pd.concat([votes_f] + supp, ignore_index=True)

    # Aggregate
    votes_agg = votes_f.groupby(
        ["year", "from_country", "to_country"], as_index=False
    )["points"].sum()
    votes_agg["recipient_rank"] = votes_agg.apply(
        lambda r: rank_map.get((r.year, r.to_country), np.nan), axis=1
    )
    votes_agg = votes_agg.dropna(subset=["recipient_rank"])
    votes_agg["recipient_rank"] = votes_agg["recipient_rank"].astype(int)

    dark_horse = votes_agg[
        (votes_agg["points"] >= 10) &
        (votes_agg["recipient_rank"] > 5) &
        (votes_agg["from_country"] != votes_agg["to_country"])
    ].copy()

    # Dark-horse stats
    n_years = votes_agg.groupby("from_country")["year"].nunique().rename("n_years")
    dh_stats = dark_horse.groupby("from_country").agg(
        dh_votes=("points", "count")
    ).join(n_years)
    dh_stats["dh_rate"] = dh_stats["dh_votes"] / dh_stats["n_years"]
    dh_stats = dh_stats[dh_stats["n_years"] >= 5].sort_values("dh_rate", ascending=False)

    dh_countries = dh_stats[
        (dh_stats["dh_rate"] >= dh_stats["dh_rate"].quantile(0.75)) &
        (dh_stats["dh_votes"] >= 5)
    ].index.tolist()

    # Binary matrix: song × country
    dh_pivot = (
        dark_horse[dark_horse["from_country"].isin(dh_countries)]
        .assign(song_id=lambda d: d["year"].astype(str) + " · " + d["to_country"])
        .groupby(["song_id", "from_country"])["points"].sum()
        .unstack(fill_value=0)
    )
    dh_binary = (dh_pivot >= 10).astype(int)
    dh_binary = dh_binary[dh_binary.sum(axis=1) >= 2]

    # Select top N countries by dark-horse rate
    top_c = dh_stats[dh_stats.index.isin(dh_binary.columns)].head(n_countries).index.tolist()
    dh_fca = dh_binary[top_c]

    # Iceberg: only songs championed by ≥ min_championed_by of these countries
    dh_fca = dh_fca[dh_fca.sum(axis=1) >= min_championed_by]

    return dh_fca, dark_horse, dh_stats, rank_map


# ── 2. LATTICE ────────────────────────────────────────────────────────────────

def build_lattice(dh_fca):
    from concepts import Context
    objects    = list(dh_fca.index)
    properties = list(dh_fca.columns)
    bools      = [tuple(bool(v) for v in row) for row in dh_fca.values]
    ctx        = Context(objects, properties, bools)
    return ctx.lattice, properties


def build_graph(lattice, properties, min_extent=3):
    """
    Build a pruned Hasse diagram:
    - Only keep concepts with extent ≥ min_extent, plus top (extent=max) and bottom (extent=0)
    - Edges are cover relations only
    - Reconnect edges around pruned nodes so the diagram stays connected
    """
    # Full graph first
    G_full = nx.DiGraph()
    for i, c in enumerate(lattice):
        G_full.add_node(i,
            n_extent = len(list(c.extent)),
            n_intent = len(list(c.intent)),
            extent   = list(c.extent),
            intent   = list(c.intent),
        )
    for i, c in enumerate(lattice):
        for upper in c.upper_neighbors:
            G_full.add_edge(i, upper.index)

    max_extent = max(G_full.nodes[n]["n_extent"] for n in G_full)

    # Decide which nodes to keep
    keep = {
        n for n in G_full
        if G_full.nodes[n]["n_extent"] >= min_extent
        or G_full.nodes[n]["n_extent"] == max_extent   # top
        or G_full.nodes[n]["n_extent"] == 0            # bottom
    }

    # Build pruned subgraph; reconnect skipped nodes
    G = nx.DiGraph()
    for n in keep:
        G.add_node(n, **G_full.nodes[n])

    # For each kept node, find the nearest kept ancestors (via BFS over full graph)
    def nearest_kept_above(node):
        visited, queue, result = set(), [node], []
        while queue:
            cur = queue.pop(0)
            for nxt in G_full.successors(cur):
                if nxt in visited:
                    continue
                visited.add(nxt)
                if nxt in keep:
                    result.append(nxt)
                else:
                    queue.append(nxt)
        return result

    for n in keep:
        for ancestor in nearest_kept_above(n):
            G.add_edge(n, ancestor)

    # Remove transitive edges (keep only cover relations)
    G = nx.transitive_reduction(G)
    for n in G.nodes():
        G.nodes[n].update({k: v for k, v in G_full.nodes[n].items()})

    return G


# ── 3. LAYOUT ─────────────────────────────────────────────────────────────────

def layered_layout(G):
    layers = defaultdict(list)
    for n, d in G.nodes(data=True):
        layers[d["n_intent"]].append(n)

    layer_keys = sorted(layers.keys())
    y_map = {k: 1.0 - i / max(1, len(layer_keys) - 1)
             for i, k in enumerate(layer_keys)}

    pos = {}
    for k, nodes in layers.items():
        for j, n in enumerate(sorted(nodes)):
            pos[n] = np.array([(j + 1) / (len(nodes) + 1), y_map[k]], dtype=float)

    # Barycentric sweep (5 rounds)
    for _ in range(5):
        for k in layer_keys[1:]:
            for n in layers[k]:
                preds = list(G.predecessors(n))
                if preds:
                    pos[n] = pos[n].copy()
                    pos[n][0] = np.mean([pos[p][0] for p in preds])
        for k in reversed(layer_keys[:-1]):
            for n in layers[k]:
                succs = list(G.successors(n))
                if succs:
                    pos[n] = pos[n].copy()
                    pos[n][0] = np.mean([pos[s][0] for s in succs])

    # Re-space within layers
    for k, nodes in layers.items():
        xs = sorted(nodes, key=lambda n: pos[n][0])
        for j, n in enumerate(xs):
            pos[n] = pos[n].copy()
            pos[n][0] = (j + 1) / (len(xs) + 1)

    return pos


# ── 4. HASSE DIAGRAM ──────────────────────────────────────────────────────────

def node_color(d, max_intent):
    if d["n_intent"] == 0:
        return ACCENT2          # top node: teal
    if d["n_intent"] == max_intent:
        return ACCENT1          # bottom: gold
    # Colour by dominant country in intent
    if len(d["intent"]) == 1:
        return COUNTRY_COLORS.get(d["intent"][0], ACCENT3)
    return ACCENT3              # mixed: rose


def plot_hasse(G, pos, properties, ax):
    ax.set_facecolor(BG)

    max_intent = max(d["n_intent"] for _, d in G.nodes(data=True))
    max_extent = max(d["n_extent"] for _, d in G.nodes(data=True))

    # Edges — vary opacity by "importance" (both ends have large extents)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = (G.nodes[u]["n_extent"] / max_extent) ** 0.4
        ax.plot([x0, x1], [y0, y1],
                color=MID, lw=0.5 + 0.8 * weight,
                alpha=0.3 + 0.4 * weight, zorder=1, solid_capstyle="round")

    # Nodes
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        size = 30 + 260 * (d["n_extent"] / max_extent) ** 0.55
        color = node_color(d, max_intent)
        ax.scatter(x, y, s=size, c=color, zorder=3,
                   linewidths=0.5, edgecolors=WHITE + "55", alpha=0.92)

    # Labels — only for notable nodes
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        intent = d["intent"]
        extent = d["n_extent"]

        # Always label top/bottom and single-country nodes
        if d["n_intent"] == 0:
            label = "ALL\nSONGS"
            fs, fw = 4.5, "bold"
        elif d["n_intent"] == max_intent:
            label = "∅"
            fs, fw = 5, "bold"
        elif len(intent) == 1:
            label = intent[0].split()[0]
            fs, fw = 4.2, "bold"
        elif extent >= max_extent * 0.15:
            abbrs = [c[:3].upper() for c in intent[:3]]
            label = "+".join(abbrs) + (f"\n+{len(intent)-3}" if len(intent) > 3 else "")
            fs, fw = 3.8, "normal"
        else:
            continue

        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, color=WHITE, fontweight=fw,
                fontfamily="monospace", zorder=5, linespacing=1.15,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    ax.set_title("Concept Lattice — Underdog Taste Network",
                 color=ACCENT1, fontsize=10.5, fontweight="bold",
                 pad=8, fontfamily="monospace")

    # Layer tick marks on left
    layers = defaultdict(list)
    for n, d in G.nodes(data=True):
        layers[d["n_intent"]].append(n)
    for k, nodes in sorted(layers.items()):
        y_mean = np.mean([pos[n][1] for n in nodes])
        label = {0: "ALL", max_intent: "SPECIFIC"}.get(k, f"{k} countries")
        ax.text(-0.01, y_mean, label, va="center", ha="right",
                fontsize=4.5, color=GREY, fontfamily="monospace")

    # Country colour legend
    shown = sorted(set(
        d["intent"][0] for _, d in G.nodes(data=True) if len(d["intent"]) == 1
    ))
    legend_patches = [
        mpatches.Patch(color=COUNTRY_COLORS.get(c, ACCENT3), label=c)
        for c in shown
    ] + [
        mpatches.Patch(color=ACCENT3, label="Mixed cluster"),
        mpatches.Patch(color=ACCENT2, label="All songs (top)"),
    ]
    # Country colour legend — bottom right, compact
    leg = ax.legend(handles=legend_patches, loc="lower right",
                    facecolor=SURFACE, edgecolor=MID,
                    labelcolor=WHITE, fontsize=5.0, framealpha=0.9,
                    title="Node colour", title_fontsize=5.0,
                    handlelength=0.9, borderpad=0.5, labelspacing=0.3)
    leg.get_title().set_color(GREY)

    # Node size legend — bottom left, compact
    for frac, label in [(0.15, "3 songs"), (0.5, "~10 songs"), (1.0, "all songs")]:
        sz = 30 + 260 * frac ** 0.55
        ax.scatter([], [], s=sz, c=GREY, alpha=0.7,
                   label=label, edgecolors=WHITE + "44")
    sz_leg = ax.legend(loc="lower left", facecolor=SURFACE, edgecolor=MID,
                       labelcolor=WHITE, fontsize=5.0, framealpha=0.9,
                       title="Node size", title_fontsize=5.0,
                       scatterpoints=1, borderpad=0.5, labelspacing=0.3)
    sz_leg.get_title().set_color(GREY)
    ax.add_artist(leg)  # restore country legend


# ── 5. CLUSTER DETAIL ─────────────────────────────────────────────────────────

def plot_clusters(G, properties, rank_map, ax, top_n=9):
    ax.set_facecolor(BG)

    candidates = [
        (n, G.nodes[n]) for n in G.nodes()
        if 2 <= G.nodes[n]["n_intent"] <= len(properties) - 1
        and G.nodes[n]["n_extent"] >= 3
    ]
    candidates.sort(key=lambda x: (x[1]["n_intent"], x[1]["n_extent"]), reverse=True)
    top = candidates[:top_n]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Most Cohesive Taste Clusters",
                 color=ACCENT1, fontsize=10.5, fontweight="bold",
                 pad=8, fontfamily="monospace")

    if not top:
        ax.text(0.5, 0.5, "No clusters found", ha="center", va="center",
                color=GREY, transform=ax.transAxes)
        return

    n_rows   = len(top)
    bar_h    = 0.80 / n_rows
    pad      = bar_h * 0.10
    dot_x0   = 0.38
    count_x  = 0.96
    max_songs = max(d["n_extent"] for _, d in top)

    # Column headers
    for ci, c in enumerate(properties):
        cx = dot_x0 + (ci + 0.5) / len(properties) * (count_x - dot_x0 - 0.02)
        color = COUNTRY_COLORS.get(c, ACCENT1)
        ax.text(cx, 0.96, c[:3].upper(), ha="center", va="top",
                fontsize=4.0, color=color, fontfamily="monospace",
                fontweight="bold", rotation=55)
    ax.axhline(0.91, color=MID, lw=0.4)
    ax.text(count_x, 0.96, "songs", ha="right", va="top",
            fontsize=4.0, color=GREY, fontfamily="monospace")

    for row_i, (n, d) in enumerate(top):
        y_top = 0.89 - row_i * bar_h
        y_mid = y_top - (bar_h - pad) / 2

        intent = set(d["intent"])
        extent = d["n_extent"]
        songs  = sorted(d["extent"])

        # Row background
        bg = CARD if row_i % 2 == 0 else SURFACE
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y_mid - (bar_h - pad) * 0.5),
            0.98, bar_h - pad,
            boxstyle="round,pad=0.002", color=bg, zorder=1
        ))

        # Cluster label
        label = ", ".join(c.split()[0] for c in sorted(intent))
        ax.text(0.02, y_mid, label, va="center", ha="left",
                fontsize=4.8, color=ACCENT1, fontfamily="monospace",
                fontweight="bold")

        # Dots per country
        for ci, c in enumerate(properties):
            cx = dot_x0 + (ci + 0.5) / len(properties) * (count_x - dot_x0 - 0.02)
            in_intent = c in intent
            ax.scatter(cx, y_mid,
                       s=22 if in_intent else 5,
                       c=COUNTRY_COLORS.get(c, ACCENT2) if in_intent else MID,
                       zorder=3, linewidths=0)

        # Song count bar
        blen = 0.02 + 0.055 * (extent / max_songs)
        ax.add_patch(mpatches.FancyBboxPatch(
            (count_x - blen - 0.005, y_mid - 0.007), blen, 0.014,
            boxstyle="round,pad=0.001", color=ACCENT3, zorder=3, alpha=0.85
        ))
        ax.text(count_x, y_mid, f"{extent}", va="center", ha="right",
                fontsize=5.0, color=WHITE, fontfamily="monospace", fontweight="bold")

        # Show example songs (first 2) as tooltip-style text
        ex = songs[:2]
        ex_text = "  ·  ".join(
            f"{s}  (#{rank_map.get((int(s.split('·')[0].strip()), s.split('·')[1].strip()), '?')})"
            for s in ex
        )
        if ex_text:
            ax.text(0.02, y_mid - (bar_h - pad) * 0.33, ex_text,
                    va="center", ha="left", fontsize=3.5,
                    color=GREY, fontfamily="monospace", fontstyle="italic")

    ax.text(0.5, 0.005,
            "Filled dots = countries in the cluster  ·  Number = shared underdog songs  ·  #n = final placement",
            ha="center", va="bottom", fontsize=4.2, color=GREY, fontstyle="italic")


# ── 6. CO-OCCURRENCE HEATMAP ──────────────────────────────────────────────────

def plot_cooccurrence(dark_horse, dh_stats, properties, ax):
    ax.set_facecolor(BG)

    top_c = [c for c in dh_stats.sort_values("dh_rate", ascending=False).index
             if c in properties]

    dh_piv = (
        dark_horse[dark_horse["from_country"].isin(top_c)]
        .assign(song_id=lambda d: d["year"].astype(str) + " · " + d["to_country"])
        .groupby(["song_id", "from_country"])["points"].sum()
        .unstack(fill_value=0)
    )
    dh_bin = (dh_piv >= 10).reindex(columns=top_c, fill_value=False).astype(int)
    comat = dh_bin.values.T @ dh_bin.values
    comat_copy = comat.copy()
    np.fill_diagonal(comat_copy, 0)
    co_df = pd.DataFrame(comat_copy, index=top_c, columns=top_c)

    im = ax.imshow(co_df.values, cmap=CMAP_HEAT, aspect="auto",
                   interpolation="nearest", vmin=0)

    ticks = list(range(len(top_c)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    xlabels = [c.split()[0] for c in top_c]  # First word only
    ax.set_xticklabels(xlabels, rotation=40, ha="right",
                       fontsize=7, color=WHITE, fontfamily="monospace")
    ax.set_yticklabels(xlabels, fontsize=7, color=WHITE, fontfamily="monospace")

    # Colour tick labels by country colour
    for label, c in zip(ax.get_xticklabels(), top_c):
        label.set_color(COUNTRY_COLORS.get(c, WHITE))
    for label, c in zip(ax.get_yticklabels(), top_c):
        label.set_color(COUNTRY_COLORS.get(c, WHITE))

    ax.tick_params(colors=GREY, length=2)

    vmax = co_df.values.max()
    for i in range(len(top_c)):
        for j in range(len(top_c)):
            val = co_df.values[i, j]
            if val > 0:
                tc = BG if val > vmax * 0.55 else WHITE
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=6, color=tc, fontfamily="monospace",
                        fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(labelsize=6, colors=GREY)
    cbar.set_label("Shared underdog picks", fontsize=6.5, color=GREY)
    cbar.outline.set_edgecolor(MID)

    ax.set_title("Pairwise Shared Underdog Taste",
                 color=ACCENT1, fontsize=10.5, fontweight="bold",
                 pad=8, fontfamily="monospace")

    for i in range(len(top_c) + 1):
        ax.axhline(i - 0.5, color=BG, lw=0.7)
        ax.axvline(i - 0.5, color=BG, lw=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(MID)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("[1] Building FCA context (2004–2025) …")
    dh_fca, dark_horse, dh_stats, rank_map = build_context(
        n_countries=8, min_championed_by=3
    )
    properties = list(dh_fca.columns)
    print(f"    Countries : {properties}")
    print(f"    Songs     : {len(dh_fca)}")

    print("[2] Computing concept lattice …")
    lattice, _ = build_lattice(dh_fca)

    print("[3] Building pruned Hasse graph …")
    G = build_graph(lattice, properties, min_extent=3)
    print(f"    {G.number_of_nodes()} concepts, {G.number_of_edges()} edges")

    print("[4] Computing layout …")
    pos = layered_layout(G)

    print("[5] Rendering …")
    fig = plt.figure(figsize=(20, 26), facecolor=BG)
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.977,
             "DARK HORSE SUPPORTER NETWORK",
             ha="center", va="top", fontsize=18, fontweight="bold",
             color=ACCENT1, fontfamily="monospace")
    fig.text(0.5, 0.965,
             "Eurovision Song Contest  ·  Countries that gave ≥10 pts to songs finishing outside the Top 5",
             ha="center", va="top", fontsize=8.5, color=GREY, fontfamily="monospace")
    fig.text(0.5, 0.956,
             "Grand Finals 2004–2025  ·  Iceberg lattice: concepts covering ≥3 underdog songs",
             ha="center", va="top", fontsize=7.5, color=GREY + "aa", fontfamily="monospace")

    ax_hasse   = fig.add_axes([0.02, 0.38, 0.54, 0.56])
    ax_cluster = fig.add_axes([0.58, 0.38, 0.40, 0.56])
    ax_heat    = fig.add_axes([0.08, 0.05, 0.84, 0.28])

    plot_hasse(G, pos, properties, ax_hasse)
    plot_clusters(G, properties, rank_map, ax_cluster)
    plot_cooccurrence(dark_horse, dh_stats, properties, ax_heat)

    fig.text(0.5, 0.008,
             "Data: TidyTuesday (1975–2022) + official eurovision.tv results (2023–2025)  ·  "
             "FCA via concepts library  ·  Top 8 dark-horse countries by average high-point votes to underdogs",
             ha="center", va="bottom", fontsize=5.5, color=GREY, fontfamily="monospace")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "esc_output")
    os.makedirs(out_dir, exist_ok=True)

    pdf_path = os.path.join(out_dir, "lattice_visualization.pdf")
    png_path = os.path.join(out_dir, "lattice_visualization.png")
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)

    print(f"\n✓  {pdf_path}")
    print(f"✓  {png_path}\n")


if __name__ == "__main__":
    main()