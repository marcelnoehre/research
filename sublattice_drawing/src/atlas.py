from itertools import product
import networkx as nx
import pandas as pd


def _extent(df, attrs):
    attrs = list(attrs)
    if not attrs:
        return set(df.index)
    return set(df.index[df[attrs].all(axis=1)])

def _intent(df, objs):
    objs = list(objs)
    if not objs:
        return set(df.columns)
    return set(df.columns[df.loc[objs].all(axis=0)])

def _attr_closure(df, attrs):
    return _intent(df, _extent(df, attrs))


def _obj_closure(df, objs):
    return _extent(df, _intent(df, objs))

def _reduced_context(G):
    join_irr = [x for x in G if G.out_degree(x) == 1]
    meet_irr = [x for x in G if G.in_degree(x) == 1]
    leq = {x: nx.ancestors(G, x) | {x} for x in G}
    df = pd.DataFrame(False, index=join_irr, columns=meet_irr)
    for j in join_irr:
        for m in meet_irr:
            if m in leq[j]:
                df.at[j, m] = True
    return df


def _node_index(G):
    join_irr = [x for x in G if G.out_degree(x) == 1]
    leq = {x: nx.ancestors(G, x) | {x} for x in G}
    sig = {}
    for x in G:
        ext = frozenset(j for j in join_irr if x in leq[j])   # { j : j <= x }
        sig[ext] = x
    return sig

def _concepts(ctx):
    G = set(ctx.index)
    col_ext = {m: set(ctx.index[ctx[m]]) for m in ctx.columns}
    extents, frontier = {frozenset(G)}, [set(G)]
    while frontier:
        A = frontier.pop()
        for m in ctx.columns:
            B = frozenset(A & col_ext[m])
            if B not in extents:
                extents.add(B)
                frontier.append(set(B))
    return {(A, frozenset(_intent(ctx, A))) for A in extents}


def _arrow_relation(df):
    objects, attributes = list(df.index), list(df.columns)
    intents = {g: set(df.columns[df.loc[g]]) for g in objects}
    extents = {m: set(df.index[df[m]]) for m in attributes}
    down, up, dbl = set(), set(), set()
    for g, m in product(objects, attributes):
        if df.loc[g, m]:
            continue
        is_down = all(m in intents[h] for h in objects if intents[g] < intents[h])
        is_up = all(g in extents[n] for n in attributes if extents[m] < extents[n])
        if is_down:
            down.add((g, m))
        if is_up:
            up.add((g, m))
        if is_down and is_up:
            dbl.add((g, m))
    return down, up, dbl


def _block_relation_closure(df, seed):
    """Smallest block relation containing I (=df) and the seed pairs."""
    J = df.copy()
    for g, m in seed:
        J.loc[g, m] = True
    changed = True
    while changed:
        changed = False
        for g in J.index:                                  # rows -> intents
            for m in _attr_closure(df, J.columns[J.loc[g]]):
                if not J.loc[g, m]:
                    J.loc[g, m] = True
                    changed = True
        for m in J.columns:                                # columns -> extents
            for g in _obj_closure(df, J.index[J[m]]):
                if not J.loc[g, m]:
                    J.loc[g, m] = True
                    changed = True
    return J

def atlas_decomposition(G):
    """Atlas decomposition of the lattice `G`.

    Returns a dict with `skeleton_size`, `chart_covers` (pairs of chart positions
    that are glued), and `charts`.  Every chart reports its concepts as the
    corresponding nodes of G:
        lower, upper     : G node at the bottom / top of the chart interval
        concepts         : G nodes in the chart (ordered bottom -> top)
        concept_extents  : the raw join-irreducible extents, for reference
        skeleton_extent / skeleton_intent : the skeleton concept (raw irreducibles)
    """
    df = _reduced_context(G)
    node_of = _node_index(G)

    seed = _arrow_relation(df)[2]                 # double arrows
    J = _block_relation_closure(df, seed)

    skeleton = _concepts(J)
    L = _concepts(df)

    charts = []
    for A, B in skeleton:
        low_ext = frozenset(_extent(df, B))           # B^I  (chart bottom)
        up_ext = frozenset(_obj_closure(df, A))       # A^II (chart top)
        members = sorted((X for X, _ in L if low_ext <= X <= up_ext), key=len)
        charts.append({
            "_low": low_ext, "_up": up_ext,
            "skeleton_extent": tuple(sorted(map(str, A))),
            "skeleton_intent": tuple(sorted(map(str, B))),
            "lower": node_of[low_ext],
            "upper": node_of[up_ext],
            "concepts": [node_of[X] for X in members],
            "concept_extents": [tuple(sorted(map(str, X))) for X in members],
        })

    charts.sort(key=lambda c: (len(c["_low"]), c["skeleton_extent"]))

    tops = [c["_up"] for c in charts]
    chart_covers = [
        (i, j)
        for i in range(len(charts)) for j in range(len(charts))
        if tops[i] < tops[j] and not any(
            tops[i] < tops[k] < tops[j] for k in range(len(charts))
        )
    ]
    for c in charts:                                  # drop internals
        del c["_low"], c["_up"]

    return {
        "block_relation": J,
        "skeleton_size": len(skeleton),
        "chart_covers": chart_covers,
        "charts": charts,
    }


def chart_overlap(res, i, j):
    """Shared G nodes of charts i and j (their gluing). An interval, hence a lattice."""
    return sorted(set(res["charts"][i]["concepts"]) & set(res["charts"][j]["concepts"]),
                  key=str)


def gluings(res):
    """{(i, j): shared G nodes} for every glued (covering) pair of charts."""
    return {(i, j): chart_overlap(res, i, j) for i, j in res["chart_covers"]}