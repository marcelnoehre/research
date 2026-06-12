"""
Atlas (gluing) decomposition of a concept lattice via block relations.

Background
----------
Tolerance relations on B(G, M, I) correspond bijectively to *block relations*:
relations J with I subset of J subset of G x M whose every row is an intent and
every column an extent of (G, M, I).  Given such a J:

  * the SKELETON is B(G, M, J)            (the factor lattice L / Theta)
  * each skeleton concept (A, B) yields a CHART, the interval
        [ (B^I, B^II), (A^II, A^I) ]  in B(G, M, I)
    (lower bound = attribute-concept of B, upper bound = object-concept of A).

Charts overlap along the skeleton's covering relation -- that overlap is the
gluing, and is the whole point of an atlas.  Connected components of the arrow
graph cannot express it because they partition G and M into disjoint pieces.

The finest non-trivial atlas is seeded by the double arrows: take them together
with I and close (rows -> intents, columns -> extents) to the smallest block
relation containing them.

Everything below works on a boolean pandas DataFrame (index = objects,
columns = attributes).  `atlas_decomposition` accepts a fcapy context and routes
through `to_pandas`, matching the original pipeline.
"""

from itertools import product
import pandas as pd
from data import Parser

try:
    from fcapy.context.converters import to_pandas  # noqa: F401
except Exception:  # keep the core usable / testable without fcapy
    to_pandas = None


# --------------------------------------------------------------------------- #
# derivation operators in a fixed context df  (df bool, index=G, cols=M)
# --------------------------------------------------------------------------- #
def _extent(df, attrs):
    """{ g : forall m in attrs, (g,m) in I }  (empty attrs -> all objects)."""
    attrs = list(attrs)
    if not attrs:
        return set(df.index)
    return set(df.index[df[attrs].all(axis=1)])


def _intent(df, objs):
    """{ m : forall g in objs, (g,m) in I }  (empty objs -> all attributes)."""
    objs = list(objs)
    if not objs:
        return set(df.columns)
    return set(df.columns[df.loc[objs].all(axis=0)])


def _attr_closure(df, attrs):   # B -> B'' : smallest intent containing attrs
    return _intent(df, _extent(df, attrs))


def _obj_closure(df, objs):     # A -> A'' : smallest extent containing objs
    return _extent(df, _intent(df, objs))


# --------------------------------------------------------------------------- #
# arrow relations  (corrected `standard_context`)
# --------------------------------------------------------------------------- #
def arrow_relations(df):
    """Return (down, up, double) sets of (g, m) pairs.

    g down-arrow m : (g,m) not in I and  g' < h'  => (h,m) in I  for all objects h
    g up-arrow   m : (g,m) not in I and  m' < n'  => (g,n) in I  for all attributes n
    """
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


# --------------------------------------------------------------------------- #
# block relation: smallest J with I subset J containing `seed`
# --------------------------------------------------------------------------- #
def block_relation_closure(df, seed):
    """Smallest block relation containing I (=df) and the seed pairs.

    Alternately close every row to an intent and every column to an extent of
    (G, M, I) until stable.  Both operations only add pairs and any block
    relation containing the seed must contain these closures, so the fixpoint is
    the least block relation >= seed.
    """
    J = df.copy()
    for g, m in seed:
        J.loc[g, m] = True

    changed = True
    while changed:
        changed = False
        for g in J.index:                                  # rows -> intents
            closed = _attr_closure(df, J.columns[J.loc[g]])
            for m in closed:
                if not J.loc[g, m]:
                    J.loc[g, m] = True
                    changed = True
        for m in J.columns:                                # columns -> extents
            closed = _obj_closure(df, J.index[J[m]])
            for g in closed:
                if not J.loc[g, m]:
                    J.loc[g, m] = True
                    changed = True
    return J


# --------------------------------------------------------------------------- #
# concepts of a context (extents = closure system of column extents)
# --------------------------------------------------------------------------- #
def concepts(ctx):
    """All concepts of boolean DataFrame `ctx` as (frozenset extent, frozenset intent)."""
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


# --------------------------------------------------------------------------- #
# the atlas
# --------------------------------------------------------------------------- #
def atlas_from_df(df, seed=None):
    """Atlas decomposition of B(G, M, I).  Default seed = double arrows (finest atlas)."""
    df = df.astype(bool)
    if seed is None:
        seed = arrow_relations(df)[2]            # double arrows

    J = block_relation_closure(df, seed)
    skeleton = concepts(J)
    L = concepts(df)                             # concepts of (G, M, I)

    charts = []
    for A, B in skeleton:
        low_ext = frozenset(_extent(df, B))               # B^I
        up_ext = frozenset(_obj_closure(df, A))           # A^II
        members = sorted(
            ((tuple(sorted(X)), tuple(sorted(Y))) for X, Y in L
             if low_ext <= X <= up_ext),
            key=lambda c: len(c[0]),
        )
        charts.append({
            "skeleton_extent": tuple(sorted(A)),
            "skeleton_intent": tuple(sorted(B)),
            "lower_concept": (tuple(sorted(low_ext)),
                              tuple(sorted(_intent(df, low_ext)))),
            "upper_concept": (tuple(sorted(up_ext)),
                              tuple(sorted(_intent(df, up_ext)))),
            "concepts": members,
        })
    charts.sort(key=lambda c: (len(c["lower_concept"][0]), c["skeleton_extent"]))

    # gluing = covers between charts, ordered by their upper extent (a real concept)
    tops = [frozenset(c["upper_concept"][0]) for c in charts]
    chart_covers = [
        (i, j)
        for i in range(len(charts)) for j in range(len(charts))
        if tops[i] < tops[j] and not any(
            tops[i] < tops[k] < tops[j] for k in range(len(charts))
        )
    ]
    return {
        "block_relation": J,
        "skeleton_size": len(skeleton),
        "chart_covers": chart_covers,          # pairs of chart positions that are glued
        "charts": charts,
    }


def atlas_decomposition(cxt, seed=None, lattice=None):
    """Drop-in entry point for a fcapy context.

    Charts are reported by *lattice index*: every concept becomes its position in
    the original lattice B(G, M, I).  Pass your own `lattice` (a fcapy
    ConceptLattice) to match the numbering used elsewhere in your pipeline;
    otherwise one is built from `cxt`.

    Each chart gains:
        lower, upper   : index of the chart's bottom / top concept
        concepts       : sorted indices of every concept in the chart
    `chart_covers` lists pairs of chart positions that are glued.  The
    extent/intent tuples stay under *_concept / concept_tuples for names.
    """
    if to_pandas is None:
        raise RuntimeError("fcapy not available; call atlas_from_df on a DataFrame.")

    res = atlas_from_df(to_pandas(cxt), seed=seed)

    if lattice is None:
        from fcapy.lattice import ConceptLattice
        lattice = ConceptLattice.from_context(cxt)
    idx = {frozenset(c.extent): lattice.index(c) for c in lattice.elements}

    def ix(extent_tuple):
        return idx[frozenset(extent_tuple)]

    for ch in res["charts"]:
        ch["concept_tuples"] = ch.pop("concepts")            # keep names available
        ch["concepts"] = sorted(ix(e) for e, _ in ch["concept_tuples"])
        ch["lower"] = ix(ch["lower_concept"][0])
        ch["upper"] = ix(ch["upper_concept"][0])

    res["lattice"] = lattice
    return res

if __name__ == "__main__":
    file = '../../data/6'
    ctx = Parser().decode_cxt(f'{file}.cxt')
    df = to_pandas(ctx).astype(bool)

    down, up, double = arrow_relations(df)
    print('down only :', sorted(down - double))
    print('up only   :', sorted(up - double))
    print('double    :', sorted(double))
    print()

    res = atlas_decomposition(ctx)                  # or atlas_decomposition(ctx, lattice=L)
    print(f'atlas: {res["skeleton_size"]} chart(s); glued: {res["chart_covers"]}')
    for i, ch in enumerate(res['charts']):
        print(f'\n  chart {i}')
        print(f'    skeleton concept: {ch["skeleton_extent"]} / {ch["skeleton_intent"]}')
        print(f'    interval: {ch["lower"]}  ->  {ch["upper"]}')
        print(f'    concepts (indices): {ch["concepts"]}')
        print(f'    concepts (extents): {[e for e, _ in ch["concept_tuples"]]}')