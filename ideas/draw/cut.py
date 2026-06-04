"""
Decompose a DAG (Hasse diagram / transitive reduction) into its parts.

Anchor nodes are the nodes that lie on EVERY maximal chain from top to
bottom. They form a totally ordered sequence top = a0 < a1 < ... < ak =
bottom, and each part is the region between two consecutive anchors:
all chains inside it start at a_i and end at a_(i+1), and no edge
bypasses an anchor.

Works directly on the output of nx.transitive_reduction(lat.to_networkx()).
"""

import networkx as nx


def _immediate_dominators(G, topo):
    """idom for every node (Cooper-Harvey-Kennedy; one topological pass
    suffices on a DAG). topo[0] must be the unique source of G."""
    root = topo[0]
    idom = {root: root}
    depth = {root: 0}
    for v in topo[1:]:
        preds = iter(G.predecessors(v))
        a = next(preds)  # every non-root node has >= 1 predecessor
        for b in preds:
            x, y = a, b
            while x != y:  # walk up to the nearest common ancestor
                if depth[x] >= depth[y]:
                    x = idom[x]
                else:
                    y = idom[y]
            a = x
        idom[v] = a
        depth[v] = depth[a] + 1
    return idom


def find_anchor_pairs(G):
    """
    Return (anchors, parts).

    anchors: list of nodes lying on every maximal top-to-bottom chain,
             in order from top to bottom.
    parts:   list of (A, B, region) for consecutive anchors A, B, where
             region = all nodes X with A <= X <= B. Concatenating the
             parts covers the whole graph; neighbouring parts share
             exactly their anchor node.
    """
    topo = list(nx.topological_sort(G))
    sources = [v for v in topo if G.in_degree(v) == 0]
    sinks = [v for v in topo if G.out_degree(v) == 0]
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError(
            "Graph must have a unique source and sink (a concept lattice "
            "always does); add a virtual top/bottom otherwise."
        )

    dom = _immediate_dominators(G, topo)

    # a node lies on every maximal chain  <=>  it dominates the bottom;
    # so the anchors are exactly the dominator chain of the sink
    anchors = [sinks[0]]
    while anchors[-1] != sources[0]:
        anchors.append(dom[anchors[-1]])
    anchors.reverse()

    parts = []
    for a, b in zip(anchors, anchors[1:]):
        region = (nx.descendants(G, a) & nx.ancestors(G, b)) | {a, b}
        parts.append((a, b, region))
    return anchors, parts


if __name__ == "__main__":
    from data import Parser
    from fcapy.lattice import ConceptLattice

    file = '7'
    parser = Parser()
    cxt = parser.decode_cxt(f'../../data/{file}.cxt')
    lat = ConceptLattice.from_context(cxt)
    G = nx.transitive_reduction(lat.to_networkx())

    anchors, parts = find_anchor_pairs(G)

    print(f"anchors (top -> bottom): {anchors}\n")
    print(f"{len(parts)} part(s):")
    for a, b, region in parts:
        inner = len(region) - 2
        print(f"A = {a}  ->  B = {b}   ({inner} node(s) enclosed)")
        print(f"    region: {sorted(region, key=str)}")