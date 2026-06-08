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


def parts_as_lattices(G, lat):
    """
    Return (anchors, enriched_parts).

    enriched_parts: list of (a, b, region, sub_lat, index_map) where
      sub_lat   -- ConceptLattice for the concepts in this region, with
                   local indices 0, 1, ..., len(region)-1.
      index_map -- dict mapping original concept index (node id in G) to the
                   new local index, assigned in topological order of G.
    """
    from fcapy.lattice import ConceptLattice

    anchors, parts = find_anchor_pairs(G)
    topo = list(nx.topological_sort(G))

    enriched = []
    for a, b, region in parts:
        region_topo = [v for v in topo if v in region]
        index_map = {i: v for i, v in enumerate(region_topo)}
        orig_to_local = {v: i for i, v in index_map.items()}

        children_dict = {
            orig_to_local[v]: [orig_to_local[w] for w in G.successors(v) if w in region]
            for v in region
        }
        sub_lat = ConceptLattice(
            [lat[v] for v in region_topo],
            children_dict=children_dict,
        )
        enriched.append((a, b, region, sub_lat, index_map))

    return anchors, enriched
