"""
intersection.py
---------------
Detects geometric edge-edge intersections in a straight-line graph drawing
and builds a planar NetworkX graph by splitting edges at those intersections.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
from shapely.geometry import LineString
from shapely.strtree import STRtree


def find_intersections(
        edges_list: List[Tuple],
        coords: Dict,
) -> List[Tuple]:
    """
    Return all proper edge–edge intersections (non-adjacent edges only).

    Parameters
    ----------
    edges_list : list of (u, v) edge tuples
    coords     : dict mapping node id → (x, y)

    Returns
    -------
    list of (edge_i, edge_j, shapely_point)
    """
    lines = [LineString([coords[e[0]], coords[e[1]]]) for e in edges_list]
    tree = STRtree(lines)

    intersections = []
    for i, line in enumerate(lines):
        for j in tree.query(line, predicate='intersects'):
            if j <= i:
                continue
            # Skip adjacent edges — they always touch at the shared endpoint
            if set(edges_list[i]) & set(edges_list[j]):
                continue
            pt = line.intersection(lines[j])
            if not pt.is_empty:
                intersections.append((i, j, pt))

    return intersections


def build_planar_graph(
        edges_list: List[Tuple],
        coords: Dict,
        intersections: List[Tuple],
) -> nx.Graph:
    """
    Build a planar NetworkX graph by splitting each edge at its intersections.

    Original concept nodes are keyed by their integer id; synthetic
    intersection nodes are keyed by their (x, y) float tuple.

    Parameters
    ----------
    edges_list    : list of (u, v) edge tuples (original lattice edges)
    coords        : dict mapping node id → (x, y)
    intersections : output of find_intersections()

    Returns
    -------
    nx.Graph with a 'pos' attribute on every node
    """
    G = nx.Graph()

    for node_idx, pos in coords.items():
        G.add_node(node_idx, pos=pos)

    # Collect split points per edge, parameterised by t ∈ [0, 1]
    edge_splits: Dict[int, list] = defaultdict(list)

    for i, j, pt in intersections:
        x, y = pt.coords[0]
        new_node = (x, y)
        G.add_node(new_node, pos=(x, y))
        for edge_idx in (i, j):
            e = edges_list[edge_idx]
            x0, y0 = coords[e[0]]
            x1, y1 = coords[e[1]]
            dx, dy = x1 - x0, y1 - y0
            denom = dx * dx + dy * dy
            t = ((x - x0) * dx + (y - y0) * dy) / denom if denom else 0
            edge_splits[edge_idx].append((t, new_node))

    # Replace each original edge with a chain of sub-edges through split points
    for edge_idx, e in enumerate(edges_list):
        splits = sorted(edge_splits[edge_idx])
        chain = [e[0]] + [node for _, node in splits] + [e[1]]
        for a, b in zip(chain, chain[1:]):
            G.add_edge(a, b)

    return G
