from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
from shapely.geometry import LineString
from shapely.strtree import STRtree

def find_intersections(
        edges_list: List[Tuple],
        coords: Dict,
) -> List[Tuple]:
    '''
    Return all edge-edge intersections (non-adjacent edges only).

    Parameters
    ----------
    edges_list : List
        list of (u, v) edge tuples
    coords : Dict 
        dictionary mapping node id to (x, y) coordinates

    Returns
    -------
    intersections : List
        list of (edge_i, edge_j, shapely_point)
    '''
    lines = [LineString([coords[e[0]], coords[e[1]]]) for e in edges_list]
    tree = STRtree(lines)

    intersections = []
    for i, line in enumerate(lines):
        for j in tree.query(line, predicate='intersects'):
            # symmetric invariant
            if j <= i:
                continue
            # common endpoint
            if set(edges_list[i]) & set(edges_list[j]):
                continue
            # geometric intersection
            pt = line.intersection(lines[j])
            if not pt.is_empty:
                intersections.append((i, j, pt))

    return intersections


def build_planar_graph(
        edges_list: List[Tuple],
        coords: Dict,
        intersections: List[Tuple],
) -> nx.Graph:
    '''
    Build a planar NetworkX graph by splitting each edge at its intersections.

    Original concept nodes are keyed by their integer id; synthetic
    intersection nodes are keyed by their (x, y) float tuple.

    Parameters
    ----------
    edges_list : List
        list of (u, v) edge tuples (original lattice edges)
    coords : Dict
        dictionary mapping node id to (x, y) coordinates
    intersections : List
        list of intersections

    Returns
    -------
    planar_graph : nx.Graph 
        graph with a 'pos' attribute on every node
    '''
    G = nx.Graph()

    # original nodes
    for node_idx, pos in coords.items():
        G.add_node(node_idx, pos=pos)

    # derive coordinates of dummy vertices from intersections 
    edge_crossings: Dict[int, list] = defaultdict(list)
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
            edge_crossings[edge_idx].append((t, new_node))

    # subdivide edges after inserting dummy vertices
    for edge_idx, e in enumerate(edges_list):
        splits = sorted(edge_crossings[edge_idx])
        chain = [e[0]] + [node for _, node in splits] + [e[1]]
        for a, b in zip(chain, chain[1:]):
            G.add_edge(a, b)

    return G
