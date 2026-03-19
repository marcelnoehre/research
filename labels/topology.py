"""
topology.py
-----------
Extracts planar faces from a combinatorial embedding and computes basic
topological invariants (first Betti number β₁).

Faces are computed via NetworkX's planar embedding half-edge traversal.
The outer (unbounded) face is identified as the face with the largest area.
"""

from typing import Dict, List, Tuple

import networkx as nx
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _shoelace(G: nx.Graph, nodes) -> float:
    """Area of a polygon via the shoelace formula."""
    pts = [G.nodes[n]['pos'] for n in nodes]
    n = len(pts)
    return abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )) / 2


def _centroid(G: nx.Graph, nodes) -> Tuple[float, float]:
    """Centroid of a polygon given an ordered list of graph nodes."""
    pts = [G.nodes[n]['pos'] for n in nodes]
    c = Polygon(pts).centroid
    return c.x, c.y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def betti_1(G: nx.Graph) -> int:
    """
    First Betti number β₁ = E - V + C (number of independent cycles).

    Parameters
    ----------
    G : planar graph (after edge splitting)
    """
    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G)
    return E - V + C


def extract_faces(G: nx.Graph):
    """
    Extract all faces of a planar graph using its combinatorial embedding.

    Parameters
    ----------
    G : planar nx.Graph (must actually be planar)

    Returns
    -------
    bounded_faces : list of node-lists for each bounded (interior) face
    outer_nodes   : ordered node-list of the outer (unbounded) face
    areas         : area of each bounded face (same order as bounded_faces)
    centers       : centroid of each bounded face (same order as bounded_faces)

    Raises
    ------
    ValueError if G is not planar.
    """
    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Graph is not planar — cannot extract faces.")

    # Traverse every directed half-edge to collect unique faces
    seen = set()
    faces = []
    for u, v in G.edges():
        for a, b in [(u, v), (v, u)]:
            face = embedding.traverse_face(a, b)
            key = frozenset(face)
            if key not in seen:
                seen.add(key)
                faces.append(face)

    # The outer face is the one enclosing all others → largest area
    face_areas = [(f, _shoelace(G, f)) for f in faces]
    outer_face = max(face_areas, key=lambda x: x[1])[0]
    outer_nodes: List = list(outer_face)

    bounded = [(f, a) for f, a in face_areas if f is not outer_face]
    bounded_faces = [f for f, _ in bounded]
    areas         = [a for _, a in bounded]
    centers       = [_centroid(G, f) for f in bounded_faces]

    return bounded_faces, outer_nodes, areas, centers
