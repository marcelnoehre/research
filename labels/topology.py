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

def normalize_positions(G: nx.Graph, target_height: float = 10.0) -> float:
    """
    Normalize all node 'pos' attributes in-place so that the graph's
    bounding box has a height of *target_height*, with the aspect ratio
    preserved and the origin placed at the bounding-box minimum.

    Parameters
    ----------
    G             : graph whose nodes have a 'pos' attribute
    target_height : desired height of the normalized coordinate space
                    (default 10.0)

    Returns
    -------
    mm_per_unit : scale factor — multiply normalized coordinates by this
                  to convert to mm, once you know the physical height:
                      mm_per_unit = physical_height_mm / target_height
                  Stored on the graph as G.graph['mm_per_unit_at_1mm_height']
                  so callers can recompute for any physical height.

    Example
    -------
    normalize_positions(G, target_height=10.0)
    mm_per_unit = 100.0 / 10.0   # drawing will be 100 mm tall
    fits = face_w * mm_per_unit >= label_w_mm
    """
    xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    ys = [G.nodes[n]['pos'][1] for n in G.nodes]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    height = max_y - min_y
    if height == 0:
        raise ValueError("All nodes have the same y-coordinate — cannot normalize.")

    scale = target_height / height

    for n in G.nodes:
        x, y = G.nodes[n]['pos']
        G.nodes[n]['pos'] = ((x - min_x) * scale, (y - min_y) * scale)

    # Store normalization metadata on the graph for later use.
    # min_x, min_y and scale let callers transform any raw point with:
    #   norm_pt = ((x - min_x) * scale, (y - min_y) * scale)
    G.graph['normalized_height'] = target_height
    G.graph['normalized_width']  = (max_x - min_x) * scale
    G.graph['norm_min_x']        = min_x
    G.graph['norm_min_y']        = min_y
    G.graph['norm_scale']        = scale

    return scale


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
    seen  = set()
    faces = []
    for u, v in G.edges():
        for a, b in [(u, v), (v, u)]:
            face = embedding.traverse_face(a, b)
            key  = frozenset(face)
            if key not in seen:
                seen.add(key)
                faces.append(face)

    # The outer face is the one enclosing all others → largest area
    face_areas = [(f, _shoelace(G, f)) for f in faces]
    outer_face = max(face_areas, key=lambda x: x[1])[0]
    outer_nodes: List = list(outer_face)

    bounded      = [(f, a) for f, a in face_areas if f is not outer_face]
    bounded_faces = [f for f, _ in bounded]
    areas         = [a for _, a in bounded]
    centers       = [_centroid(G, f) for f in bounded_faces]

    return bounded_faces, outer_nodes, areas, centers


def label_fits(G: nx.Graph,
               face_nodes,
               label_w_mm: float,
               label_h_mm: float,
               physical_height_mm: float) -> bool:
    """
    Check whether a label of size (label_w_mm x label_h_mm) fits inside
    the axis-aligned bounding box of a face, given the physical drawing height.

    Requires normalize_positions() to have been called first.

    Parameters
    ----------
    G                  : graph with normalized 'pos' attributes
    face_nodes         : ordered node-list of the face (from extract_faces)
    label_w_mm         : label ink width in mm  (from label.py)
    label_h_mm         : label ink height in mm (from label.py)
    physical_height_mm : how tall the drawing is in mm
    """
    if 'normalized_height' not in G.graph:
        raise ValueError("Call normalize_positions(G) before label_fits().")

    mm_per_unit = physical_height_mm / G.graph['normalized_height']

    pts = [G.nodes[n]['pos'] for n in face_nodes]
    minx, miny, maxx, maxy = Polygon(pts).bounds

    face_w_mm = (maxx - minx) * mm_per_unit
    face_h_mm = (maxy - miny) * mm_per_unit

    return face_w_mm >= label_w_mm and face_h_mm >= label_h_mm