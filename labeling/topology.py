import networkx as nx

from typing import List, Tuple, Set
from shapely.geometry import Polygon

def _shoelace(G: nx.Graph, nodes: List, scale: float) -> float:
    '''
    Area of a polygon via the shoelace formula
    '''
    pts = [G.nodes[n]['pos'] for n in nodes]
    n = len(pts)
    norm_area = abs(sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    )) / 2
    return norm_area / (scale**2)

def _centroid(G: nx.Graph, nodes: List) -> Tuple[float, float]:
    '''
    Centroid of a polygon given an ordered list of graph nodes
    '''
    pts = [G.nodes[n]['pos'] for n in nodes]
    c = Polygon(pts).centroid
    return c.x, c.y

def _clockwise(node_ids, G):
    edge_sum = 0
    pos = nx.get_node_attributes(G, 'pos') 

    for i in range(len(node_ids)):
        u = node_ids[i]
        v = node_ids[(i + 1) % len(node_ids)]
        p1, p2 = pos[u], pos[v]
        edge_sum += (p2[0] - p1[0]) * (p2[1] + p1[1])

    lst = node_ids if edge_sum >= 0 else node_ids[::-1]
    zero_index = lst.index(0)
    return lst[zero_index:] + lst[:zero_index]

def betti_1(G: nx.Graph) -> int:
    '''
    First Betti number (number of independent cycles) 
    β₁ = E - V + C

    Parameters
    ----------
    G : nx.Graph
        planar graph
    '''
    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = nx.number_connected_components(G)
    return E - V + C

def extract_faces(G: nx.Graph, scale: float):
    '''
    Extract all faces of a planar graph using its combinatorial embedding.

    Parameters
    ----------
    G : nx.Graph
        planar graph
    scale : float
        mm per unit

    Returns
    -------
    bounded_faces : List[List]
        list of node-lists for each bounded face
    outer_nodes : List[int]
        ordered node-list of the outer face
    areas : List[float]
        area of each bounded face (same order as bounded_faces)
    centers : List[float]
        centroid of each bounded face (same order as bounded_faces)

    Raises
    ------
    ValueError if G is not planar
    '''
    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError('Graph is not planar — cannot extract faces!')

    # traverse every directed half-edge to collect unique faces
    seen  = set()
    faces = []
    for u, v in G.edges():
        for a, b in [(u, v), (v, u)]:
            face = embedding.traverse_face(a, b)
            key  = frozenset(face)
            if key not in seen:
                seen.add(key)
                faces.append(face)

    # The outer face is the one enclosing all others
    face_areas = [(f, _shoelace(G, f, scale)) for f in faces]
    outer_face = max(face_areas, key=lambda x: x[1])[0]
    outer_nodes = _clockwise(list(outer_face), G)

    bounded = [(f, a) for f, a in face_areas if f is not outer_face]
    bounded_faces = [f for f, _ in bounded]
    areas = [a for _, a in bounded]
    centers = [_centroid(G, f) for f in bounded_faces]

    return bounded_faces, outer_nodes, areas, centers

def node_to_face_edges(
        node: int,
        bounded_faces: List[List],
        outer_nodes: List,
) -> List[Tuple]:
    '''
    Collect all unique edges from faces that contain `node`
    
    Returns
    face_edges : List
        a list of (u, v) tuples.
    '''
    all_faces = bounded_faces + [outer_nodes]
    seen: Set[frozenset] = set()
    edges = []
    for face in all_faces:
        if node not in face:
            continue

        face_edges = {frozenset((face[i], face[(i + 1) % len(face)])) for i in range(len(face))}
        for e in face_edges:
            if e not in seen:
                seen.add(e)
                u, v = tuple(e)
                edges.append((u, v))
    return edges
