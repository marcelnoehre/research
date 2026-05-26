import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from shapely.geometry import LineString
from shapely.strtree import STRtree
import networkx as nx
import numpy as np

G = nx.read_graphml('graphs/hand_drawn/98.graphml')
_positions = {node: np.array([float(data['x']), float(data['y'])]) for node, data in G.nodes(data=True)}

def find_intersections(edges_list: List[Tuple], coords: Dict) -> List[Tuple]:
    lines = [LineString([coords[e[0]], coords[e[1]]]) for e in edges_list]
    tree = STRtree(lines)
    intersections = []
    for i, line in enumerate(lines):
        for j in tree.query(line, predicate='intersects'):
            if j <= i:
                continue
            if set(edges_list[i]) & set(edges_list[j]):
                continue
            pt = line.intersection(lines[j])
            if not pt.is_empty:
                intersections.append((i, j, pt))
    return intersections


def edge_crossing_score(G, positions):
    edges = list(G.edges())
    coords = {n: (positions[n][0], positions[n][1]) for n in G.nodes}

    # precompute reachability
    tc = nx.transitive_closure(G)

    def comparable(u, v):
        return tc.has_edge(u, v) or tc.has_edge(v, u) or u == v

    def can_cross(e1, e2):
        u1, v1 = e1
        u2, v2 = e2
        if set(e1) & set(e2):  # shared endpoint
            return False
        # edges can only cross if their endpoints are incomparable across edges
        return not comparable(u1, u2) and not comparable(u1, v2) \
           and not comparable(v1, u2) and not comparable(v1, v2)

    max_crossings = sum(
        1 for i in range(len(edges))
        for j in range(i + 1, len(edges))
        if can_cross(edges[i], edges[j])
    )

    crossings = len(find_intersections(edges, coords))
    return 1.0 - (crossings / max_crossings) if max_crossings > 0 else 1.0

score = edge_crossing_score(G, _positions)
print(f"score:              {score:.4f}")