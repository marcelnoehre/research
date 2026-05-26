import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from shapely.geometry import LineString
from shapely.strtree import STRtree
import networkx as nx
import numpy as np

G = nx.read_graphml('graphs/sup_inf_attribute/drive_concepts.graphml')
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

def crossing_angle_score(G, positions):
    edges = list(G.edges())
    coords = {n: (positions[n][0], positions[n][1]) for n in G.nodes}

    intersections = find_intersections(edges, coords)
    if not intersections:
        return 1.0

    deviations = []
    for i, j, pt in intersections:
        d1 = positions[edges[i][1]] - positions[edges[i][0]]
        d2 = positions[edges[j][1]] - positions[edges[j][0]]

        cos_angle = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_angle)))

        deviations.append((abs(90.0 - angle) / 90.0))

    return 1.0 - np.mean(deviations)

score = crossing_angle_score(G, _positions)

print(f"score:              {score:.4f}")