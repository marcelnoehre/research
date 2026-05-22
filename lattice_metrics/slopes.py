from sklearn.cluster import DBSCAN
from math import gcd
from itertools import combinations
import numpy as np
import networkx as nx

def slope_harmony_score(G, positions, angle_eps=5.0):
    node_index = {n: i for i, n in enumerate(G.nodes())}
    coords = np.array([positions[n] for n in G.nodes()])

    angles = []
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        if dx == 0 and dy == 0:
            continue
        angles.append(np.degrees(np.arctan2(abs(dy), abs(dx))))

    if not angles:
        return 0.0

    angles = np.array(angles)
    n_edges = len(angles)

    labels = DBSCAN(eps=angle_eps, min_samples=1).fit(angles.reshape(-1, 1)).labels_
    n_clusters = len(set(labels))

    # 1 cluster → score 1.0, n_edges clusters (all different) → score 0.0
    score = 1.0 - (n_clusters - 1) / (n_edges - 1)
    return score

G = nx.read_graphml('graphs/sup_inf_attribute/15.graphml')
_positions = {node: np.array([float(data['x']), float(data['y'])]) for node, data in G.nodes(data=True)}

score = slope_harmony_score(G, _positions)
print(score)