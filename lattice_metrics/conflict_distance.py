import numpy as np
import networkx as nx


G = nx.read_graphml('graphs/hand_drawn/convex_ordinal.graphml')

_positions = {node: np.array([float(data['x']), float(data['y'])]) for node, data in G.nodes(data=True)}
_tr = nx.transitive_reduction(G)

conflict_distance = 0.0

for v, w in _positions.items():
        
    for (i, j) in _tr.edges:
        if v == i or v == j:
            continue # incident edge

        # v_1 below v_2
        if _positions[i][1] > _positions[j][1]:
            w_1, w_2 = _positions[j], _positions[i]
        else:
            w_1, w_2 = _positions[i], _positions[j]

        # v below v_1
        if np.dot(w_1 - w, w_2 - w_1) > 0:
            # |w_1 - w|
            dist = np.linalg.norm(w_1 - w)
        # v above v_2
        elif np.dot(w_2 - w, w_2 - w_1) < 0:
            # |w_2 - w|
            dist = np.linalg.norm(w_2 - w)
        # v above v_1 and below v_2 
        else:
            # perpendicular distance
            A = np.abs((w_1[0] - w[0]) * (w_2[1] - w[1]) - (w_1[1] - w[1]) * (w_2[0] - w[0]))
            # A = np.abs(np.cross(w_1 - w, w_2 - w))
            f = w_2 - w_1
            dist = np.maximum(A / np.linalg.norm(f), 1e-3)

        conflict_distance += 1.0 / dist**2

print(conflict_distance)