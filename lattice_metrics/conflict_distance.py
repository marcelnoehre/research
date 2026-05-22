import numpy as np
import networkx as nx


G = nx.read_graphml('graphs/dim_flux/forum_romanum.graphml')

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

edge_lengths = [
    np.linalg.norm(_positions[i] - _positions[j])
    for (i, j) in _tr.edges
]
avg_edge_len = np.mean(edge_lengths)
conflict_distance_norm = conflict_distance * avg_edge_len**2
n_pairs = sum(
    1 for v in _positions
    for (i, j) in _tr.edges
    if v != i and v != j
)

raw = conflict_distance_norm / n_pairs
score = np.tanh(1.0 / (raw + 1e-9))
print(f"conflict_distance: {conflict_distance:.4f}")
print(f"avg_edge_len:      {avg_edge_len:.4f}")
print(f"raw:               {raw:.4f}")
print(f'score:             {score:.4f}')