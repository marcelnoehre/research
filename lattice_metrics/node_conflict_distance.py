import numpy as np
import networkx as nx
from itertools import combinations


# fraction of the average edge length below which two nodes are considered "too close"
THRESHOLD_FACTOR = 0.5

G = nx.read_graphml('graphs/hand_drawn/forum_romanum.graphml')

_positions = {node: np.array([float(data['x']), float(data['y'])]) for node, data in G.nodes(data=True)}
_tr = nx.transitive_reduction(G)

edge_lengths = [
    np.linalg.norm(_positions[i] - _positions[j])
    for (i, j) in _tr.edges
]
avg_edge_len = np.mean(edge_lengths)
threshold = avg_edge_len * THRESHOLD_FACTOR

node_conflict_distance = 0.0
n_conflicts = 0

for v, w in combinations(_positions, 2):
    dist = np.maximum(np.linalg.norm(_positions[v] - _positions[w]), 1e-3)
    if dist >= threshold:
        continue # far enough apart, not a conflict

    n_conflicts += 1
    # 0 at dist == threshold, grows like 1/dist^2 as dist -> 0
    node_conflict_distance += (1.0 / dist - 1.0 / threshold)**2

n_pairs = len(_positions) * (len(_positions) - 1) // 2
raw = (node_conflict_distance * avg_edge_len**2) / n_pairs
score = np.tanh(1.0 / (raw + 1e-9))
print(f"node_conflict_distance: {node_conflict_distance:.4f}")
print(f"n_conflicts:            {n_conflicts} / {n_pairs}")
print(f"avg_edge_len:           {avg_edge_len:.4f}")
print(f"threshold:              {threshold:.4f}")
print(f"raw:                    {raw:.4f}")
print(f'score:                  {score:.4f}')
