import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN

G = nx.read_graphml('graphs/sup_inf_doubly/convex_ordinal.graphml')
_positions = {node: np.array([float(data['x']), float(data['y'])]) for node, data in G.nodes(data=True)}

# Normalize x and y to [0, 1]
coords = np.array(list(_positions.values()))
x_coords = coords[:, 0]
y_coords = coords[:, 1]

def normalize(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo)

x_norm = normalize(x_coords)
y_norm = normalize(y_coords)

max_chain_length = nx.dag_longest_path_length(G) + 1

def layer_purity_score(norm_coords, n_layers):
    eps = 0.05
    labels = DBSCAN(eps=eps, min_samples=1).fit(norm_coords.reshape(-1, 1)).labels_
    avg_inter_layer_gap = 1.0 / (n_layers - 1) if n_layers > 1 else 1.0
    cluster_stds = [np.std(norm_coords[labels == label]) for label in set(labels)]
    mean_std = np.mean(cluster_stds)
    return 1.0 - np.clip(mean_std / avg_inter_layer_gap, 0.0, 1.0)

purity_y = layer_purity_score(y_norm, max_chain_length)
purity_x = layer_purity_score(x_norm, max_chain_length)

score = purity_y * purity_x

print(f"max_chain_length:   {max_chain_length}")
print(f"layer_purity_y:     {purity_y:.4f}")
print(f"layer_purity_x:     {purity_x:.4f}")
print(f"score:              {score:.4f}")