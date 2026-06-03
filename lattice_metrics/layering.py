import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN

G = nx.read_graphml('graphs/dim_flux/convex_ordinal.graphml')
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

def lattice_width(G):
    assert nx.is_directed_acyclic_graph(G), "Graph must be a DAG."
    TC = nx.transitive_closure_dag(G)                # all comparabilities, not just cover edges
    B = nx.Graph()
    left  = {v: ('l', v) for v in G}
    right = {v: ('r', v) for v in G}
    B.add_nodes_from(left.values(),  bipartite=0)
    B.add_nodes_from(right.values(), bipartite=1)
    for u, v in TC.edges():
        B.add_edge(left[u], right[v])
    matching = nx.bipartite.maximum_matching(B, top_nodes=set(left.values()))
    matched = sum(1 for k in matching if k[0] == 'l')  # symmetric dict, count one side
    return G.number_of_nodes() - matched

purity_y = layer_purity_score(y_norm, max_chain_length)
purity_x = layer_purity_score(x_norm, max_chain_length)
score = purity_y * purity_x
width = lattice_width(G)

print(f"max_chain_length:   {max_chain_length}")
print(f"lattice_width:      {width}")
print(f"layer_purity_y:     {purity_y:.4f}")
print(f"layer_purity_x:     {purity_x:.4f}")
print(f"score:              {score:.4f}")