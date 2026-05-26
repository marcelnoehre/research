import numpy as np
import networkx as nx

G = nx.read_graphml('graphs/dim_flux/27.graphml')
_positions = {node: np.array([float(data['x']), float(data['y'])]) for node, data in G.nodes(data=True)}
max_chain_length = nx.dag_longest_path_length(G) + 1

def visual_chain_linearity_score(positions, G, max_chain_length):
    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks   = [n for n in G.nodes if G.out_degree(n) == 0]

    total, straight = 0, 0
    for src in sources:
        for snk in sinks:
            for path in nx.all_simple_paths(G, src, snk):
                if len(path) < 3:
                    continue
                for i in range(1, len(path) - 1):
                    a = positions[path[i - 1]]
                    b = positions[path[i]]
                    c = positions[path[i + 1]]

                    ac = c - a
                    length = np.linalg.norm(ac)
                    if length < 1e-9:
                        continue
                    ux, uy = -ac[1] / length, ac[0] / length
                    residual = abs((b - a) @ np.array([ux, uy]))

                    avg_gap = length / 2
                    total += 1
                    if residual / avg_gap < 0.05:
                        straight += 1

    raw = straight / total if total > 0 else 1.0
    # normalize by fraction of chain length vs total nodes
    # a longer max chain means more nodes can theoretically be collinear
    structural_max = (max_chain_length - 2) / (G.number_of_nodes() - 2) if G.number_of_nodes() > 2 else 1.0
    return min(raw / structural_max, 1.0)

linearity = visual_chain_linearity_score(_positions, G, max_chain_length)
print(f"visual_chain_linearity:     {linearity:.4f}")