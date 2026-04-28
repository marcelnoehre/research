# {
#   "graph": { "directed": false },
#   "nodes": [
#     { "id": "a", "position": [0, 0] },
#     { "id": "b", "position": [100, 0] }
#   ],
#   "edges": [
#     { "id": "e0", "source": "a", "target": "b", "path": "M 0,0 L 100,0" }
#   ]
# }

from typing import Dict, Tuple
from data.parser import Parser
from fcapy.lattice import ConceptLattice

from geg import (
    aspect_ratio, angular_resolution, crossing_angle, edge_crossings,
    edge_length_deviation, edge_orthogonality, kruskal_stress,
    neighbourhood_preservation, node_edge_occlusion, node_resolution,
    node_uniformity,
)
import networkx as nx

def parse_to_geg(G: nx.Graph, coordinates: Dict[int, Tuple]):
    G = nx.transitive_reduction(G)

    for node_id, (x, y) in coordinates.items():
        G.nodes[node_id]['x'] = float(x)
        G.nodes[node_id]['y'] = float(y)

    for u, v in G.edges():
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        G.edges[u, v]['path'] = f"M {x1},{y1} L {x2},{y2}"

    G.graph['covering'] = list(G.edges())
    metrics = {
        "AR":  angular_resolution(G),
        "Asp": aspect_ratio(G),
        "CA":  crossing_angle(G),
        "EC":  edge_crossings(G),
        "EL":  edge_length_deviation(G),
        "EO":  edge_orthogonality(G),
        "KSM": kruskal_stress(G),
        "NP":  neighbourhood_preservation(G),
        "NEO": node_edge_occlusion(G),
        "NR":  node_resolution(G),
        "NU":  node_uniformity(G),
    }
    print(metrics)

if __name__ == '__main__':
    parser = Parser()
    cxt = parser.decode_cxt(f'./100.cxt')
    coordinates = {}
    with open('./100.pos', 'r') as f:
        for index, line in enumerate(f):
            parts = line.split()
            coordinates[index] = (float(parts[0]), float(parts[1]))

    lattice = ConceptLattice.from_context(cxt)
    parse_to_geg(lattice.to_networkx(), coordinates)
