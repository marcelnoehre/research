import csv
import networkx as nx
from typing import Dict, Tuple
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice
from geg import (
    aspect_ratio, angular_resolution, crossing_angle, edge_crossings,
    edge_length_deviation, edge_orthogonality, kruskal_stress,
    neighbourhood_preservation, node_edge_occlusion, node_resolution,
    node_uniformity,
)

algorithms = ['hand_drawn', 'sup_inf_attribute', 'sup_inf_double', 'dim_draw', 'dim_draw_double']
algorithms_label = {
    'hand_drawn': 'hand_drawn', 
    'sup_inf_attribute': 'sup_inf_attribute', 
    'sup_inf_double': 'sup_inf_doubly', 
    'dim_draw': 'dim_draw', 
    'dim_draw_double': 'dim_flux'
}
path = {
    'hand_drawn': 'm4', 
    'sup_inf_attribute': 'm4_original', 
    'sup_inf_double': 'm4', 
    'dim_draw': 'm4_dim_draw', 
    'dim_draw_double': 'm4_dim_draw'  
}
metric_labels = ["AR","Asp","CA","EC","EL","EO","KSM","NP","NEO","NR","NU"]

def decode_cxt(cxt: str) -> FormalContext:
        '''
        Decode a Burmeister (B) string into a Formal Context.

        The string starts with a B, followed by the dimension of the context and the incidence matrix.

        'x' or 'X' indicates that a object (row) has a feature (column), while a any other character
        indicates that a object does not have a feature. 

        Parameters
        ----------
        cxt : str
            A string representing the burmeister format or a path to the .cxt file

        Returns
        -------
        formal_context : FormalContext
            The formal context.
        '''
        if cxt.endswith('.cxt'):
            with open(cxt, 'r') as f:
                cxt = f.read()

        _, ns, cxt = cxt.split('\n\n')
        n_objs, n_attrs = [int(x) for x in ns.split('\n')]

        cxt = cxt.strip().split('\n')
        obj_names, cxt = cxt[:n_objs], cxt[n_objs:]
        attr_names, cxt = cxt[:n_attrs], cxt[n_attrs:]
        cxt = [[(c == 'X' or c == 'x') for c in line] for line in cxt]

        return FormalContext(data=cxt, object_names=obj_names, attribute_names=attr_names)

def compute_metrics(G: nx.Graph, coordinates: Dict[int, Tuple]):
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
    return metrics

output_file = 'lattice_metrics_results.csv'
with open(output_file, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(['lattice', 'algo'] + metric_labels)

    for i in range(1, 127):
        for algo in algorithms:
            cxt = decode_cxt(f'data/{path[algo]}/{i}.cxt')
            coordinates = {}
            with open(f'positions/{algo}/{i}.pos', 'r') as f:
                for index, line in enumerate(f):
                    parts = line.split()
                    coordinates[index] = (float(parts[0]), float(parts[1]))

            lattice = ConceptLattice.from_context(cxt)
            metrics = compute_metrics(lattice.to_networkx(), coordinates)
            row = [i, algo] + [metrics[label] for label in metric_labels]
            writer.writerow(row)
            f_out.flush()

    for name in ['Forum-Romanum', 'living_beings_and_water', 'car', 'triangles', 'convex-ordinal']:
        for algo in algorithms:
            cxt = decode_cxt(f'data/study_reduced/{name}.cxt')
            coordinates = {}
            with open(f'positions/{algo}/{name}.pos', 'r') as f:
                for index, line in enumerate(f):
                    parts = line.split()
                    coordinates[index] = (float(parts[0]), float(parts[1]))

            lattice = ConceptLattice.from_context(cxt)
            metrics = compute_metrics(lattice.to_networkx(), coordinates)
            row = [name, algo] + [metrics[label] for label in metric_labels]
            writer.writerow(row)
            f_out.flush()