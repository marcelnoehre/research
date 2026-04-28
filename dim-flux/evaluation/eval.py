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

mirror = {
    'hand_drawn': [],
    'sup_inf_attribute': [3, 7, 10, 12, 15, 28, 32, 35, 36, 46, 48, 51, 55, 56, 57, 62, 68, 88, 92, 102, 106, 107, 'Forum-Romanum', 'living_beings_and_water', 'triangles'],
    'sup_inf_double': [7, 3, 10, 14, 16, 17, 18, 20, 25, 29, 31, 32, 33, 34, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 55, 56, 57, 58, 59, 61, 62, 64, 68, 71, 72, 80, 81, 82, 85, 86, 87, 88, 94, 96, 99, 100, 101, 102, 106, 107, 110, 112, 113, 114, 115, 117, 118, 119, 121, 123],
    'dim_draw': [7, 10, 12, 13, 15, 16, 29, 45, 46, 47, 53, 55, 56, 61, 62, 63, 64, 65, 70, 71, 72, 77, 79, 81, 86, 87, 88, 94, 96, 101, 104, 106, 116, 'car'],
    'dim_draw_double': [7, 10, 12, 13, 15, 16, 29, 45, 46, 47, 53, 55, 56, 61, 62, 63, 64, 65, 70, 71, 72, 77, 79, 81, 86, 87, 88, 94, 96, 101, 104, 106, 116, 'car']
}

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

def scaled_positions(cxt, type, size, coords):
    x_vals = [p[0] for p in coords.values()]
    y_vals = [p[1] for p in coords.values()]
    
    if isinstance(cxt, int) and cxt > 1:
        min_x, max_x = min(x_vals), max(x_vals)
        min_y_orig = min(y_vals)
        
        range_x = max_x - min_x
        scale = size / range_x if range_x != 0 else 1.0
        
        scaled_coords = [
            (-(size * 0.5) + (x - min_x) * scale, (y - min_y_orig) * scale)
            for x, y in coords.values()
        ]
    else:
        scaled_coords = coords.values()

    y_vals_scaled = [p[1] for p in scaled_coords]
    min_y_scaled, max_y_scaled = min(y_vals_scaled), max(y_vals_scaled)
    height = (max_y_scaled - min_y_scaled)

    if height > (size * 0.25):
        # Determine if mirroring is required for this context
        mirror_factor = -1.0 if cxt in mirror.get(type, []) else 1.0
        shrink_factor = (size * 0.25) / height
        
        processed_coords = [
            (x * shrink_factor * mirror_factor, (y - min_y_scaled) * shrink_factor)
            for x, y in scaled_coords
        ]
    else:
        # Simple normalization to Y=0
        processed_coords = [
            (x, (y - min_y_scaled))
            for x, y in scaled_coords
        ]

    final_min_y = min(p[1] for p in processed_coords)
    final_coords = {
        i: (p[0], p[1] - final_min_y)
        for i, p in enumerate(processed_coords)
    }

    return final_coords

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

            coordinates = scaled_positions(i, algo, 10, coordinates)

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