import copy
import networkx as nx

from typing import Dict, List
from fcapy.lattice import ConceptLattice
from shapely.geometry import LineString, Point, box

from label import LabelCandidate
from topology import node_to_face_edges

def restrict_outer_node_candidates(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        outer_nodes: List,
        top_node: int,
        bottom_node: int,
) -> Dict[int, List[LabelCandidate]]:
    '''
    '''
    if not outer_nodes:
        raise ValueError('Received no outer nodes!')

    filtered_candidates = copy.deepcopy(candidates)
    left = False

    for node in outer_nodes:
        if node == top_node:
            allowed = {'bottom'}
        elif node == bottom_node:
            allowed = {'top'}
            left = True
        else:
            allowed = {'right', 'top_right', 'bottom_right'} if left else {'left', 'top_left', 'bottom_left'}

        filtered_candidates[node] = [c for c in filtered_candidates[node] if c.anchor in allowed]

    return filtered_candidates

def filter_candidates_by_nodes(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        concepts: List[int],
) -> Dict[int, List[LabelCandidate]]:
    filtered_candidates: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        surviving = []
        for candidate in node_candidates:
            (bl_x, bl_y), _, (tr_x, tr_y), _ = candidate.expanded_bbox_corners
            
            occupied = False
            for other in concepts:
                if other == node:
                    continue
                ox, oy = G.nodes[other]['pos']
                
                if (bl_x <= ox <= tr_x) and (bl_y <= oy <= tr_y):
                    occupied = True
                    break
            
            if not occupied:
                surviving.append(candidate)
                
        filtered_candidates[node] = surviving
    return filtered_candidates

def filter_candidates_by_edges(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        bounded_faces: List[List],
        outer_nodes: List
) -> Dict[int, List[LabelCandidate]]:
    '''
    '''
    filtered_candidates: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        incident_edges = node_to_face_edges(node, bounded_faces, outer_nodes)
        lines = [
            LineString([G.nodes[u]['pos'], G.nodes[v]['pos']])
            for u, v in incident_edges
        ]

        surviving = []
        for candidate in node_candidates:
            ibl, _, itr, _ = candidate.inner_bbox_corners
            inner_shape = box(ibl[0], ibl[1], itr[0], itr[1])
            if not any(inner_shape.intersects(line) for line in lines):
                surviving.append(candidate)

        filtered_candidates[node] = surviving

    return filtered_candidates

def filter_candidates_by_neighbor_direction(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        lattice: ConceptLattice
) -> Dict[int, List[LabelCandidate]]:
    '''
    '''
    filtered_candidates = copy.deepcopy(candidates)

    for node, node_candidates in candidates.items():
        print(f'### {node} ###')
        if not node_candidates:
            filtered_candidates[node] = node_candidates
            continue
        if len(node_candidates) == 1:
            filtered_candidates[node] = node_candidates
            continue

        node_x = G.nodes[node]['pos'][0]
        node_y = G.nodes[node]['pos'][1]

        neighbors = list(lattice.children(node)) + list(lattice.parents(node))

        has_top_left = any((G.nodes[nb]['pos'][0] < node_x and G.nodes[nb]['pos'][1] > node_y) for nb in neighbors)
        has_bottom_left = any((G.nodes[nb]['pos'][0] < node_x and G.nodes[nb]['pos'][1] < node_y) for nb in neighbors)
        has_top_right = any((G.nodes[nb]['pos'][0] > node_x and G.nodes[nb]['pos'][1] > node_y) for nb in neighbors)
        has_bottom_right = any((G.nodes[nb]['pos'][0] > node_x and G.nodes[nb]['pos'][1] < node_y) for nb in neighbors)

        filter = []
        if has_top_left:
            filter.append('bottom')
            filter.append('right')
            filter.append('bottom_right')
        if has_top_right:
            filter.append('bottom')
            filter.append('left')
            filter.append('bottom_left')
        if has_bottom_left:
            filter.append('top')
            filter.append('right')
            filter.append('top_right')
        if has_bottom_right:
            filter.append('top')
            filter.append('left')
            filter.append('top_left')

        by_type: Dict[str, List[LabelCandidate]] = {}
        for c in node_candidates:
            by_type.setdefault(c.label_type, []).append(c)

        for group in by_type.values():
            filtered = [c.anchor for c in group if c.anchor in filter]

            # do not filter all
            if len(filtered) == len(group):
                continue
            filtered_candidates[node] = [c for c in filtered_candidates[node] if c.anchor not in filtered]
    
    return filtered_candidates
