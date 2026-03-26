import networkx as nx
import statistics

from typing import Dict, List
from shapely.geometry import LineString, box

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

    outer_set = set(outer_nodes)

    # split into left <-> right by x-coordinates
    all_xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    x_mid = statistics.median(all_xs)
    filtered_candidates: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        if node not in outer_set:
            filtered_candidates[node] = node_candidates
            continue
        if node == top_node:
            allowed = {'bottom', 'bottom_left', 'bottom_right'}
        elif node == bottom_node:
            allowed = {'top', 'top_left', 'top_right'}
        else:
            x_pos = G.nodes[node]['pos'][0]
            # left
            if x_pos <= x_mid:
                allowed = {'right', 'top_right', 'bottom_right'}
            #right
            else:
                allowed = {'left', 'top_left', 'bottom_left'}

        filtered_candidates[node] = [c for c in node_candidates if c.anchor in allowed]

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