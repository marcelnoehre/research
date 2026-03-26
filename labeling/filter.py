import copy
import networkx as nx

from typing import Dict, List
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
            allowed = {'bottom', 'bottom_left', 'bottom_right'}
        elif node == bottom_node:
            allowed = {'top', 'top_left', 'top_right'}
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
    '''
    '''
    filtered_candidates: Dict[int, List[LabelCandidate]] = {}

    for node, node_candidates in candidates.items():
        surviving = []
        for candidate in node_candidates:
            ebl, _, etr, _ = candidate.expanded_bbox_corners
            expanded = box(ebl[0], ebl[1], etr[0], etr[1])
            occupied = any(
                expanded.contains(Point(G.nodes[other]['pos']))
                for other in concepts
                if other != node and other in G.nodes
            )
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