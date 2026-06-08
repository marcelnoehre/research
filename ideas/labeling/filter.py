from collections import defaultdict
import numpy as np
import copy
from itertools import combinations
import networkx as nx

from typing import Dict, List
from fcapy.lattice import ConceptLattice
from shapely import Polygon
from shapely.geometry import LineString, Point, box

from label import LabelCandidate
from topology import _clockwise, node_to_face_edges

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
            padded_shape = inner_shape.buffer(0.01)
            if not any(padded_shape.intersects(line) for line in lines):
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
        #print(f'### {node} ###')
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

def filter_optimal_space(
        G: nx.Graph,
        candidates: Dict[int, List[LabelCandidate]],
        bounded_faces: List[List],
        outer_nodes: List
    ):
    filtered_candidates = copy.deepcopy(candidates)
    active_keys = {k for k, v in candidates.items() if len(v) > 1}

    all_polys = []
    for node_id, candidate_list in candidates.items():
        for candidate in candidate_list:
            poly = Polygon(candidate.bbox_corners)
            cid = id(candidate)
            all_polys.append((cid, node_id, poly))

    intersections = []
    for (cid1, fid1, p1), (cid2, fid2, p2) in combinations(all_polys, 2):
        if fid1 == fid2:
            continue
        if p1.intersects(p2) and p1.intersection(p2).area > 0:
            intersections.append(((cid1, fid1, p1), (cid2, fid2, p2)))

    intersecting_nodes = {fid for pair in intersections for (cid, fid, p) in pair}
    missing_keys = list(active_keys - intersecting_nodes)
    outside, inside = [k for k in missing_keys if k in outer_nodes], [k for k in missing_keys if k not in outer_nodes]

    normalized_faces = [
        [node if isinstance(node, tuple) else G.nodes[node]['pos'] for node in face]
        for face in bounded_faces
    ]

    def _get_nearest_active(start_idx, step, nodes, candidates, G):
        """
        Steps through the cycle to find the first node that has 
        either candidates or a fixed position in G.
        """
        n = len(nodes)
        curr = (start_idx + step) % n
        while curr != start_idx:
            node_id = nodes[curr]
            if node_id in candidates and candidates[node_id]:
                return Point(candidates[node_id][0].center)
            if node_id in G.nodes and 'pos' in G.nodes[node_id]:
                return Point(G.nodes[node_id]['pos'])
            curr = (curr + step) % n
        return None

    clockwise_nodes = _clockwise(outer_nodes, G)
    node_to_idx = {node_id: i for i, node_id in enumerate(clockwise_nodes)}
    face_polygons = [Polygon(face) for face in normalized_faces]

    for node in outside:
        idx = node_to_idx[node]
        pos_prev = _get_nearest_active(idx, -1, clockwise_nodes, candidates, G)
        pos_next = _get_nearest_active(idx, 1, clockwise_nodes, candidates, G)

        best_candidate = None
        min_orthogonal_dist = float('inf')

        if pos_prev and pos_next:
            # Convert to numpy arrays for vector math
            p1 = np.array([pos_prev.x, pos_prev.y])
            p2 = np.array([pos_next.x, pos_next.y])
            
            # The line segment connecting neighbors
            chord_vec = p2 - p1
            chord_len_sq = np.dot(chord_vec, chord_vec)
            
            for cand in candidates[node]:
                pt_vec = np.array([cand.center[0], cand.center[1]])
                
                # Find the projection of the candidate onto the chord
                # t is the position along the line (0.5 is perfectly centered)
                if chord_len_sq == 0:
                    t = 0
                else:
                    t = np.dot(pt_vec - p1, chord_vec) / chord_len_sq
                
                # 1. We want t to be near 0.5 (centered between neighbors)
                centering_error = abs(t - 0.5)
                
                projection = p1 + t * chord_vec
                dist_to_chord = np.linalg.norm(pt_vec - projection)
                
                score = centering_error - (0.1 * dist_to_chord) 
                
                if score < min_orthogonal_dist:
                    min_orthogonal_dist = score
                    best_candidate = cand

            if best_candidate:
                filtered_candidates[node] = [best_candidate]

    for node in inside:
        points = [Point(candidate.center) for candidate in candidates[node]]

        node_face_matches = [
            [i for i, poly in enumerate(face_polygons) if poly.contains(pt)]
            for pt in points
        ]
        all_matching_face_indices = [i for sublist in node_face_matches for i in sublist]

        max_face_idx = max(all_matching_face_indices, key=lambda i: face_polygons[i].area)

        tied_candidates = [
            (cand_idx, candidates[node][cand_idx]) 
            for cand_idx, face_indices in enumerate(node_face_matches)
            if max_face_idx in face_indices
        ]

        face_centroid = face_polygons[max_face_idx].centroid
        best_candidate = min(
            tied_candidates, 
            key=lambda item: Point(item[1].center).distance(face_centroid)
        )[1]
        filtered_candidates[node] = [best_candidate]

    return filtered_candidates
