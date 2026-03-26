import networkx as nx
import statistics

from typing import Dict, List

from label import LabelCandidate

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
    filtered_candidates = {}

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