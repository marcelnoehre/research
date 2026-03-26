import networkx as nx

from typing import List

def normalize_positions(G: nx.Graph, target_height: float = 10.0) -> float:
    '''
    Normalize all node 'pos' attributes in-place so that the graph's
    bounding box has a height of *target_height*, with the aspect ratio
    preserved and the origin placed at the bounding-box minimum.

    Parameters
    ----------
    G : nx.Graph
        graph whose nodes have a 'pos' attribute
    target_height : float
        desired height of the normalized coordinate space

    Returns
    -------
    mm_per_unit : float
        scale factor - multiply normalized coordinates by this to convert to mm
    '''
    xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    ys = [G.nodes[n]['pos'][1] for n in G.nodes]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    height = max_y - min_y
    if height == 0:
        raise ValueError('All nodes have the same y-coordinate!')

    scale = target_height / height

    for n in G.nodes:
        x, y = G.nodes[n]['pos']
        G.nodes[n]['pos'] = ((x - min_x) * scale, (y - min_y) * scale)

    G.graph['normalized_height'] = target_height
    G.graph['normalized_width']  = (max_x - min_x) * scale
    G.graph['norm_min_x']        = min_x
    G.graph['norm_min_y']        = min_y
    G.graph['norm_scale']        = scale

    return scale

def normalize_intersections(G: nx.Graph, intersections: List):
    '''
    Normalize intersection coordinates to match the normalized graph

    Parameters
    ----------
    TODO: description

    Returns
    -------
    TODO: description
    '''
    _s   = G.graph['norm_scale']
    _mx  = G.graph['norm_min_x']
    _my  = G.graph['norm_min_y']
    return [
        ((pt.coords[0][0] - _mx) * _s, (pt.coords[0][1] - _my) * _s)
        for _, _, pt in intersections
    ]