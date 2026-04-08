import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

from dataclasses import dataclass
from typing import Dict, List, Tuple

MAX_ROW_CHARS = 10
K_ROWS = 2
PHYSICAL_HEIGHT_MM = 100.0
DPI = 150.0

def compute_suggested_font_size(G: nx.Graph, base_font_size: float = 12.0) -> float:
    '''
    Calculates a font size that scales down if the lattice is narrow.
    '''
    xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    w_span = max(xs) - min(xs) if xs else 1.0
    reference_width = 100.0 
    scale_factor = max(0.5, min(np.sqrt(w_span / reference_width), 1.2))
    return base_font_size * scale_factor

@dataclass
class LabelCandidate:
    """
    One candidate placement for a node label.

    Parameters
    ----------
    anchor : str
        which corner or side of the outer bbox is placed at the node position
    label_type : str
        'general', 'extent' (objects, below node) or 'intent' (attributes, above node)
    bbox_corners : Tuple[Tuple, Tuple, Tuple, Tuple]
        outer bbox (BL, BR, TR, TL) in graph units
    inner_bbox_corners : Tuple[Tuple, Tuple, Tuple, Tuple]
        inner (ink) bbox (BL, BR, TR, TL) in graph units
    expanded_bbox_corners : Tuple[Tuple, Tuple, Tuple, Tuple]
        outer bbox expanded on the two free sides for node exclusion check (BL, BR, TR, TL) in graph units
    center : Tuple[float, float]
        center of the outer bbox in graph units
    """
    anchor: str # 'top' | 'left' | 'bottom' | 'right' | 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right'
    label_type: str # 'general' | 'extent' | 'intent'
    bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]
    inner_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]
    expanded_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple]
    center: Tuple[float, float]

def wrap_label_text(plain_text: str, formatter=str) -> str:
    '''
    '''
    # split into words
    words = plain_text.split()

    # forced to one row or smaller than split size
    if K_ROWS == 1 or len(plain_text) <= MAX_ROW_CHARS or len(words) == 1:
        return formatter(plain_text)

    # split into k rows
    # TODO: more than 2 rows
    mid = len(plain_text) / 2
    best_split = 1
    best_diff = float('inf')
    for i in range(1, len(words)):
        row1 = ' '.join(words[:i])
        diff = abs(len(row1) - mid)
        if diff < best_diff:
            best_diff = diff
            best_split = i

    # format
    row1 = formatter(' '.join(words[:best_split]))
    row2 = formatter(' '.join(words[best_split:]))
    return rf'\begin{{tabular}}{{@{{}}c@{{}}}}{row1} \\[-4pt] {row2}\end{{tabular}}'

def measure_ink_mm(text: str, fontsize_pt: float) -> tuple[float, float]:
    '''
    Return (ink_width_mm, ink_height_mm) by rasterising at 150 DPI.
    '''
    fig = plt.figure(dpi=DPI)
    ax = fig.add_subplot(111)
    t = ax.text(0.5, 0.5, text, fontsize=fontsize_pt, usetex=matplotlib.rcParams['text.usetex'])

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()

    bbox = t.get_window_extent(renderer=renderer)
    width_px = bbox.width
    height_px = bbox.height

    plt.close(fig)

    px_per_mm = DPI / 25.4
    return width_px / px_per_mm, height_px / px_per_mm

def compute_label_candidates(
        G: nx.Graph,
        concepts: List[int],
        label_text: str,
        label_type: str = 'extent'
) -> Dict[int, List[LabelCandidate]]:
    '''
    For every concept node, return candidate label placements.

    Parameters
    ----------
    G                  : graph with normalised 'pos' attributes
    concepts           : list of concept node ids
    label_text         : the label string (used to measure ink size)
    physical_height_mm : actual drawing height in mm (sets the unit scale)
    label_type         : 'extent' only top anchors, 'intent' only bottom anchors
    padding_x_mm       : horizontal padding around the ink box
    padding_y_mm       : vertical padding around the ink box
    fontsize_pt        : font size in points

    Returns
    -------
    dict mapping concept node id → list of LabelCandidate objects
    '''
    if 'normalized_height' not in G.graph:
        raise ValueError("Call normalize_positions(G) before compute_label_candidates().")
    
    # Restrict anchors by label type
    if label_type == 'general':
        allowed_anchors = {'top', 'left', 'bottom', 'right', 'top_left', 'top_right', 'bottom_left', 'bottom_right'}
    elif label_type == 'extent':
        allowed_anchors = {'top', 'top_left', 'top_right'} # objects below node, anchor at top
    elif label_type == 'intent':
        allowed_anchors = {'bottom', 'bottom_left', 'bottom_right'} # attributes above node, anchor at bottom
    else:
        raise ValueError(f"invalid label_type must be 'general', 'extent' or 'intent', got '{label_type}'")

    mm_per_unit = PHYSICAL_HEIGHT_MM / G.graph['normalized_height']
    units_per_mm = 1.0 / mm_per_unit

    # Measure label in mm, convert to graph units
    font_size = compute_suggested_font_size(G)
    ink_w_mm, ink_h_mm = measure_ink_mm(label_text, font_size)

    # padding
    padding = ink_h_mm

    # Inner half-extents (ink only, no padding)
    half_iw = (ink_w_mm  * units_per_mm) / 2.0
    half_ih = (ink_h_mm  * units_per_mm) / 2.0

    corner_multiplier = 1 / np.sqrt(2)
    anchor_padding_scales = {
        'top': 1.0, 'bottom': 1.0, 'left': 1.0, 'right': 1.0,
        'top_left': corner_multiplier, 
        'top_right': corner_multiplier,
        'bottom_left': corner_multiplier, 
        'bottom_right': corner_multiplier
    }
    anchor_extents: Dict[str, Tuple[float, float]] = {}

    for anchor in allowed_anchors:
        scale = anchor_padding_scales[anchor]
        
        # Scale the padding specifically for this anchor type
        p_x = padding * scale
        p_y = padding * scale
        
        # Calculate outer dimensions with dampened padding
        o_w_units = (ink_w_mm + 2 * p_x) * units_per_mm
        o_h_units = (ink_h_mm + 2 * p_y) * units_per_mm
        
        anchor_extents[anchor] = (o_w_units / 2.0, o_h_units / 2.0)

    anchor_offsets: Dict[str, Tuple[float, float]] = {
        'top':          (0.0, -anchor_extents['top'][1]),
        'bottom':       (0.0,  anchor_extents['bottom'][1]),
        'left':         ( anchor_extents['left'][0], 0.0),
        'right':        (-anchor_extents['right'][0], 0.0),
        
        'top_left':     ( anchor_extents['top_left'][0],     -anchor_extents['top_left'][1]),
        'top_right':    (-anchor_extents['top_right'][0],    -anchor_extents['top_right'][1]),
        'bottom_left':  ( anchor_extents['bottom_left'][0],   anchor_extents['bottom_left'][1]),
        'bottom_right': (-anchor_extents['bottom_right'][0],  anchor_extents['bottom_right'][1])
    }

    candidates: Dict[int, List[LabelCandidate]] = {}

    for node in concepts:
        node_x, node_y = G.nodes[node]['pos']
        node_candidates: List[LabelCandidate] = []
        
        for anchor, (dx, dy) in anchor_offsets.items():
            if anchor not in allowed_anchors:
                continue
            
            current_half_w, current_half_h = anchor_extents[anchor]
            
            cx = node_x + dx
            cy = node_y + dy

            # Now the bbox corners match the adjusted padding
            bl = (cx - current_half_w,  cy - current_half_h)
            br = (cx + current_half_w,  cy - current_half_h)
            tr = (cx + current_half_w,  cy + current_half_h)
            tl = (cx - current_half_w,  cy + current_half_h)

            # Inner ink stays the same (half_iw/half_ih are global)
            ibl = (cx - half_iw, cy - half_ih)
            ibr = (cx + half_iw, cy - half_ih)
            itr = (cx + half_iw, cy + half_ih)
            itl = (cx - half_iw, cy + half_ih)

            # --- FIX 2: Recalculate expansion factors based on the current anchor's padding ---
            # This ensures the 'expanded' collision box shrinks along with the visual box
            current_scale = anchor_padding_scales[anchor]
            cur_exp_x = (padding * current_scale * units_per_mm)
            cur_exp_y = (padding * current_scale * units_per_mm)

            # Map the local expansion factors
            _cur_expand = {
                'top': (cur_exp_x, cur_exp_x, cur_exp_y, 0),
                'bottom': (cur_exp_x, cur_exp_x, 0, cur_exp_y),
                'left': (0, cur_exp_x, cur_exp_y, cur_exp_y),
                'right': (cur_exp_x, 0, cur_exp_y, cur_exp_y),
                'bottom_right': (cur_exp_x, 0, 0, cur_exp_y),
                'bottom_left': (0, cur_exp_x, 0, cur_exp_y),
                'top_right': (cur_exp_x, 0, cur_exp_y, 0),
                'top_left': (0, cur_exp_x, cur_exp_y, 0)
            }
            
            dl, dr, db, dt = _cur_expand[anchor]
            ebl = (bl[0] - dl, bl[1] - db)
            etr = (tr[0] + dr, tr[1] + dt)
            ebr = (etr[0], ebl[1])
            etl = (ebl[0], etr[1])

            node_candidates.append(LabelCandidate(
                anchor=anchor,
                label_type=label_type,
                bbox_corners=(bl, br, tr, tl),
                inner_bbox_corners=(ibl, ibr, itr, itl),
                expanded_bbox_corners=(ebl, ebr, etr, etl),
                center=(cx, cy),
            ))

        candidates[node] = node_candidates

    return candidates

@dataclass
class OverflowLabel:
    """
    A label for nodes that exceed standard placement constraints.
    Defaults to an overflow anchor and ignores expanded bboxes for 
    node exclusion checks.
    """
    node_id: int
    label_type: str
    bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple] # BL, BR, TR, TL
    inner_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple] # Ink only
    expanded_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple] # Full visual bbox (padding)
    center: Tuple[float, float]
    anchor: str = 'overflow'
    text: str = ''

def compute_overflow_label(
    G: nx.Graph,
    node: int,
    label_text: str,
    label_type: str,
    padding_x_mm: float = 1.5,
    padding_y_mm: float = 1.5,
    scaling_intensity: float = 0.25
) -> OverflowLabel:
    '''
    Creates a centered overflow label for a specific node.
    bbox_corners        — tight bbox (tight_padding_mm), used for collision
    inner_bbox_corners  — ink only, no padding
    expanded_bbox_corners — full visual bbox (padding_x/y_mm), used for rendering
    '''
    if 'normalized_height' not in G.graph:
        raise ValueError("Call normalize_positions(G) before compute.")

    mm_per_unit = PHYSICAL_HEIGHT_MM / G.graph['normalized_height']
    units_per_mm = 1.0 / mm_per_unit

    font_size = compute_suggested_font_size(G)
    ink_w_mm, ink_h_mm = measure_ink_mm(label_text, font_size)

    # padding
    padding = ink_h_mm

    # Ink only
    half_iw = (ink_w_mm * units_per_mm) / 2.0
    half_ih = (ink_h_mm * units_per_mm) / 2.0

    # Tight bbox (collision)
    half_w  = ((ink_w_mm + 2 * padding / 3) * units_per_mm) / 2.0
    half_h  = ((ink_h_mm + 2 * padding / 3) * units_per_mm) / 2.0

    # Extended bbox (visual)
    half_ew = ((ink_w_mm + 2 * padding) * units_per_mm) / 2.0
    half_eh = ((ink_h_mm + 2 * padding) * units_per_mm) / 2.0

    cx, cy = G.nodes[node]['pos']

    def corners(hx, hy):
        return (
            (cx - hx, cy - hy),  # BL
            (cx + hx, cy - hy),  # BR
            (cx + hx, cy + hy),  # TR
            (cx - hx, cy + hy),  # TL
        )

    return OverflowLabel(
        node_id=node,
        label_type=label_type,
        bbox_corners=corners(half_w, half_h),
        inner_bbox_corners=corners(half_iw, half_ih),
        expanded_bbox_corners=corners(half_ew, half_eh),
        center=(cx, cy),
        text=label_text,
    )