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
FONT_SIZE = matplotlib.rcParams.get('font.size', 10.0)
PHYSICAL_HEIGHT_MM = 100.0
DPI = 150.0

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
    fig, ax = plt.subplots(figsize=(8, 3), dpi=DPI)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.text(0.5, 0.5, text, 
            ha='center', va='center', fontsize=fontsize_pt, transform=ax.transAxes, 
            color='black', usetex=matplotlib.rcParams['text.usetex'])
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    ww, hh = canvas.get_width_height()
    buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(hh, ww, 4)
    mask = buf[:, :, 1:].min(axis=2) < 220
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    plt.close(fig)

    px_per_mm = DPI / 25.4
    return (cmax - cmin + 1) / px_per_mm, (rmax - rmin + 1) / px_per_mm

def compute_label_candidates(
        G: nx.Graph,
        concepts: List[int],
        label_text: str,
        label_type: str = 'extent',
        padding_x_mm: float = 1.0,
        padding_y_mm: float = 1.0,
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
    ink_w_mm, ink_h_mm = measure_ink_mm(label_text, FONT_SIZE)

    outer_w_mm = ink_w_mm + 2 * padding_x_mm
    outer_h_mm = ink_h_mm + 2 * padding_y_mm

    half_w  = (outer_w_mm * units_per_mm) / 2.0
    half_h  = (outer_h_mm * units_per_mm) / 2.0

    # Inner half-extents (ink only, no padding)
    half_iw = (ink_w_mm  * units_per_mm) / 2.0
    half_ih = (ink_h_mm  * units_per_mm) / 2.0

    anchor_offsets: Dict[str, Tuple[float, float]] = {
        'top': ( 0.0, -half_h),
        'left': ( half_w, 0.0),
        'bottom': ( 0.0, half_h),
        'right': (-half_w, 0.0),
        'top_left': ( half_w, -half_h),
        'top_right': (-half_w, -half_h),
        'bottom_left': ( half_w, half_h),
        'bottom_right': (-half_w, half_h)
    }

    # Expansion = padding in each free direction, doubling the total gap
    exp_x = (padding_x_mm * units_per_mm)
    exp_y = (padding_y_mm * units_per_mm)

    _expand_factors: Dict[str, Tuple[float, float, float, float]] = {
        'top': (exp_x, exp_x, exp_y, 0),
        'bottom': (exp_x, exp_x, 0, exp_y),
        'left': (0, exp_x, exp_y, exp_y),
        'right': (exp_x, 0, exp_y, exp_y),
        'bottom_right': (exp_x, 0, 0, exp_y),
        'bottom_left': (0, exp_x, 0, exp_y),
        'top_right': (exp_x, 0, exp_y, 0),
        'top_left': (0, exp_x, exp_y, 0)
    }

    candidates: Dict[int, List[LabelCandidate]] = {}

    for node in concepts:
        node_x, node_y = G.nodes[node]['pos']

        node_candidates: List[LabelCandidate] = []
        for anchor, (dx, dy) in anchor_offsets.items():
            if anchor not in allowed_anchors:
                continue
            cx = node_x + dx
            cy = node_y + dy

            bl = (cx - half_w,  cy - half_h)
            br = (cx + half_w,  cy - half_h)
            tr = (cx + half_w,  cy + half_h)
            tl = (cx - half_w,  cy + half_h)

            ibl = (cx - half_iw, cy - half_ih)
            ibr = (cx + half_iw, cy - half_ih)
            itr = (cx + half_iw, cy + half_ih)
            itl = (cx - half_iw, cy + half_ih)

            dl, dr, db, dt = _expand_factors[anchor]
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
    label_type: str
    bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple] # BL, BR, TR, TL
    inner_bbox_corners: Tuple[Tuple, Tuple, Tuple, Tuple] # Ink only
    center: Tuple[float, float]
    anchor: str = 'overflow'
    text: str = ''

def compute_overflow_label(
    G: nx.Graph,
    node: int,
    label_text: str,
    label_type: str,
    padding_x_mm: float = 1.0,
    padding_y_mm: float = 1.0,
) -> OverflowLabel:
    '''
    Creates a centered overflow label for a specific node.
    '''
    if 'normalized_height' not in G.graph:
        raise ValueError("Call normalize_positions(G) before compute.")

    mm_per_unit = PHYSICAL_HEIGHT_MM / G.graph['normalized_height']
    units_per_mm = 1.0 / mm_per_unit

    # 1. Measure Ink
    ink_w_mm, ink_h_mm = measure_ink_mm(label_text, FONT_SIZE)

    # 2. Calculate Dimensions in Graph Units
    half_w = ((ink_w_mm + 2 * padding_x_mm) * units_per_mm) / 2.0
    half_h = ((ink_h_mm + 2 * padding_y_mm) * units_per_mm) / 2.0
    half_iw = (ink_w_mm * units_per_mm) / 2.0
    half_ih = (ink_h_mm * units_per_mm) / 2.0

    # 3. Position (Defaulting to center of node)
    cx, cy = G.nodes[node]['pos']

    # 4. Construct Bounding Boxes
    bl, br = (cx - half_w, cy - half_h), (cx + half_w, cy - half_h)
    tr, tl = (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)

    ibl, ibr = (cx - half_iw, cy - half_ih), (cx + half_iw, cy - half_ih)
    itr, itl = (cx + half_iw, cy + half_ih), (cx - half_iw, cy + half_ih)

    return OverflowLabel(
        label_type=label_type,
        bbox_corners=(bl, br, tr, tl),
        inner_bbox_corners=(ibl, ibr, itr, itl),
        center=(cx, cy),
        text=label_text
    )