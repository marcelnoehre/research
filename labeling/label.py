import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from TexSoup import TexSoup
import re

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"
})

from dataclasses import dataclass
from typing import Dict, List, Tuple

MAX_ROW_CHARS = 10
K_ROWS = 2
PHYSICAL_HEIGHT_MM = 100.0
DPI = 150.0
FONT_SIZE = 8.0

def compute_suggested_font_size(G: nx.Graph, base_font_size: float = FONT_SIZE) -> float:
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

def wrap_label_text(plain_text: str, type: str, formatter=str, k_rows: int = K_ROWS, max_row_chars: int = MAX_ROW_CHARS) -> str:
    '''
    Wraps label text into at most k_rows rows, minimizing variance in row lengths.
    Works with both plain text and LaTeX source (uses TexSoup for word-boundary detection).
    '''
    if not plain_text:
        return plain_text
    
    words = _split_words(plain_text)

    # Forced to one row or too short to bother splitting
    if k_rows == 1 or len(plain_text) <= max_row_chars or len(words) <= 1:
        return format_row(formatter(plain_text), type)

    best_splits = _find_best_splits(words, k_rows)

    rows = []
    prev = 0
    for split in best_splits:
        rows.append(formatter(' '.join(words[prev:split])))
        prev = split
    rows.append(formatter(' '.join(words[prev:])))

    # Build nested LaTeX tabular for multi-line labels
    formatted_rows = [format_row(row, type) for row in rows]
    joined = r' \\[-2pt] '.join(formatted_rows)
    return rf'\begin{{tabular}}{{@{{}}c@{{}}}}{joined}\end{{tabular}}'

def format_row(row: str, label_type: str) -> str:
    if label_type != 'intent':
        return rf'{{\small\textrm{{{row}}}}}'
    
    parts = re.split(r'(\$[^$]+\$)', row)
    result = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            result.append(part[1:-1])
        elif part.strip():
            result.append(rf'\mathit{{{part}}}')
    
    return rf'${{\small {"".join(result)}}}$'

def _split_words(text: str) -> list[str]:
    '''
    Split text into words, respecting LaTeX command boundaries via TexSoup.
    Falls back to whitespace splitting for plain text.
    '''
    if '\\' in text or '{' in text or '_' in text or '^' in text:
        try:
            soup = TexSoup(text, skip_envs=(), tolerance=1)
            tokens = []
            for node in soup.children:
                s = str(node).strip()
                if s:
                    tokens.append(s)
            if len(tokens) > 1 and ''.join(tokens) == ''.join(text.split()):
                return tokens
        except Exception:
            pass

    return text.split()


def _find_best_splits(words: list[str], k: int) -> list[int]:
    '''
    Dynamic programming: partition `words` into at most k non-empty rows
    minimising variance (equiv. minimising sum of squared lengths).
    Returns split indices (exclusive end of each row except the last).
    '''
    n = len(words)
    k = min(k, n)  # can't have more rows than words

    # Precompute cumulative lengths including spaces
    # cum_len[i] = total chars in words[0..i-1] joined by spaces
    cum_len = [0] * (n + 1)
    for i in range(n):
        cum_len[i + 1] = cum_len[i] + len(words[i]) + (1 if i > 0 else 0)

    def row_len(i, j):
        '''Length of words[i:j] joined by spaces.'''
        length = cum_len[j] - cum_len[i]
        if i > 0:
            length -= 1  # remove the leading space offset
        return cum_len[j] - cum_len[i] - (1 if i > 0 else 0)

    # dp[r][i] = (min cost, split list) using r rows for words[0:i]
    INF = float('inf')
    dp  = [[INF] * (n + 1) for _ in range(k + 1)]
    par = [[None] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0

    for r in range(1, k + 1):
        for j in range(r, n + 1):          # need at least r words for r rows
            for i in range(r - 1, j):      # words[i:j] form row r
                rl = row_len(i, j)
                cost = dp[r - 1][i] + rl * rl
                if cost < dp[r][j]:
                    dp[r][j] = cost
                    par[r][j] = i

    # Find best number of rows (1..k) — might be better to use fewer
    best_r, best_cost = k, dp[k][n]
    for r in range(1, k):
        if dp[r][n] < best_cost:
            best_cost, best_r = dp[r][n], r

    # Traceback
    splits = []
    j = n
    for r in range(best_r, 0, -1):
        i = par[r][j]
        if r > 1:
            splits.append(i)
        j = i
    splits.reverse()
    return splits

def measure_ink_mm(text: str, fontsize_pt: float, desired_height: float) -> tuple[float, float, float]:
    '''
    Calculates the scale needed to reach desired_height and returns
    (final_width_mm, final_height_mm, calculated_scale)
    '''
    # 1. First pass: Measure at base fontsize to find the height error
    fig = plt.figure(dpi=DPI)
    ax = fig.add_subplot(111)
    t = ax.text(0.5, 0.5, text, fontsize=fontsize_pt, usetex=True)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    
    bbox = t.get_tightbbox(renderer)
    px_per_mm = DPI / 25.4
    initial_height_mm = bbox.height / px_per_mm
    plt.close(fig)

    if desired_height < 0:
        return -1, initial_height_mm, -1

    # 2. Calculate the scale
    # Ratio of how much bigger/smaller the font needs to be
    rows = len(text.split(r'\\[-2pt]'))
    height_ratio = (desired_height * rows + 0.1 * rows) / initial_height_mm
    new_fontsize = fontsize_pt * height_ratio
    scale = new_fontsize - fontsize_pt

    # 3. Second pass: Measure at the new scaled size for final accuracy
    fig = plt.figure(dpi=DPI)
    ax = fig.add_subplot(111)
    t = ax.text(0.5, 0.5, text, fontsize=fontsize_pt + scale, usetex=True)
    
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    
    bbox = t.get_tightbbox(renderer)
    final_w_mm = bbox.width / px_per_mm
    final_h_mm = bbox.height / px_per_mm
    plt.close(fig)

    return final_w_mm, final_h_mm, new_fontsize

def compute_label_candidates(
        G: nx.Graph,
        concepts: List[int],
        label_text: str,
        label_type: str,
        desired_height: float
):
    '''
    For every concept node, return candidate label placements.

    Parameters
    ----------
    G                  : graph with normalised 'pos' attributes
    concepts           : list of concept node ids
    label_text         : the label string (used to measure ink size)
    physical_height_mm : actual drawing height in mm (sets the unit scale)
    label_type         : 'extent' only top anchors, 'intent' only bottom anchors

    Returns
    -------
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
    ink_w_mm, ink_h_mm, scale = measure_ink_mm(label_text, font_size, desired_height)

    # padding
    rows = len(label_text.split(r'\\[-2pt]'))
    padding = (ink_h_mm / rows - 0.1 * rows) * 0.75

    # Inner half-extents (ink only, no padding)
    half_iw = (ink_w_mm  * units_per_mm) / 2.0
    half_ih = (ink_h_mm  * units_per_mm) / 2.0

    corner_multiplier = 1 / np.sqrt(2)

    candidates: Dict[int, List[LabelCandidate]] = {}

    for node in concepts:
        node_x, node_y = G.nodes[node]['pos']
        node_candidates: List[LabelCandidate] = []
        
        for anchor in allowed_anchors:
            is_top    = 'top' in anchor
            is_bottom = 'bottom' in anchor
            is_left   = 'left' in anchor
            is_right  = 'right' in anchor
            is_corner = '_' in anchor

            # visually closer corner
            s = corner_multiplier if is_corner else (1.0 / corner_multiplier)

            p_top    = (padding * s) if is_top else padding
            p_bottom = (padding * s) if is_bottom else padding
            p_left   = (padding * s) if is_left else padding
            p_right  = (padding * s) if is_right else padding

            total_w_units = (ink_w_mm + p_left + p_right) * units_per_mm
            total_h_units = (ink_h_mm + p_top + p_bottom) * units_per_mm

            half_w = total_w_units / 2.0
            half_h = total_h_units / 2.0

            offset_x = 0
            if is_left:  offset_x = half_w   # Box moves right, so left edge touches node
            if is_right: offset_x = -half_w  # Box moves left, so right edge touches node
            
            offset_y = 0
            if is_top:    offset_y = -half_h # Box moves down, so top edge touches node
            if is_bottom: offset_y = half_h  # Box moves up, so bottom edge touches node

            cx = node_x + offset_x
            cy = node_y + offset_y

            # 5. Define Bounding Boxes
            bl = (cx - half_w, cy - half_h)
            tr = (cx + half_w, cy + half_h)
            br, tl = (tr[0], bl[1]), (bl[0], tr[1])

            # Inner ink stays centered within the box relative to its specific padding
            # Ink center = Box center shifted by the difference in padding
            ink_cx = cx + ((p_left - p_right) * units_per_mm / 2.0)
            ink_cy = cy + ((p_bottom - p_top) * units_per_mm / 2.0)
            
            ibl = (ink_cx - half_iw, ink_cy - half_ih)
            itr = (ink_cx + half_iw, ink_cy + half_ih)
            ibr, itl = (itr[0], ibl[1]), (ibl[0], itr[1])

            # 6. Expanded Collision (Only expand outward from the node)
            exp_l = (padding * units_per_mm) if not is_left else 0
            exp_r = (padding * units_per_mm) if not is_right else 0
            exp_t = (padding * units_per_mm) if not is_top else 0
            exp_b = (padding * units_per_mm) if not is_bottom else 0

            ebl = (bl[0] - exp_l, bl[1] - exp_b)
            etr = (tr[0] + exp_r, tr[1] + exp_t)

            node_candidates.append(LabelCandidate(
                anchor=anchor,
                label_type=label_type,
                bbox_corners=(bl, br, tr, tl),
                inner_bbox_corners=(ibl, ibr, itr, itl),
                expanded_bbox_corners=(ebl, (etr[0], ebl[1]), etr, (ebl[0], etr[1])),
                center=(cx, cy),
            ))

        candidates[node] = node_candidates

    return scale, candidates

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
        desired_height: float
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
    ink_w_mm, ink_h_mm, _ = measure_ink_mm(label_text, font_size, desired_height)

    # padding
    corner_multiplier = 1 / np.sqrt(2)
    s = (1.0 / corner_multiplier)
    rows = len(label_text.split(r'\\[-2pt]'))
    padding = (ink_h_mm / rows - 0.1 * rows) * 0.75

    # Ink only
    half_iw = (ink_w_mm * units_per_mm) / 2.0
    half_ih = (ink_h_mm * units_per_mm) / 2.0

    # Tight bbox (collision)
    half_w  = ((ink_w_mm + 2 * padding / 3) * units_per_mm) / 2.0
    half_h  = ((ink_h_mm + 2 * padding / 3) * units_per_mm) / 2.0

    # Extended bbox (visual)
    half_ew = ((ink_w_mm + 2 * padding * s) * units_per_mm) / 2.0
    half_eh = ((ink_h_mm + 2 * padding * s) * units_per_mm) / 2.0

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