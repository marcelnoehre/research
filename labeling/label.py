import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import re
from functools import lru_cache

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
_MATH_RE     = re.compile(r'(\$[^$]+\$)')
_LATEX_CHARS = frozenset('\\{_^')

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

def wrap_label_text(
    plain_text:    str,
    label_type:    str,
    formatter=     str,
    k_rows:        int = K_ROWS,
    max_row_chars: int = MAX_ROW_CHARS,
) -> str:
    """
    Wrap *plain_text* into at most *k_rows* rows and return a LaTeX string.
 
    Parameters
    ----------
    plain_text    : raw label text (plain or containing LaTeX).
    label_type    : 'general', 'extent', or 'intent' — controls formatting.
    formatter     : optional callable applied to each word-chunk before
                    LaTeX formatting (e.g. a translation function).
    k_rows        : maximum number of rows.
    max_row_chars : labels shorter than this are never split.
    """
    if not plain_text:
        return plain_text
 
    words = _split_words(plain_text)
    n     = len(words)
 
    if k_rows == 1 or len(plain_text) <= max_row_chars or n <= 1:
        return _format_row(formatter(plain_text), label_type)
 
    # Only word *lengths* (not the actual words) determine where to split,
    # so the cache key is length-tuples → very high hit rate for repeated labels.
    splits = _best_splits_cached(tuple(len(w) for w in words), k_rows)
 
    rows = []
    prev = 0
    for s in splits:
        rows.append(formatter(' '.join(words[prev:s])))
        prev = s
    rows.append(formatter(' '.join(words[prev:])))
 
    formatted = [_format_row(r, label_type) for r in rows]
    joined    = r' \\[-1pt] '.join(formatted)
    return rf'\begin{{tabular}}{{@{{}}c@{{}}}}{joined}\end{{tabular}}'

def _format_row(row: str, label_type: str) -> str:
    if label_type != 'intent':
        return rf'{{\small\textrm{{{row}}}}}'
 
    parts  = _MATH_RE.split(row)
    result = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            result.append(part[1:-1])          # strip delimiters; already math
        elif part.strip():
            result.append(rf'\textit{{{part}}}')
    return rf'${{\small {"".join(result)}}}$'

def _split_words(text: str) -> list[str]:
    """Split into words respecting LaTeX/math boundaries."""
    if not any(c in text for c in _LATEX_CHARS):
        return text.split()
 
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
 
        if c in ' \t\n':
            i += 1
            continue
 
        if c == '$':
            j = text.find('$', i + 1)
            if j == -1:
                j = n - 1
            tokens.append(text[i : j + 1])
            i = j + 1
            continue
 
        if c == '\\':
            j = i + 1
            while j < n and text[j].isalpha():
                j += 1
            if j == i + 1 and j < n:   # single non-alpha command char
                j += 1
            if j < n and text[j] == '{':
                depth, k = 1, j + 1
                while k < n and depth:
                    if   text[k] == '{': depth += 1
                    elif text[k] == '}': depth -= 1
                    k += 1
                j = k
            tokens.append(text[i : j])
            i = j
            continue
 
        if c == '{':
            depth, j = 1, i + 1
            while j < n and depth:
                if   text[j] == '{': depth += 1
                elif text[j] == '}': depth -= 1
                j += 1
            tokens.append(text[i : j])
            i = j
            continue
 
        # plain word
        j = i
        while j < n and text[j] not in ' \t\n\\{$':
            j += 1
        tokens.append(text[i : j])
        i = j
 
    return tokens or text.split()

@lru_cache(maxsize=4096)
def _best_splits_cached(word_lens: tuple[int, ...], k: int) -> tuple[int, ...]:
    """
    Cached wrapper around the DP.  Key is word *lengths* only, so labels
    that differ in content but have the same length profile share a result.
    Returns split indices as a tuple (hashable, cache-friendly).
    """
    return tuple(_dp_splits(word_lens, k))
 
 
def _dp_splits(word_lens: tuple[int, ...], k: int) -> list[int]:
    """
    O(k · n²) DP partitioning *word_lens* into at most *k* rows,
    minimising the sum of squared row lengths (≡ minimising variance).
    Returns a list of split indices (exclusive end of each row except last).
    """
    n = len(word_lens)
    k = min(k, n)
 
    # cum[i] = total chars in words[0..i-1] joined by single spaces
    cum = [0] * (n + 1)
    for i, wl in enumerate(word_lens):
        cum[i + 1] = cum[i] + wl + (1 if i else 0)
 
    # seg_sq[i][j] = (length of words[i:j])²
    seg_sq = [[0] * (n + 1) for _ in range(n)]
    for i in range(n):
        base = cum[i]
        off  = 1 if i else 0
        for j in range(i + 1, n + 1):
            raw = cum[j] - base - off
            seg_sq[i][j] = raw * raw
 
    INF    = float('inf')
    stride = n + 1
    dp     = [INF] * ((k + 1) * stride)
    par    = [-1]  * ((k + 1) * stride)
    dp[0]  = 0.0
 
    for r in range(1, k + 1):
        r_off  = r       * stride
        r1_off = (r - 1) * stride
        sq_r   = seg_sq  # local ref for speed
        for j in range(r, n + 1):
            best = INF
            bi   = -1
            for i in range(r - 1, j):
                prev = dp[r1_off + i]
                if prev == INF:
                    continue
                cost = prev + sq_r[i][j]
                if cost < best:
                    best = cost
                    bi   = i
            dp[r_off + j] = best
            par[r_off + j] = bi
 
    # choose fewest rows that achieve the minimum cost
    best_r    = k
    best_cost = dp[k * stride + n]
    for r in range(1, k):
        c = dp[r * stride + n]
        if c < best_cost:
            best_cost = c
            best_r    = r
 
    # traceback
    splits: list[int] = [0] * (best_r - 1)
    j = n
    for r in range(best_r, 1, -1):
        i          = par[r * stride + j]
        splits[r - 2] = i
        j          = i
 
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
    
    bbox = t.get_window_extent(renderer)
    px_per_mm = DPI / 25.4
    initial_height_mm = bbox.height / px_per_mm
    plt.close(fig)

    if desired_height < 0:
        return -1, initial_height_mm, -1

    # 2. Calculate the scale
    # Ratio of how much bigger/smaller the font needs to be
    rows = len(text.split(r'\\[-1pt]'))
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
    
    bbox = t.get_window_extent(renderer)
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
    rows = len(label_text.split(r'\\[-1pt]'))
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
    rows = len(label_text.split(r'\\[-1pt]'))
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