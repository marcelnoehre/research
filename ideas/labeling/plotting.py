import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from typing import Dict, List, Tuple
from fcapy.context import FormalContext
from fca.lattice import Lattice

from label import *

MARGIN = 1.0

_ANCHOR_COLOURS = {
    'top_left':     '#d62728',   # tab:red
    'top_right':    '#1f77b4',   # tab:blue
    'bottom_left':  '#2ca02c',   # tab:green
    'bottom_right': '#ff7f0e',   # tab:orange
    'top':          '#9467bd',   # tab:purple
    'bottom':       '#8c564b',   # tab:brown
    'left':         '#e377c2',   # tab:pink
    'right':        '#7f7f7f',   # tab:gray
}


# ---------------------------------------------------------------------------
# Inline whitespace trimmer
# ---------------------------------------------------------------------------

def _trim_figure(fig, ax, padding_inches: float = 0.05, white_threshold: int = 250) -> None:
    """
    Trim whitespace around drawn content directly on *fig* / *ax* by:
      1. Rasterising the figure to an RGBA array (no file I/O).
      2. Finding the bounding box of all non-white pixels.
      3. Resizing the figure and shifting the axes so only that region is kept.

    This preserves every coloured pixel and does NOT use tight_layout or
    bbox_inches='tight', so your data-coordinate geometry is untouched.

    Parameters
    ----------
    fig             : the Figure to trim in-place
    ax              : the Axes that owns the content (used to reposition)
    padding_inches  : whitespace margin to leave around detected content
    white_threshold : RGB value (0–255) below which a pixel counts as content;
                      raise to also strip light-grey margins
    """
    # --- 1. Rasterise to RGBA numpy array --------------------------------
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    # shape: (height_px, width_px, 4)
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,)
    )
    rgb = rgba[:, :, :3]  # drop alpha

    # --- 2. Find bounding box of non-white pixels ------------------------
    # A pixel is "background" when all R, G, B >= white_threshold
    is_content = np.any(rgb < white_threshold, axis=2)  # bool (H, W)

    rows = np.any(is_content, axis=1)   # True for each row that has content
    cols = np.any(is_content, axis=0)   # True for each col that has content

    if not rows.any():
        return  # nothing to trim

    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # --- 3. Convert pixel bbox → inches ----------------------------------
    dpi = fig.get_dpi()
    fig_w_px, fig_h_px = fig.canvas.get_width_height()

    pad_px = int(round(padding_inches * dpi))

    # clamp
    col_min = max(0,        col_min - pad_px)
    col_max = min(fig_w_px, col_max + pad_px)
    row_min = max(0,        row_min - pad_px)
    row_max = min(fig_h_px, row_max + pad_px)

    new_w_in = (col_max - col_min) / dpi
    new_h_in = (row_max - row_min) / dpi

    # --- 4. Work out where the axes sits in figure-fraction space --------
    # We want the axes position to remain the same *in data coordinates*
    # but shift within the new, smaller figure canvas.

    # Current axes position in figure pixels (origin = bottom-left of figure)
    ax_pos = ax.get_position()          # fraction coords, bottom-left origin
    ax_l_px = ax_pos.x0 * fig_w_px
    ax_b_px = ax_pos.y0 * fig_h_px
    ax_w_px = ax_pos.width  * fig_w_px
    ax_h_px = ax_pos.height * fig_h_px

    # row_min/max are top-left origin → convert to bottom-left origin for y
    crop_b_px = fig_h_px - row_max   # distance from bottom of figure to crop bottom
    crop_l_px = col_min

    # New axes position within the cropped figure (fraction of new size)
    new_ax_l = (ax_l_px - crop_l_px) / (col_max - col_min)
    new_ax_b = (ax_b_px - crop_b_px) / (row_max - row_min)
    new_ax_w = ax_w_px / (col_max - col_min)
    new_ax_h = ax_h_px / (row_max - row_min)

    # --- 5. Apply --------------------------------------------------------
    fig.set_size_inches(new_w_in, new_h_in)
    ax.set_position([new_ax_l, new_ax_b, new_ax_w, new_ax_h])


# ---------------------------------------------------------------------------
# Drawing helpers (unchanged)
# ---------------------------------------------------------------------------

def _draw_candidate(
        ax,
        candidate: LabelCandidate,
        label_text: str,
        fontsize_pt: float,
        colored_label_candidates: bool = False
):
    ibl, ibr, itr, itl = candidate.inner_bbox_corners
    bl, br, tr, tl = candidate.bbox_corners
    colour = _ANCHOR_COLOURS[candidate.anchor]

    if colored_label_candidates:
        ax.add_patch(mpatches.Polygon(
            [ibl, ibr, itr, itl], closed=True,
            facecolor='none', edgecolor=colour,
            alpha=0.9, linestyle='-', linewidth=0.8,
            zorder=4, clip_on=False
        ))
        ax.add_patch(mpatches.Polygon(
            [bl, br, tr, tl], closed=True,
            facecolor=colour, edgecolor=colour,
            alpha=0.30, linestyle='-', linewidth=1.6,
            zorder=3, clip_on=False
        ))
        # Expanded bbox — dotted outline, clearly visible for chosen label
        ebl, ebr, etr, etl = candidate.expanded_bbox_corners
        ax.add_patch(mpatches.Polygon(
            [ebl, ebr, etr, etl], closed=True,
            facecolor='none', edgecolor=colour,
            alpha=0.55, linestyle=':', linewidth=1.0,
            zorder=3, clip_on=False
        ))

        # Anchor dot — full size for chosen, tiny for rejected
        anchor_pt = {
            'top':          ((tl[0] + tr[0]) / 2, tl[1]),
            'bottom':       ((bl[0] + br[0]) / 2, bl[1]),
            'left':         (tl[0] - 0.05, (tl[1] + bl[1]) / 2),
            'right':        (tr[0] + 0.05, (tr[1] + br[1]) / 2),
            'top_left':     tl,
            'top_right':    tr,
            'bottom_left':  bl,
            'bottom_right': br,
        }[candidate.anchor]
        ax.scatter(*anchor_pt, color=colour, s=30, zorder=10, alpha=0.9, clip_on=False)

    # else:
    #     padding = 0.1
    #     ax.add_patch(mpatches.Polygon(
    #         [
    #             (ibl[0] - padding, ibl[1] - padding), # Bottom Left
    #             (ibr[0] + padding, ibr[1] - padding), # Bottom Right
    #             (itr[0] + padding, itr[1] + padding), # Top Right
    #             (itl[0] - padding, itl[1] + padding)  # Top Left
    #         ], closed=True, 
    #         facecolor='white', edgecolor='grey',
    #         alpha=0.30, linestyle='-', linewidth=0.5,
    #         zorder=3, clip_on=False
    #     ))

    text_color = colour if colored_label_candidates else 'black'
    cx, cy = candidate.center
    
    rows = len(label_text.split(r'\\[-1pt]'))
    translate_x = 0.0
    if '\textit' in label_text:
        translate_x = -0.025 if 'left' in candidate.anchor else 0.025
    translate_y = 0.025 if ('\textit' in label_text and candidate.anchor != 'bottom') else 0.0
    translate_y *= rows

    return ax.text(
        cx+translate_x, cy-translate_y, label_text,
        ha='center', va='center',
        fontsize=fontsize_pt,
        color=text_color,
        clip_on=False,
        alpha=1.0,
        zorder=7,
    )

def _draw_overflow_candidate(
        G: nx.Graph,
        ax,
        candidate: OverflowLabel,
        label_text: str,
        fontsize_pt: float,
        colored_label_candidates: bool = False
):
    ibl, ibr, itr, itl = candidate.inner_bbox_corners
    bl, br, tr, tl = candidate.bbox_corners
    colour = '#17becf' # tab:cyan
    # Anchor dot — full size for chosen, tiny for rejected
    anchor_pt = {
        'top':          ((tl[0] + tr[0]) / 2, tl[1]),
        'bottom':       ((bl[0] + br[0]) / 2, bl[1]),
        'left':         (tl[0] - 0.05, (tl[1] + bl[1]) / 2),
        'right':        (tr[0] + 0.05, (tr[1] + br[1]) / 2),
        'top_left':     tl,
        'top_right':    tr,
        'bottom_left':  bl,
        'bottom_right': br,
        'overflow':     candidate.center
    }[candidate.anchor]

    if colored_label_candidates:
        ax.add_patch(mpatches.Polygon(
            [ibl, ibr, itr, itl], closed=True,
            facecolor='none', edgecolor=colour,
            alpha=0.9, linestyle='-', linewidth=0.8,
            zorder=4, clip_on=False
        ))
        ax.add_patch(mpatches.Polygon(
            [bl, br, tr, tl], closed=True,
            facecolor=colour, edgecolor=colour,
            alpha=0.30, linestyle='-', linewidth=1.6,
            zorder=3, clip_on=False
        ))

        ax.scatter(*anchor_pt, color=colour, s=30, zorder=6, alpha=0.9, clip_on=False)

        ebl, ebr, etr, etl = candidate.expanded_bbox_corners
        ax.add_patch(mpatches.Polygon(
            [ebl, ebr, etr, etl], closed=True,
            facecolor='none', edgecolor=colour,
            alpha=0.55, linestyle=':', linewidth=1.0,
            zorder=3, clip_on=False
        ))
    # else:
    #     ax.add_patch(mpatches.Polygon(
    #         [
    #             (bl[0] - 0.05, bl[1]), # Bottom Left
    #             (br[0] + 0.05, br[1]), # Bottom Right
    #             (tr[0] + 0.05, tr[1]), # Top Right
    #             (tl[0] - 0.05, tl[1])  # Top Left
    #         ], closed=True, 
    #         facecolor='white', edgecolor='grey',
    #         alpha=0.30, linestyle='-', linewidth=0.5,
    #         zorder=3, clip_on=False
    #     ))

    if candidate.anchor != 'overflow':
        ax.scatter(*anchor_pt, color='grey', s=5, zorder=6, alpha=0.8, clip_on=False)
        nx_, ny = G.nodes[candidate.node_id]['pos']
        ax.plot([anchor_pt[0], nx_], [anchor_pt[1], ny], color='grey', linestyle='--', linewidth=0.5, alpha=0.4, zorder=4, clip_on=False)

    text_color = colour if colored_label_candidates else 'black'
    cx, cy = candidate.center

    rows = len(label_text.split(r'\\[-1pt]'))
    translate_x = 0.0
    if '\textit' in label_text:
        translate_x = -0.025 if 'left' in candidate.anchor else 0.025
    translate_y = 0.025 if ('\textit' in label_text) else 0.0
    translate_y *= rows

    return ax.text(
        cx+translate_x, cy-translate_y, label_text,
        ha='center', va='center',
        fontsize=fontsize_pt,
        color=text_color,
        clip_on=False,
        alpha=1.0,
        zorder=7,
    )


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_lattice(
        G: nx.Graph,
        context: FormalContext,
        concepts: List[int],
        coordinates: Dict,
        output_path: str = "labeling.pdf",
        label_scale: Dict = {},
        title: str = '',
        show_vertex_ids: bool  = False,
        # intersections
        intersections: List[Tuple] = [],
        show_intersections: bool = False,
        # faces
        cycles: List = [],
        areas: List[float] = [],
        centers: List[Tuple] = [],
        show_face_areas: bool = False,
        show_face_sizes: bool = False,
        label_candidates: Dict = {},
        label_texts: Dict = {},
        show_label_candidates: bool = False,
        colored_label_candidates: bool = False,
        show_legend: bool = False,
        overflow_labels: Dict = {},
        show_overflow_labels: bool = False,
        # trimming
        trim_whitespace: bool = True,
        trim_padding_inches: float = 0.05,
        trim_white_threshold: int = 250,
) -> None:
    '''
    Draw the lattice diagram and save to a PDF.

    Parameters
    ----------
    trim_whitespace      : crop excess white margins before saving (default True)
    trim_padding_inches  : whitespace to keep around detected content
    trim_white_threshold : pixels with all channels >= this are treated as background
    '''
    NODE_SIZE = 50
    LINE_WIDTH = 1.0
    FONT_SIZE = compute_suggested_font_size(G)
    cmap = cm.YlOrRd

    def _pos(node):
        if node in G.nodes and 'pos' in G.nodes[node]:
            return G.nodes[node]['pos']
        return coordinates[node]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)

    fig.canvas.manager.set_window_title(title)

    ##### vertices #####
    for concept in concepts:
        x, y = _pos(concept)
        ax.scatter(x, y, facecolor='white', edgecolor='black',
                   linewidth=LINE_WIDTH, s=NODE_SIZE, zorder=100)
        if show_vertex_ids:
            ax.annotate(
                concept,
                (x, y), textcoords='offset points', xytext=(0, 10),
                ha='center', va='bottom', fontsize=12, fontweight='bold', c='blue',
            )

    ##### edges (cover-relation) #####
    for i, j in Lattice(context).cover_relations():
        x0, y0 = _pos(i)
        x1, y1 = _pos(j)
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=LINE_WIDTH, zorder=2)

    ##### intersection markers #####
    if show_intersections:
        for pt in intersections:
            ax.scatter(
                pt[0], pt[1], 
                facecolor='white', 
                edgecolor='tab:red', 
                linewidth=LINE_WIDTH, 
                s=NODE_SIZE, 
                zorder=10
            )

    ##### faces #####
    if show_face_areas and cycles and areas:
        norm = plt.Normalize(min(areas), max(areas))
        for face, area, center in zip(cycles, areas, centers):
            pts = [G.nodes[n]['pos'] for n in face]
            patch = mpatches.Polygon(
                pts, closed=True,
                facecolor=cmap(norm(area)), edgecolor='none',
                alpha=0.4, zorder=1,
            )
            ax.add_patch(patch)
            if show_face_sizes:
                ax.annotate(
                    f'{area:.2f}', xy=center,
                    ha='center', va='center',
                    fontsize=9, color='black', zorder=5,
                )
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(areas), max(areas)))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.8)
        cb.set_label(r'Face area ($\mathrm{mm}^2$)', fontsize=16)
        cb.ax.tick_params(labelsize=14)

    ##### label candidates #####
    text_colour_pairs = []
    if show_label_candidates and label_candidates and label_texts:
        for node_id, candidates in label_candidates.items():
            for candidate in candidates:
                text = label_texts.get((node_id, candidate.label_type))
                if text is not None:
                    txt = _draw_candidate(ax, candidate, text, label_scale[(node_id, candidate.label_type)], colored_label_candidates)
                    # only measure ink box for chosen labels
                    text_colour_pairs.append((txt, _ANCHOR_COLOURS[candidate.anchor]))

        if colored_label_candidates:
            legend_handles = [
                mpatches.Patch(facecolor=colour, edgecolor=colour,
                            alpha=0.6, label=anchor.replace('_', ' '))
                for anchor, colour in _ANCHOR_COLOURS.items()
            ]
            if show_legend:
                ax.legend(handles=legend_handles, loc='upper left', fontsize=8, title='Label anchor', framealpha=0.8)
    
    ##### overflow candidates #####
    if show_overflow_labels and overflow_labels and label_texts:
        for overflow_label in overflow_labels.values():
            text = overflow_label.text
            txt = _draw_overflow_candidate(G, ax, overflow_label, text, label_scale[(overflow_label.node_id, overflow_label.label_type)], colored_label_candidates)
            text_colour_pairs.append((txt, '#17becf')) # tab:cyan for overflow

    # 1. Get the range of the nodes only
    xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    ys = [G.nodes[n]['pos'][1] for n in G.nodes]

    # 2. Set the axes limits manually based on nodes + a fixed margin
    ax.set_xlim(min(xs) - MARGIN, max(xs) + MARGIN)
    ax.set_ylim(min(ys) - MARGIN, max(ys) + MARGIN)

    # 3. LOCK the aspect ratio and limits
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')

    # ── Trim whitespace (pixel-based, no tight_layout) ──────────────────
    if trim_whitespace:
        _trim_figure(fig, ax,
                     padding_inches=trim_padding_inches,
                     white_threshold=trim_white_threshold)

    plt.savefig(output_path, format="pdf")
    plt.close('all')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dynamic candidate helper (used in __main__)
# ---------------------------------------------------------------------------

def _draw_dynamic_candidate(ax, anchor_type, text, pos_coord, fontsize_pt, alpha, renderer=None):
    is_top    = 'top' in anchor_type
    is_bottom = 'bottom' in anchor_type
    is_left   = 'left' in anchor_type
    is_right  = 'right' in anchor_type
    is_corner = '_' in anchor_type

    corner_multiplier = 1 / np.sqrt(2)
    s = corner_multiplier if is_corner else (1.0 / corner_multiplier)

    t = ax.text(
        pos_coord[0], pos_coord[1], text,
        ha='center', va='center', fontsize=fontsize_pt,
        zorder=7, clip_on=False
    )
    plt.gcf().canvas.draw()

    if renderer is None:
        renderer = ax.figure.canvas.get_renderer()

    bbox_pixels = t.get_tightbbox(renderer=renderer)
    inv = ax.transData.inverted()
    bbox_data = inv.transform(bbox_pixels)
    (xmin, ymin), (xmax, ymax) = bbox_data
    
    pad = 0.005 * s
    bl, br, tr, tl = (xmin-pad, ymin-pad), (xmax+pad, ymin-pad), (xmax+pad, ymax+pad), (xmin-pad, ymax+pad)

    anchor_pt_raw = {
        'top':          ((tl[0] + tr[0]) / 2, tl[1]),
        'bottom':       ((bl[0] + br[0]) / 2, bl[1]),
        'left':         (tl[0], (tl[1] + bl[1]) / 2),
        'right':        (tr[0], (tr[1] + br[1]) / 2),
        'top_left':     tl,
        'top_right':    tr,
        'bottom_left':  bl,
        'bottom_right': br,
    }[anchor_type]

    dx = pos_coord[0] - anchor_pt_raw[0]
    dy = pos_coord[1] - anchor_pt_raw[1]

    t.set_fontsize(FONT_SIZE + 2.0)
    t.set_position((pos_coord[0] + dx, pos_coord[1] + dy - 0.0005))

    def shift(pts): return [(p[0] + dx, p[1] + dy) for p in pts]

    ibl, ibr, itr, itl = shift([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    bl, br, tr, tl = shift([bl, br, tr, tl])
    
    # Expanded BBox logic
    is_top, is_bottom = 'top' in anchor_type, 'bottom' in anchor_type
    is_left, is_right = 'left' in anchor_type, 'right' in anchor_type
    exp_l = pad if not is_left else 0
    exp_r = pad if not is_right else 0
    exp_t = pad if not is_top else 0
    exp_b = pad if not is_bottom else 0

    ebl = (bl[0] - exp_l, bl[1] - exp_b)
    ebr = (br[0] + exp_r, br[1] - exp_b)
    etr = (tr[0] + exp_r, tr[1] + exp_t)
    etl = (tl[0] - exp_l, tl[1] + exp_t)

    NODE_SIZE = 50
    LINE_WIDTH = 1.0
    ax.add_patch(mpatches.Polygon([ibl, ibr, itr, itl], closed=True, fill=False, edgecolor=_ANCHOR_COLOURS[anchor_type], lw=0.8, alpha=0.9))
    ax.add_patch(mpatches.Polygon([bl, br, tr, tl], closed=True, color=_ANCHOR_COLOURS[anchor_type], alpha=0.55, lw=1.6))
    ax.add_patch(mpatches.Polygon([ebl, ebr, etr, etl], closed=True, fill=False, edgecolor=_ANCHOR_COLOURS[anchor_type], ls=':', lw=1.0))
    ax.scatter(pos_coord[0], pos_coord[1], facecolor='white', edgecolor='black', linewidth=LINE_WIDTH, s=NODE_SIZE, zorder=8)

    return t

if __name__ == "__main__":
    G = nx.Graph()
    node_id = 0
    pos = (0, 0)
    G.add_node(node_id, pos=pos)

    side_anchors = [('left', 'L'), ('right', 'R'), ('top', 'T'), ('bottom', 'B')]
    corner_anchors = [('top_left', 'TL'), ('top_right', 'TR'), ('bottom_left', 'BL'), ('bottom_right', 'BR')]

    for anchors, fname in [(side_anchors, 'figs/anchors_side.pdf'), (corner_anchors, 'figs/anchors_corner.pdf')]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for anchor, text in anchors:
            _draw_dynamic_candidate(ax, anchor, text, pos, 12.0, 0.7, renderer=renderer)

        ax.axis('off')

        # Trim instead of tight_layout — geometry-safe
        _trim_figure(fig, ax, padding_inches=0.05)

        plt.savefig(fname, format="pdf")
        plt.close('all')