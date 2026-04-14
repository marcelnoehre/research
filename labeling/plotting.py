import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from typing import Dict, List, Tuple
from fcapy.context import FormalContext
from fca.lattice import Lattice

from label import *

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

    text_color = colour if colored_label_candidates else 'black'
    cx, cy = candidate.center
    return ax.text(
        cx, cy, label_text,
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

    if candidate.anchor != 'overflow':
        ax.scatter(*anchor_pt, color='grey', s=5, zorder=6, alpha=0.8, clip_on=False)
        nx, ny = G.nodes[candidate.node_id]['pos']
        ax.plot([anchor_pt[0], nx], [anchor_pt[1], ny], color='grey', linestyle='--', linewidth=0.5, alpha=0.4, zorder=4, clip_on=False)

    text_color = colour if colored_label_candidates else 'black'
    cx, cy = candidate.center

    return ax.text(
        cx, cy, label_text,
        ha='center', va='center',
        fontsize=fontsize_pt,
        color=text_color,
        clip_on=False,
        alpha=1.0,
        zorder=7,
    )

def plot_lattice(
        G: nx.Graph,
        context: FormalContext,
        concepts: List[int],
        coordinates: Dict,
        output_path: str = "labeling.pdf",
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
        show_overflow_labels: bool = False
) -> None:
    '''
    Draw the lattice diagram and save to a PDF.

    Parameters
    ----------
    TODO: Description of parameters
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
                   linewidth=LINE_WIDTH, s=NODE_SIZE, zorder=8)
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
                    txt = _draw_candidate(ax, candidate, text, FONT_SIZE, colored_label_candidates)
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
            txt = _draw_overflow_candidate(G, ax, overflow_label, text, FONT_SIZE, colored_label_candidates)
            text_colour_pairs.append((txt, '#17becf')) # tab:cyan for overflow

    # 1. Get the range of the nodes only
    xs = [G.nodes[n]['pos'][0] for n in G.nodes]
    ys = [G.nodes[n]['pos'][1] for n in G.nodes]

    # 2. Set the axes limits manually based on nodes + a fixed margin
    margin = 1
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # 3. LOCK the aspect ratio and limits
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    plt.savefig(output_path, format="pdf")
    plt.close('all')
    plt.close(fig)


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

    t.set_fontsize(font_size + 2.0)
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
    font_size = 12.0
    units_per_mm = 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)

    # Draw the figure once to initialize the renderer
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    for anchor, text in side_anchors:
        _draw_dynamic_candidate(ax, anchor, text, pos, 12.0, 0.7, renderer=renderer)
    
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figs/anchors_side.pdf', format="pdf", bbox_inches='tight')
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for anchor, text in corner_anchors:
        _draw_dynamic_candidate(ax, anchor, text, pos, 12.0, 0.2, renderer=renderer)

    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figs/anchors_corner.pdf', format="pdf", bbox_inches='tight')
    plt.close('all')