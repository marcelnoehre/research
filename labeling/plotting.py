import matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from typing import Dict, List, Optional, Tuple
from fcapy.context import FormalContext

from fca.lattice import Lattice
from label import LabelCandidate, OverflowLabel

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
    bl, br, tr, tl = candidate.bbox_corners
    colour = _ANCHOR_COLOURS[candidate.anchor]

    if colored_label_candidates:
        ax.add_patch(mpatches.Polygon(
            [bl, br, tr, tl], closed=True,
            facecolor=colour, edgecolor=colour,
            alpha=0.30, linestyle='-', linewidth=1.6,
            zorder=3,
        ))
        # Expanded bbox — dotted outline, clearly visible for chosen label
        ebl, ebr, etr, etl = candidate.expanded_bbox_corners
        ax.add_patch(mpatches.Polygon(
            [ebl, ebr, etr, etl], closed=True,
            facecolor='none', edgecolor=colour,
            alpha=0.55, linestyle=':', linewidth=1.0,
            zorder=3,
        ))

        # Anchor dot — full size for chosen, tiny for rejected
        anchor_pt = {
            'top':          ((tl[0] + tr[0]) / 2, tl[1]),
            'bottom':       ((bl[0] + br[0]) / 2, bl[1]),
            'left':         (tl[0], (tl[1] + bl[1]) / 2),
            'right':        (tr[0], (tr[1] + br[1]) / 2),
            'top_left':     tl,
            'top_right':    tr,
            'bottom_left':  bl,
            'bottom_right': br,
        }[candidate.anchor]
        ax.scatter(*anchor_pt, color=colour, s=30, zorder=6, alpha=0.9)

    text_color = colour if colored_label_candidates else 'black'
    cx, cy = candidate.center
    return ax.text(
        cx, cy, label_text,
        ha='center', va='center',
        fontsize=fontsize_pt,
        color=text_color,
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
    bl, br, tr, tl = candidate.bbox_corners
    colour = '#17becf' # tab:cyan
    # Anchor dot — full size for chosen, tiny for rejected
    anchor_pt = {
        'top':          ((tl[0] + tr[0]) / 2, tl[1]),
        'bottom':       ((bl[0] + br[0]) / 2, bl[1]),
        'left':         (tl[0], (tl[1] + bl[1]) / 2),
        'right':        (tr[0], (tr[1] + br[1]) / 2),
        'top_left':     tl,
        'top_right':    tr,
        'bottom_left':  bl,
        'bottom_right': br,
        'overflow':     candidate.center
    }[candidate.anchor]

    if colored_label_candidates:
        ax.add_patch(mpatches.Polygon(
            [bl, br, tr, tl], closed=True,
            facecolor=colour, edgecolor=colour,
            alpha=0.30, linestyle='-', linewidth=1.6,
            zorder=3,
        ))

        ax.scatter(*anchor_pt, color=colour, s=30, zorder=6, alpha=0.9)

        ebl, ebr, etr, etl = candidate.expanded_bbox_corners
        ax.add_patch(mpatches.Polygon(
            [ebl, ebr, etr, etl], closed=True,
            facecolor='none', edgecolor=colour,
            alpha=0.55, linestyle=':', linewidth=1.0,
            zorder=3,
        ))

    if candidate.anchor != 'overflow':
        ax.scatter(*anchor_pt, color='grey', s=15, zorder=6, alpha=0.8)
        nx, ny = G.nodes[candidate.node_id]['pos']
        ax.plot([anchor_pt[0], nx], [anchor_pt[1], ny], color='grey', linestyle='--', linewidth=0.5, alpha=0.4, zorder=4)

    text_color = colour if colored_label_candidates else 'black'
    cx, cy = candidate.center

    return ax.text(
        cx, cy, label_text,
        ha='center', va='center',
        fontsize=fontsize_pt,
        color=text_color,
        alpha=1.0,
        zorder=7,
    )

def _add_inner_boxes(ax, fig, text_colour_pairs: list) -> None:
    '''
    '''
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for txt, colour in text_colour_pairs:
        bb = txt.get_tightbbox(renderer=renderer)
        if bb is None:
            continue
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle='square,pad=0',
            facecolor='none', edgecolor=colour,
            alpha=0.9, linestyle='-', linewidth=0.8,
            zorder=4,
        ))

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
        fontsize_pt: float = matplotlib.rcParams.get('font.size', 10.0),
        show_label_candidates: bool = False,
        colored_label_candidates: bool = False,
        overflow_labels: Dict = {},
        show_overflow_labels: bool = False
) -> None:
    '''
    Draw the lattice diagram and save to a PDF.

    Parameters
    ----------
    TODO: Description of parameters
    '''
    cmap = cm.YlOrRd

    def _pos(node):
        if node in G.nodes and 'pos' in G.nodes[node]:
            return G.nodes[node]['pos']
        return coordinates[node]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title(title)

    ##### vertices #####
    for concept in concepts:
        x, y = _pos(concept)
        ax.scatter(x, y, facecolor='white', edgecolor='black',
                   linewidth=2.5, s=150, zorder=8)
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
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=2.5, zorder=2)

    ##### intersection markers #####
    if show_intersections:
        for pt in intersections:
            ax.scatter(
                pt[0], pt[1], 
                facecolor='white', 
                edgecolor='tab:red', 
                linewidth=2.5, 
                s=150,        # Matched vertex size (150)
                zorder=10     # Kept high zorder to stay on top
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
                    txt = _draw_candidate(ax, candidate, text, fontsize_pt, colored_label_candidates)
                    # only measure ink box for chosen labels
                    text_colour_pairs.append((txt, _ANCHOR_COLOURS[candidate.anchor]))

        if colored_label_candidates:
            legend_handles = [
                mpatches.Patch(facecolor=colour, edgecolor=colour,
                            alpha=0.6, label=anchor.replace('_', ' '))
                for anchor, colour in _ANCHOR_COLOURS.items()
            ]
            ax.legend(handles=legend_handles, loc='upper left',
                    fontsize=8, title='Label anchor', framealpha=0.8)
    
    ##### overflow candidates #####
    if show_overflow_labels and overflow_labels and label_texts:
        for overflow_label in overflow_labels.values():
            text = overflow_label.text
            txt = _draw_overflow_candidate(G, ax, overflow_label, text, fontsize_pt, colored_label_candidates)
            text_colour_pairs.append((txt, '#17becf')) # tab:cyan for overflow

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    if colored_label_candidates and text_colour_pairs:
        _add_inner_boxes(ax, fig, text_colour_pairs)

    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    plt.close(fig)