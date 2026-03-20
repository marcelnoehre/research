"""
plot.py
-------
Visualises a concept-lattice drawing with optional face colouring,
intersection markers, concept annotations, and label placement candidates.
"""

from typing import Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fcapy.context import FormalContext
from fcapy.lattice import ConceptLattice

from fca.lattice import Lattice
from placement import LabelCandidate

# One distinct colour per anchor position
_ANCHOR_COLOURS = {
    'top_left':     '#e41a1c',   # red
    'top_right':    '#377eb8',   # blue
    'bottom_left':  '#4daf4a',   # green
    'bottom_right': '#ff7f00',   # orange
}


def _draw_candidate(
        ax,
        candidate: LabelCandidate,
        label_text: str,
        fontsize_pt: float,
        alpha: float = 0.20,
):
    """
    Draw outer bbox and text for a single candidate.
    Returns the text Artist so the caller can measure its true ink extent
    after a canvas draw and add the inner box.
    """
    bl, br, tr, tl = candidate.bbox_corners
    colour = _ANCHOR_COLOURS[candidate.anchor]

    # Outer bbox — dashed, semi-transparent fill
    ax.add_patch(mpatches.Polygon(
        [bl, br, tr, tl], closed=True,
        facecolor=colour, edgecolor=colour,
        alpha=alpha, linestyle='--', linewidth=1.0,
        zorder=3,
    ))

    # Expanded bbox — dotted outline only, no fill
    ebl, ebr, etr, etl = candidate.expanded_bbox_corners
    ax.add_patch(mpatches.Polygon(
        [ebl, ebr, etr, etl], closed=True,
        facecolor='none', edgecolor=colour,
        alpha=0.3, linestyle=':', linewidth=0.8,
        zorder=3,
    ))

    # Small dot at the anchor corner
    anchor_pt = {
        'top_left':     tl,
        'top_right':    tr,
        'bottom_left':  bl,
        'bottom_right': br,
    }[candidate.anchor]
    ax.scatter(*anchor_pt, color=colour, s=20, zorder=6, alpha=0.8)

    # Text — return artist so caller can measure it
    cx, cy = candidate.center
    return ax.text(
        cx, cy, label_text,
        ha='center', va='center',
        fontsize=fontsize_pt,
        color=colour,
        alpha=1.0,
        zorder=7,
    )


def _add_inner_boxes(ax, fig, text_colour_pairs: list) -> None:
    """
    After the canvas has been drawn, measure each text artist's true ink
    extent via get_tightbbox (no descender padding) and draw a tight inner
    box around it.
    """
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
        output_path: str = "size_of_faces.pdf",
        title: str = '',
        # --- node display ---
        annotations: bool = False,
        origin: bool = False,
        # --- intersection markers ---
        intersections: List[Tuple] = [],
        show_intersections: bool = True,
        # --- face areas ---
        show_face_areas: bool = False,
        cycles: List = [],
        areas: List[float] = [],
        centers: List[Tuple] = [],
        # --- label candidates ---
        show_label_candidates: bool = False,
        label_candidates: Dict[int, List[LabelCandidate]] = {},
        label_texts: Dict[int, str] = {},
        fontsize_pt: float = 10.0,
) -> None:
    """
    Draw the lattice diagram and save to a PDF.

    Parameters
    ----------
    G                     : planar graph (used for node positions of synthetic nodes)
    context               : FCA formal context
    concepts              : list of concept node ids to draw
    coordinates           : dict mapping concept id → (x, y) — used only as fallback
                            if G nodes have normalized 'pos', those take priority
    output_path           : where to save the PDF
    title                 : window title
    annotations           : if True, label each node with its index
    origin                : if True, mark the origin (0, 0) with a red dot
    intersections         : list of (x, y) points to mark as red dots
    show_intersections    : if True, draw red dots at intersection points
    show_face_areas       : if True, colour faces by area and show area labels
    cycles                : list of node-lists representing bounded faces
    areas                 : area of each face in `cycles`
    centers               : centroid of each face in `cycles`
    show_label_candidates : if True, draw all candidate label bboxes with text
    label_candidates      : dict mapping node id → list of LabelCandidate
    label_texts           : dict mapping node id → label string (e.g. '$c_{0}$')
    fontsize_pt           : font size used when rendering label text in candidates
    """
    cmap = cm.YlOrRd

    def pos(node):
        if node in G.nodes and 'pos' in G.nodes[node]:
            return G.nodes[node]['pos']
        return coordinates[node]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title(title)

    # ── Face areas ────────────────────────────────────────────────────────
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
            ax.annotate(
                f'{area:.2f}', xy=center,
                ha='center', va='center',
                fontsize=9, color='black', zorder=5,
            )
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(areas), max(areas)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Face area')

    # ── Origin marker ─────────────────────────────────────────────────────
    if origin:
        ax.scatter(0, 0, facecolor='red', edgecolor='red', linewidth=5, s=150, zorder=10)

    # ── Label placement candidates ────────────────────────────────────────
    text_colour_pairs = []
    if show_label_candidates and label_candidates:
        for node_id, candidates in label_candidates.items():
            for candidate in candidates:
                text = label_texts.get((node_id, candidate.label_type),
                       label_texts.get(node_id, str(node_id)))
                txt = _draw_candidate(ax, candidate, text, fontsize_pt)
                text_colour_pairs.append((txt, _ANCHOR_COLOURS[candidate.anchor]))

        legend_handles = [
            mpatches.Patch(facecolor=colour, edgecolor=colour,
                           alpha=0.6, label=anchor.replace('_', ' '))
            for anchor, colour in _ANCHOR_COLOURS.items()
        ]
        ax.legend(handles=legend_handles, loc='upper left',
                  fontsize=8, title='Label anchor', framealpha=0.8)

    # ── Concept vertices ──────────────────────────────────────────────────
    for concept in concepts:
        x, y = pos(concept)
        ax.scatter(x, y, facecolor='white', edgecolor='black',
                   linewidth=2.5, s=150, zorder=8)
        if annotations:
            ax.annotate(
                concept,
                (x, y), textcoords='offset points', xytext=(0, 10),
                ha='center', va='bottom', fontsize=12, fontweight='bold', c='blue',
            )

    # ── Cover-relation edges ──────────────────────────────────────────────
    for i, j in Lattice(context).cover_relations():
        x0, y0 = pos(i)
        x1, y1 = pos(j)
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=2.5, zorder=2)

    # ── Intersection markers ──────────────────────────────────────────────
    if show_intersections:
        for pt in intersections:
            ax.scatter(pt[0], pt[1], facecolor='red', edgecolor='red',
                       linewidth=5, s=100, zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # Draw canvas once so text extents are available, then add inner boxes
    fig.canvas.draw()
    if text_colour_pairs:
        _add_inner_boxes(ax, fig, text_colour_pairs)

    plt.savefig(output_path, format="pdf", bbox_inches='tight')
    plt.close(fig)