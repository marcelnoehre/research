"""
plot.py
-------
Visualises a concept-lattice drawing with optional face colouring,
intersection markers, and concept annotations.
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


def plot_lattice(
        G: nx.Graph,
        context: FormalContext,
        concepts: List[int],
        coordinates: Dict,
        output_path: str = "size_of_faces.pdf",
        title: str = '',
        annotations: bool = False,
        origin: bool = False,
        intersections: List[Tuple] = [],
        cycles: List = [],
        areas: List[float] = [],
        centers: List[Tuple] = [],
) -> None:
    """
    Draw the lattice diagram and save to a PDF.

    Parameters
    ----------
    G            : planar graph (used for node positions of synthetic nodes)
    context      : FCA formal context
    concepts     : list of concept node ids to draw
    coordinates  : dict mapping concept id → (x, y) — used only as fallback
                   if G nodes have normalized 'pos', those take priority
    output_path  : where to save the PDF
    title        : window title
    annotations  : if True, label each node with its index
    origin       : if True, mark the origin (0, 0) with a red dot
    intersections: list of (x, y) points to mark as red dots
    cycles       : list of node-lists representing bounded faces
    areas        : area of each face in `cycles`
    centers      : centroid of each face in `cycles`
    """
    lattice_view = ConceptLattice.from_context(context)
    cmap = cm.YlOrRd

    # Use normalized positions from G if available, fall back to coordinates dict
    def pos(node):
        if node in G.nodes and 'pos' in G.nodes[node]:
            return G.nodes[node]['pos']
        return coordinates[node]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title(title)

    # Colour bounded faces by area (yellow → red)
    if cycles and areas:
        norm = plt.Normalize(min(areas), max(areas))
        for face, area in zip(cycles, areas):
            pts = [G.nodes[n]['pos'] for n in face]
            patch = mpatches.Polygon(
                pts, closed=True,
                facecolor=cmap(norm(area)), edgecolor='none',
                alpha=0.4, zorder=1
            )
            ax.add_patch(patch)

    if origin:
        ax.scatter(0, 0, facecolor='red', edgecolor='red', linewidth=5, s=150, zorder=10)

    # Concept vertices
    for concept in concepts:
        x, y = pos(concept)
        ax.scatter(x, y, facecolor='white', edgecolor='black', linewidth=2.5, s=150, zorder=4)
        if annotations:
            ax.annotate(
                concept,
                (x, y), textcoords='offset points', xytext=(0, 10),
                ha='center', va='bottom', fontsize=12, fontweight='bold', c='blue'
            )

    # Cover-relation edges
    for i, j in Lattice(context).cover_relations():
        x0, y0 = pos(i)
        x1, y1 = pos(j)
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=2.5, zorder=2)

    # Intersection markers
    for pt in intersections:
        ax.scatter(pt[0], pt[1], facecolor='red', edgecolor='red',
                   linewidth=5, s=100, zorder=10)

    # Area labels at face centroids
    for center, area in zip(centers, areas):
        ax.annotate(f'{area:.2f}', xy=center, ha='center', va='center',
                    fontsize=10, color='black', zorder=5)

    # Colourbar
    if areas:
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(areas), max(areas)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Face area')

    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, format="pdf", bbox_inches='tight')
