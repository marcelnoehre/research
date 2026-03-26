import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from typing import Dict, List, Optional, Tuple
from fcapy.context import FormalContext

from fca.lattice import Lattice



def plot_lattice(
        G: nx.Graph,
        context: FormalContext,
        concepts: List[int],
        coordinates: Dict,
        output_path: str = "labeling.pdf",
        title: str = '',
        intersections: List[Tuple] = [],
        # --- intersection markers ---
        show_vertex_ids: bool  = False,
        show_intersections: bool = False
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

    # ##### edges (cover-relation) #####
    for i, j in Lattice(context).cover_relations():
        x0, y0 = _pos(i)
        x1, y1 = _pos(j)
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=2.5, zorder=2)

    # ##### intersection markers #####
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
            # ax.scatter(pt[0], pt[1], facecolor='red', edgecolor='red', linewidth=5, s=100, zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')

    fig.canvas.draw()

    plt.savefig(output_path, format="pdf")
    plt.close(fig)