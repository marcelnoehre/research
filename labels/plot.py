"""
plot.py
-------
Visualises a concept-lattice drawing with optional face colouring,
intersection markers, concept annotations, and label placement candidates.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from fcapy.context import FormalContext

from fca.lattice import Lattice
from placement import LabelCandidate, OverflowLabel

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
        chosen: bool = False,
):
    """
    Draw outer bbox and text for a single candidate.

    chosen=True  → solid fill, full opacity, thick border (the hybrid-selected label)
    chosen=False → dashed outline, low-alpha fill (rejected / unresolved candidate)

    Returns the text Artist so the caller can measure its true ink extent
    after a canvas draw and add the inner box.
    """
    bl, br, tr, tl = candidate.bbox_corners
    colour = _ANCHOR_COLOURS[candidate.anchor]

    if chosen:
        # Solid, prominent outer bbox
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
    else:
        # Dim dashed outline — rejected candidate
        ax.add_patch(mpatches.Polygon(
            [bl, br, tr, tl], closed=True,
            facecolor=colour, edgecolor=colour,
            alpha=0.08, linestyle='--', linewidth=0.7,
            zorder=2,
        ))
        # Expanded bbox — dotted, very dim
        ebl, ebr, etr, etl = candidate.expanded_bbox_corners
        ax.add_patch(mpatches.Polygon(
            [ebl, ebr, etr, etl], closed=True,
            facecolor='none', edgecolor=colour,
            alpha=0.12, linestyle=':', linewidth=0.6,
            zorder=2,
        ))

    # Anchor dot — full size for chosen, tiny for rejected
    anchor_pt = {
        'top_left':     tl,
        'top_right':    tr,
        'bottom_left':  bl,
        'bottom_right': br,
    }[candidate.anchor]
    ax.scatter(*anchor_pt, color=colour, s=30 if chosen else 8,
               zorder=6, alpha=0.9 if chosen else 0.35)

    # Text
    cx, cy = candidate.center
    return ax.text(
        cx, cy, label_text,
        ha='center', va='center',
        fontsize=fontsize_pt,
        color=colour,
        alpha=1.0 if chosen else 0.25,
        zorder=7 if chosen else 4,
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


_NODE_RADIUS_PT = np.sqrt(150) / 2   # matches s=150 scatter → radius in points


def _draw_leader_line(
        ax,
        node_pos:       Tuple[float, float],
        label_center:   Tuple[float, float],
        half_w:         float,
        half_h:         float,
        colour:         str,
        all_node_pos:   List[Tuple[float, float]],
        node_radius:    float,
        gap_data:       float = 0.15,
) -> None:
    """
    Draw a thin leader line from *node_pos* to the nearest edge of the label
    ink box, leaving a small gap at both ends.

    Obstruction avoidance
    ---------------------
    The route starts as a straight line from the node surface to the label
    box edge.  We then iteratively check every segment of the current route
    against all non-source node circles.  Whenever a segment would pass
    within (node_radius + gap_data) of a blocking node, a waypoint is
    inserted perpendicular to the segment at the blocker's closest point,
    on the side that gives the shorter detour.  This repeats until no
    segment crosses any node, or a safety iteration limit is reached.
    """
    import math

    def _seg_closest_t(ax_, ay_, bx_, by_, px_, py_) -> float:
        """Parameter t in [0,1] of the closest point on AB to P."""
        abx, aby = bx_ - ax_, by_ - ay_
        len2 = abx * abx + aby * aby
        if len2 < 1e-12:
            return 0.0
        return max(0.0, min(1.0, ((px_ - ax_) * abx + (py_ - ay_) * aby) / len2))

    def _seg_dist(ax_, ay_, bx_, by_, px_, py_) -> float:
        t = _seg_closest_t(ax_, ay_, bx_, by_, px_, py_)
        return math.hypot(px_ - (ax_ + t*(bx_-ax_)), py_ - (ay_ + t*(by_-ay_)))

    def _find_blocker(pts):
        """Return (segment_idx, blocker_pos) for the worst obstruction, or None."""
        worst_d   = math.inf
        worst_seg = None
        worst_blk = None
        for si in range(len(pts) - 1):
            ax_, ay_ = pts[si]
            bx_, by_ = pts[si + 1]
            for np_ in all_node_pos:
                # skip the source node
                if math.hypot(np_[0] - node_pos[0], np_[1] - node_pos[1]) < 1e-6:
                    continue
                d = _seg_dist(ax_, ay_, bx_, by_, np_[0], np_[1])
                if d < node_radius + gap_data and d < worst_d:
                    worst_d   = d
                    worst_seg = si
                    worst_blk = np_
        return (worst_seg, worst_blk) if worst_seg is not None else None

    def _insert_waypoint(pts, seg_idx, blocker):
        """Insert a waypoint that steers around *blocker* on segment *seg_idx*."""
        bx, by = blocker
        ax_, ay_ = pts[seg_idx]
        ex_, ey_ = pts[seg_idx + 1]

        # Closest point on the segment to the blocker
        t = _seg_closest_t(ax_, ay_, ex_, ey_, bx, by)
        cx_ = ax_ + t * (ex_ - ax_)
        cy_ = ay_ + t * (ey_ - ay_)

        # Perpendicular from segment to blocker
        perp_x = bx - cx_
        perp_y = by - cy_
        plen = math.hypot(perp_x, perp_y)
        if plen < 1e-9:
            # Blocker sits exactly on the segment — pick an arbitrary perp
            dx_, dy_ = ex_ - ax_, ey_ - ay_
            dlen = math.hypot(dx_, dy_) or 1.0
            perp_x, perp_y = -dy_ / dlen, dx_ / dlen
        else:
            perp_x /= plen
            perp_y /= plen

        offset = node_radius + gap_data * 2
        # Two candidate waypoints: away from blocker and toward blocker side
        wp_away   = (cx_ - perp_x * offset, cy_ - perp_y * offset)
        wp_toward = (cx_ + perp_x * offset, cy_ + perp_y * offset)

        # Choose the one farther from the blocker (i.e. the "away" side is
        # already further; pick it unless it introduces a longer detour)
        def _detour_len(wp):
            return (math.hypot(wp[0]-ax_, wp[1]-ay_) +
                    math.hypot(ex_-wp[0], ey_-wp[1]))

        wp = wp_away if _detour_len(wp_away) <= _detour_len(wp_toward) else wp_toward

        return pts[:seg_idx + 1] + [wp] + pts[seg_idx + 1:]

    nx_, ny = node_pos
    lx, ly  = label_center

    dx, dy = lx - nx_, ly - ny
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return
    ux, uy = dx / dist, dy / dist

    # Start: node surface + gap
    x0 = nx_ + ux * (node_radius + gap_data)
    y0 = ny  + uy * (node_radius + gap_data)

    # End: nearest bbox face − gap
    def _bbox_t(hw, hh, vx, vy):
        ts = []
        if abs(vx) > 1e-9: ts.append(hw / abs(vx))
        if abs(vy) > 1e-9: ts.append(hh / abs(vy))
        return min(ts) if ts else 0.0

    t_edge = _bbox_t(half_w, half_h, ux, uy)
    x1 = lx - ux * (t_edge + gap_data)
    y1 = ly - uy * (t_edge + gap_data)

    pts = [(x0, y0), (x1, y1)]

    for _ in range(20):   # safety limit
        hit = _find_blocker(pts)
        if hit is None:
            break
        seg_idx, blocker = hit
        pts = _insert_waypoint(pts, seg_idx, blocker)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=colour, linewidth=0.8, linestyle='-',
            alpha=0.6, zorder=3)


def _draw_overflow_label(
        ax,
        overflow: OverflowLabel,
        label_text: str,
        fontsize_pt: float,
        colour: str,
        all_node_pos: List[Tuple[float, float]],
        node_radius: float,
) -> None:
    """Draw the ink box, text, and leader line for one overflow label."""
    cx, cy = overflow.center
    hw, hh = overflow.half_w, overflow.half_h

    # Ink box outline
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - hw, cy - hh), 2 * hw, 2 * hh,
        boxstyle='square,pad=0',
        facecolor='none', edgecolor=colour,
        alpha=0.7, linestyle='-', linewidth=0.9,
        zorder=5,
    ))

    # Text
    ax.text(cx, cy, label_text,
            ha='center', va='center',
            fontsize=fontsize_pt, color=colour,
            alpha=0.9, zorder=6)

    # Leader line
    _draw_leader_line(
        ax,
        node_pos=overflow.node_pos,
        label_center=overflow.center,
        half_w=hw,
        half_h=hh,
        colour=colour,
        all_node_pos=all_node_pos,
        node_radius=node_radius,
    )



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
        chosen_labels: Dict[int, 'Optional[LabelCandidate]'] = {},
        label_texts: Dict[int, str] = {},
        fontsize_pt: float = 10.0,
        # --- overflow labels ---
        overflow_labels: List[OverflowLabel] = [],
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
    chosen_labels         : dict mapping node id → chosen LabelCandidate (or None)
                            produced by hybrid_label_placement(); when provided,
                            chosen candidates are drawn solid/opaque and rejected
                            ones are drawn dim/dashed for comparison
    label_texts           : dict mapping (node_id, label_type) or node_id → label string
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
            # chosen_labels[node_id] is a list of chosen LabelCandidates
            # (one per label_type that was successfully placed)
            chosen_set = set(id(c) for c in chosen_labels.get(node_id, []))
            for candidate in candidates:
                is_chosen = id(candidate) in chosen_set
                text = label_texts.get((node_id, candidate.label_type),
                       label_texts.get(node_id, str(node_id)))
                txt = _draw_candidate(ax, candidate, text, fontsize_pt,
                                      chosen=is_chosen)
                # only measure ink box for chosen labels
                if is_chosen:
                    text_colour_pairs.append((txt, _ANCHOR_COLOURS[candidate.anchor]))

        legend_handles = [
            mpatches.Patch(facecolor=colour, edgecolor=colour,
                           alpha=0.6, label=anchor.replace('_', ' '))
            for anchor, colour in _ANCHOR_COLOURS.items()
        ]
        ax.legend(handles=legend_handles, loc='upper left',
                  fontsize=8, title='Label anchor', framealpha=0.8)

    # ── Overflow labels (outside diagram) ────────────────────────────────
    if overflow_labels:
        # Collect all concept node positions and estimate node radius in data units.
        # s=150 in scatter → marker area 150 pt² → radius ≈ sqrt(150/pi) / 2 pt.
        # Convert pt → data units via the axes transform.
        all_node_pos = [pos(c) for c in concepts]
        # Approximate: measure one unit in data space vs display space
        try:
            _trans = ax.transData
            _p0 = _trans.transform((0, 0))
            _p1 = _trans.transform((1, 0))
            _pts_per_unit = abs(_p1[0] - _p0[0])   # display pts per data unit
            import math as _math
            _radius_pt = _math.sqrt(150 / _math.pi) / 2
            _node_radius = _radius_pt / _pts_per_unit if _pts_per_unit > 0 else 0.1
        except Exception:
            _node_radius = 0.1

        for ov in overflow_labels:
            text = label_texts.get((ov.node_id, ov.label_type),
                                   label_texts.get(ov.node_id, str(ov.node_id)))
            colour = '#555555'
            _draw_overflow_label(ax, ov, text, fontsize_pt, colour,
                                 all_node_pos=all_node_pos,
                                 node_radius=_node_radius)

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