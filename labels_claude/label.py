"""
label.py
--------
Renders a string using LaTeX (default font, default size) and draws:
  - an inner bounding box tightly around the rendered text (true ink bounds)
  - an outer bounding box with a configurable padding around the inner one

The figure is sized so that 1 data unit = 1 mm, making all coordinates
and dimensions directly in millimetres.

Prints the (width, height) of both boxes in mm.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

def measure_ink_mm(text: str, fontsize_pt: float) -> tuple[float, float]:
    """Return (ink_width_mm, ink_height_mm) by rasterising at 150 DPI."""
    DPI = 150.0
    fig, ax = plt.subplots(figsize=(8, 3), dpi=DPI)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.text(0.5, 0.5, text,
            ha='center', va='center', fontsize=fontsize_pt,
            transform=ax.transAxes, color='black',
            usetex=matplotlib.rcParams['text.usetex'])
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


def render_label(
        text: str,
        padding_mm: float = 2.0,
        output_path: str = "label.pdf",
) -> None:
    fontsize_pt = 10.0
    ink_w_mm, ink_h_mm = measure_ink_mm(text, fontsize_pt)

    outer_w_mm = ink_w_mm + 2 * padding_mm
    outer_h_mm = ink_h_mm + 2 * padding_mm

    print(f"Inner box : {ink_w_mm:.2f} mm x {ink_h_mm:.2f} mm")
    print(f"Outer box : {outer_w_mm:.2f} mm x {outer_h_mm:.2f} mm  (padding = {padding_mm} mm)")

    # ── Figure setup: 1 data unit = 1 mm ─────────────────────────────────
    # Size the figure so the axes fill it exactly and the coordinate scale
    # is 1 data-unit per mm.  All coordinates below are in mm.
    margin_mm  = max(ink_w_mm, ink_h_mm) * 0.5 + padding_mm * 4
    total_w_mm = outer_w_mm + 2 * margin_mm
    total_h_mm = outer_h_mm + 2 * margin_mm

    fig, ax = plt.subplots(figsize=(total_w_mm / 25.4, total_h_mm / 25.4))
    fig.subplots_adjust(0, 0, 1, 1)   # axes fills entire figure
    ax.axis('off')
    ax.set_xlim(-total_w_mm / 2,  total_w_mm / 2)
    ax.set_ylim(-total_h_mm / 2,  total_h_mm / 2)

    # Text centred at origin (0, 0)
    ax.text(0, 0, text, ha='center', va='center', fontsize=fontsize_pt)

    # ── Boxes (all coords in mm) ──────────────────────────────────────────
    inner = mpatches.Rectangle(
        (-ink_w_mm / 2, -ink_h_mm / 2), ink_w_mm, ink_h_mm,
        linewidth=1.2, edgecolor='steelblue', facecolor='none', zorder=3,
    )
    ax.add_patch(inner)

    outer = mpatches.Rectangle(
        (-outer_w_mm / 2, -outer_h_mm / 2), outer_w_mm, outer_h_mm,
        linewidth=1.2, edgecolor='tomato', facecolor='none',
        linestyle='--', zorder=3,
    )
    ax.add_patch(outer)

    # ── Annotations (coords in mm) ────────────────────────────────────────
    gap = 1.5   # mm between box edge and label

    # Blue: width above inner box, height left of inner box
    ax.text(0, ink_h_mm / 1 + gap, f"{ink_w_mm:.2f} mm",
            ha='center', va='bottom', fontsize=8, color='steelblue')
    ax.text(-ink_w_mm / 1.5 - gap, 0, f"{ink_h_mm:.2f} mm",
            ha='right', va='center', fontsize=8, color='steelblue')

    # Tomato: width below outer box, height right of outer box
    ax.text(0, -outer_h_mm / 2 - gap, f"{outer_w_mm:.2f} mm",
            ha='center', va='top', fontsize=8, color='tomato')
    ax.text(outer_w_mm / 2 + gap, 0, f"{outer_h_mm:.2f} mm",
            ha='left', va='center', fontsize=8, color='tomato')

    plt.savefig(output_path, format=output_path.rsplit('.', 1)[-1], bbox_inches='tight')
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    render_label(r"Hello, \LaTeX!", padding_mm=2.0, output_path="label.pdf")