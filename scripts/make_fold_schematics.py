"""Generate the north-fold schematic figures for ``docs/grid_topology.md``.

Two publication-quality figures are written to ``docs/images``:

* ``fold_pivots.png``  -- where the pole sits for each of the four fold
  pivot conventions (center/T, corner/F, U, V).
* ``fold_halo.png``    -- how the fold reconstructs the northern halo row,
  for a scalar (mirror only) and a vector component (mirror + sign flip),
  using the most common ``corner`` (F-point) pivot.

The halo source mapping in ``fold_halo.png`` is not hand-derived: we encode
each interior cell with its flat index and run it through ``xgcm.padding.pad``,
so every arrow shows the exact cell xgcm pulls from.

    python scripts/make_fold_schematics.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import FancyArrowPatch

from xgcm import Grid
from xgcm.padding import pad

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

# ---- palette -------------------------------------------------------------
INK = "#22303f"          # near-black for fold line / outlines
FACE = "#c9d4de"         # cell-face gridlines
CENTRE = "#e3eaf0"       # cell-centre gridlines (lighter)
ACCENT = "#0b6e4f"       # the emphasised pivot point type
POLE = "#f4a300"         # the two poles (stars)
POS = "#f0f4f8"          # positive scalar fill
NEG = "#fbe6e3"          # negative scalar fill

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.linewidth": 0.0,
        "savefig.dpi": 200,
        "figure.dpi": 120,
    }
)

# Four pivots: (label, models, X-role "seam", Y-role "fold", marker)
PIVOTS = [
    ("center  ·  T-point", "MOM5, generic tracer pole", "center", "center", "o"),
    ("corner  ·  F-point", "MOM6 / OM4, NEMO / ORCA", "edge", "edge", "s"),
    ("U-point", "Oceananigans TripolarGrid (tracer zipper)", "edge", "center", ">"),
    ("V-point", "north/south velocity-face pole", "center", "edge", "^"),
]


def _grid_backdrop(ax, nx, fold_y):
    """Light staggered-cell backdrop: faces (solid) and centres (dotted)."""
    for x in range(nx + 1):
        ax.axvline(x, color=FACE, lw=0.9, zorder=0)
    for x in np.arange(0.5, nx, 1.0):
        ax.axvline(x, color=CENTRE, lw=0.9, ls=(0, (1, 2)), zorder=0)
    for y in range(fold_y + 1):
        ax.axhline(y, color=FACE, lw=0.9, zorder=0)
    for y in np.arange(0.5, fold_y, 1.0):
        ax.axhline(y, color=CENTRE, lw=0.9, ls=(0, (1, 2)), zorder=0)


def fig_pivots():
    """2x2 schematic of the four fold pivot conventions."""
    nx, ny = 8, 2  # show the top two cell rows
    fold_y = ny
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.6))

    for ax, (label, models, seam, fold, marker) in zip(axes.ravel(), PIVOTS):
        _grid_backdrop(ax, nx, fold_y)

        # the fold line: the north edge of the logical grid
        ax.axhline(fold_y, color=INK, lw=2.6, zorder=4)

        # emphasised point type -- where this pivot's pole lives.
        # X-role: edge -> on integer columns (0..nx); center -> half-integer.
        xs = np.arange(0, nx + 1) if seam == "edge" else np.arange(0.5, nx, 1.0)
        # Y-role: edge -> rows on integer faces incl. the fold line (redundant
        # row sits on it); center -> rows at half-integers, below the line.
        ys = np.arange(0, fold_y + 1) if fold == "edge" else np.arange(0.5, fold_y, 1.0)
        gx, gy = np.meshgrid(xs, ys)
        ax.scatter(
            gx, gy, s=46, marker=marker, facecolor="white",
            edgecolor=ACCENT, linewidth=1.6, zorder=5,
        )

        # the two poles (bipolar seam): half a domain apart along the fold line.
        pole_x = (0.0, nx / 2) if seam == "edge" else (0.5, nx / 2 + 0.5)
        ax.scatter(
            pole_x, [fold_y, fold_y], s=320, marker="*",
            facecolor=POLE, edgecolor=INK, linewidth=1.2, zorder=7,
        )
        for px in pole_x:
            ax.annotate(
                "pole", (px, fold_y), xytext=(0, 9), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold",
                zorder=8,
            )

        ax.set_title(label, fontsize=13, fontweight="bold", pad=20, loc="left")
        ax.text(
            0.0, 1.045, models, transform=ax.transAxes, fontsize=9.5,
            color="0.4", ha="left", va="bottom",
        )
        ax.text(
            nx, fold_y + 0.34, "fold (north edge)", ha="right", va="bottom",
            fontsize=8.5, color=INK, style="italic",
        )
        skip = "redundant row on the fold line (skipped)" if fold == "edge" \
            else "top row half a cell below the fold line"
        ax.text(
            nx / 2, -0.42, f"X = {seam},  Y = {fold}   →   {skip}",
            ha="center", va="top", fontsize=9, color=ACCENT,
        )

        ax.set_xlim(-0.6, nx + 0.6)
        ax.set_ylim(-0.75, fold_y + 0.95)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "The four north-fold pivot conventions — where the pole sits on the grid",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.55, wspace=0.18)
    out = OUT / "fold_pivots.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _halo_source(nx):
    """Use xgcm to get the exact interior column each halo cell pulls from,
    for a center/center scalar under the ``corner`` (F) pivot."""
    coords = dict(xh=np.arange(nx), yh=np.arange(3))
    grid = Grid(
        xr.Dataset(coords=coords),
        coords={"X": {"center": "xh"}, "Y": {"center": "yh"}},
        boundary={"X": "periodic", "Y": {"fold": "corner"}},
        autoparse_metadata=False,
    )
    # encode each cell with its column index, fold one halo row, read the source
    enc = xr.DataArray(
        np.tile(np.arange(nx), (3, 1)), dims=["yh", "xh"], coords=coords
    )
    halo = pad(enc, grid, boundary_width={"Y": (0, 1)}).isel(yh=-1).values
    return halo.astype(int)  # halo[col] = interior column it came from


def fig_halo():
    """Worked example: scalar (mirror) vs vector (mirror + sign flip)."""
    nx = 8
    src = _halo_source(nx)  # corner pivot: src[c] = (-c-1) % nx

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.4))
    titles = [
        "Scalar (e.g. tracer):  halo = interior row mirrored about the pole",
        "Vector component (e.g. velocity):  same mirror, but the sign flips",
    ]
    for ax, title, is_vec in zip(axes, titles, (False, True)):
        # interior top row at y=0, reconstructed halo row at y=1
        for c in range(nx):
            s = src[c]
            # interior cell value: a smooth zonal pattern
            val = np.cos(2 * np.pi * (s + 0.5) / nx)
            # interior cell (bottom row)
            ax.add_patch(plt.Rectangle((s, 0), 1, 1, facecolor=POS if val >= 0 else NEG,
                                       edgecolor=FACE, lw=0.8, zorder=1))
            # halo cell (top row) -- value mirrored; sign-flipped if vector
            hval = -val if is_vec else val
            ax.add_patch(plt.Rectangle((c, 1.2), 1, 1, facecolor=POS if hval >= 0 else NEG,
                                       edgecolor=FACE, lw=0.8, zorder=1))

            if is_vec:
                # draw arrows whose direction encodes sign
                ax.annotate("", (s + 0.78 if val >= 0 else s + 0.22, 0.5),
                            (s + 0.22 if val >= 0 else s + 0.78, 0.5),
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=3)
                ax.annotate("", (c + 0.78 if hval >= 0 else c + 0.22, 1.7),
                            (c + 0.22 if hval >= 0 else c + 0.78, 1.7),
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6), zorder=3)
            else:
                ax.text(s + 0.5, 0.5, str(s), ha="center", va="center", fontsize=9, zorder=3)
                ax.text(c + 0.5, 1.7, str(s), ha="center", va="center", fontsize=9, zorder=3)

        # one labelled reflection arc: interior col 5 -> halo col 2, across the
        # right pole at x = nx/2 (every other column mirrors the same way).
        ax.add_patch(FancyArrowPatch(
            (5.5, 0.98), (2.5, 1.22), connectionstyle="arc3,rad=-0.22",
            arrowstyle="-|>", mutation_scale=14, color=ACCENT, lw=1.8, zorder=5,
        ))
        ax.text(nx / 2, 2.46, "mirror about the nearest pole",
                ha="center", va="top", fontsize=9.5, color=ACCENT, fontweight="bold")

        # fold line between interior and halo
        ax.axhline(1.1, color=INK, lw=2.4, zorder=4)
        ax.text(nx, 1.12, "fold (north edge)", ha="right", va="bottom",
                fontsize=8.5, color=INK, style="italic")
        # pole markers at x = 0 and nx/2 (corner pivot -> edge seam)
        for px in (0.0, nx / 2):
            ax.scatter([px], [1.1], s=260, marker="*", facecolor=POLE,
                       edgecolor=INK, linewidth=1.1, zorder=6)

        ax.text(-0.25, 0.5, "interior\n(top row)", ha="right", va="center", fontsize=9, color="0.4")
        ax.text(-0.25, 1.7, "halo\n(folded)", ha="right", va="center", fontsize=9, color="0.4")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=6)
        ax.set_xlim(-1.9, nx + 0.3)
        ax.set_ylim(-0.25, 2.55)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "How the north fold fills the halo  ·  corner (F-point) pivot, period Nx = 8",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "fold_halo.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_pivots()
    fig_halo()
