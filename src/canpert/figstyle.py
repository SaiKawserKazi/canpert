"""Shared publication style: palette, typography, panel letters, schematic primitives.

Palette = the dataviz reference instance (light mode, adjacent-validated order):
CASTER takes slot 1 (blue); baselines take slots 2-4 in fixed order; the NoChange floor is
a neutral gray (it is a reference, not a competitor). Sequential = one blue ramp;
diverging = blue <-> red with a gray midpoint.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

METHODS = ("CASTER", "PerturbMean", "ContextMean", "Linear", "LineOffset", "NoChange")
COLORS = {
    "CASTER": "#2a78d6",
    "PerturbMean": "#eb6834",
    "ContextMean": "#1baf7a",
    "Linear": "#eda100",
    "LineOffset": "#e87ba4",
    "NoChange": "#b4b2a9",
}
LINE_LABEL = {"k562": "K562", "hepg2": "HepG2", "jurkat": "Jurkat", "rpe1": "RPE1"}
LINE_DESC = {"k562": "CML", "hepg2": "Hepatoblastoma", "jurkat": "T-ALL", "rpe1": "Non-transformed"}
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
SURFACE = "#ffffff"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a82a2a"])
ACCENT_PERT = "#c0392b"  # perturbation glyph in schematics (red, as in the reference style)
CTX_FILL = {"k562": "#dbe8f8", "hepg2": "#fde3d6", "jurkat": "#d4f1e6", "rpe1": "#eeeeea"}

MM = 1 / 25.4
FULL_W = 174 * MM  # Cell full-width figure
HALF_W = 85 * MM


def setup() -> None:
    """Global rcParams for Cell-style figures."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.0,
        "axes.titlesize": 6.5,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.dpi": 600,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "hatch.linewidth": 0.4,
    })


def letter(ax: plt.Axes, s: str, x: float = -0.18, y: float = 1.08) -> None:
    """Bold panel letter at the top-left of an axes."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def fig_letter(fig: plt.Figure, s: str, x: float, y: float) -> None:
    """Bold panel letter in figure coordinates."""
    fig.text(x, y, s, fontsize=9, fontweight="bold", va="top", ha="left")


def hgrid(ax: plt.Axes) -> None:
    """Recessive horizontal hairline grid behind data."""
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str = "", fc: str = "#f4f3f0",
        ec: str = INK3, fs: float = 5.5, lw: float = 0.6, r: float = 0.02, color: str = INK,
        weight: str = "normal", style: str = "normal") -> None:
    """Rounded box with centred text (data coordinates of a 0-1 schematic axes)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight=weight, style=style, linespacing=1.15)


def arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str = INK2,
          lw: float = 0.7, style: str = "-|>", ms: float = 5, ls: str = "-") -> None:
    """Straight arrow."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def cell(ax: plt.Axes, x: float, y: float, r: float, fc: str, ec: str = INK3, nucleus: bool = True) -> None:
    """A stylised cell glyph (membrane + nucleus)."""
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.5))
    if nucleus:
        ax.add_patch(Circle((x + r * 0.12, y - r * 0.05), r * 0.42, fc="white", ec=ec, lw=0.4, alpha=0.9))


def cell_population(ax: plt.Axes, cx: float, cy: float, rad: float, fc: str, n: int = 9,
                    seed: int = 0, ec: str = INK3, jitter: float = 0.25) -> None:
    """A loose cluster of cell glyphs with heterogeneous shading."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(0, 0.4, n)
    rr = rad * np.sqrt(rng.uniform(0.05, 1, n))
    base = mpl.colors.to_rgb(fc)
    for a, r0 in zip(ang, rr, strict=True):
        shade = np.clip(np.array(base) * rng.uniform(0.9, 1.05), 0, 1)
        cell(ax, cx + r0 * np.cos(a), cy + r0 * np.sin(a) * 0.8, rad * jitter, tuple(shade), ec=ec)


def lightning(ax: plt.Axes, x: float, y: float, s: float, color: str = ACCENT_PERT) -> None:
    """Perturbation glyph (a small lightning bolt)."""
    pts = np.array([[0, 1], [0.45, 1], [0.25, 0.55], [0.6, 0.55], [0.05, -0.1], [0.25, 0.4], [-0.1, 0.4]])
    ax.fill(x + pts[:, 0] * s, y + pts[:, 1] * s, color=color, lw=0)


def schematic_axes(ax: plt.Axes) -> None:
    """Blank 0-1 canvas for drawing."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def bars_with_points(ax: plt.Axes, table: dict[str, tuple[float, float, list[float]]],
                     order: list[str], ylabel: str = "", best_line: bool = True,
                     annotate: bool = True, width: float = 0.72, ylim: tuple[float, float] | None = None) -> None:
    """Cell-style bar plot: mean bar, s.d. error bar, per-fold dots, dashed next-best line
    and the relative improvement of CASTER over the next-best method."""
    xs = np.arange(len(order))
    means = [table[m][0] for m in order]
    for i, m in enumerate(order):
        mu, sd, pts = table[m]
        ax.bar(i, mu, width, color=COLORS.get(m, INK3), lw=0, zorder=2)
        ax.errorbar(i, mu, yerr=sd, color=INK, lw=0.5, capsize=1.2, capthick=0.5, zorder=3)
        if pts:
            jit = np.linspace(-0.18, 0.18, len(pts)) if len(pts) > 1 else [0]
            ax.scatter(i + np.array(jit), pts, s=2.5, color=INK, lw=0, zorder=4)
    ax.axhline(0, color=INK2, lw=0.5, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=45, ha="right", rotation_mode="anchor")
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    hgrid(ax)
    if best_line and "CASTER" in order and len(order) > 1:
        others = [table[m][0] for m in order if m not in ("CASTER", "NoChange")]
        if others:
            nb = max(others)
            ax.axhline(nb, color=INK2, lw=0.5, ls=(0, (2, 2)), zorder=1)
            c = table["CASTER"][0]
            if annotate and nb != 0:
                rel = (c - nb) / abs(nb) * 100
                ax.text(order.index("CASTER") + 0.45, max(c, nb) + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                        f"{rel:+.0f}%", fontsize=5, color=COLORS["CASTER"] if rel > 0 else INK2,
                        ha="left", va="bottom", fontweight="bold")
    _ = means
