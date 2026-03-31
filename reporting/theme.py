"""
Midnight Ocean theme — deep navy, teals, corals, and golds.

Provides:
  - A matplotlib style context manager: `with ocean_style():`
  - Named color palettes for consistent chart styling
  - Helper functions for gradient fills and glowing effects
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
from typing import List, Tuple

# ── Core palette ────────────────────────────────────────────────────────────────

# Primary series colours (cycle through for multi-strategy plots)
SERIES_COLORS = [
    "#4DD0E1",  # teal (primary)
    "#FF8A65",  # coral accent
    "#FFD54F",  # warm gold
    "#81C784",  # sea green
    "#4FC3F7",  # light ocean blue
    "#F48FB1",  # soft pink
    "#AED581",  # lime drift
    "#BA68C8",  # violet tide
]

# Named semantic colours
OCEAN_TEAL    = "#26C6DA"   # bright teal — primary accent
OCEAN_CORAL   = "#FF7043"   # warm coral — alerts / losses
OCEAN_BLUE    = "#1E88E5"   # deep ocean blue — neutral / info
OCEAN_CYAN    = "#4DD0E1"   # cyan — light accent
OCEAN_GOLD    = "#FFD54F"   # warm gold — secondary
OCEAN_SEA     = "#81C784"   # sea green — tertiary
OCEAN_DARK    = "#0A1628"   # deep midnight navy background
OCEAN_SURFACE = "#121E36"   # raised surface / panel
OCEAN_GRID    = "#1B3050"   # subtle grid lines
OCEAN_TEXT    = "#E0ECF4"   # high-contrast text on dark
OCEAN_MUTED   = "#6B8DAD"   # muted text / axis labels

# Semantic
WIN_COLOR  = "#81C784"   # sea green for positive
LOSS_COLOR = "#FF7043"   # coral for negative
NEUTRAL    = "#90A4AE"   # steel grey for zero / inactive

# Gradient helpers
GRADIENT_OCEAN = ["#0A1628", "#121E36", "#1B3050", "#26C6DA", "#4DD0E1"]
GRADIENT_CORAL = ["#121E36", "#4E342E", "#BF360C", "#FF7043", "#FF8A65"]

# Kernel-layer colours (for heatmaps, per-layer plots)
KERNEL_COLORS = {
    "LOB":         "#4DD0E1",  # teal
    "VPIN":        "#FF8A65",  # coral
    "Lambda":      "#BA68C8",  # violet
    "Hawkes":      "#F48FB1",  # pink
    "VannaCharm":  "#FFD54F",  # gold
    "GammaEx":     "#4FC3F7",  # sky blue
    "VRP":         "#81C784",  # sea green
    "MacroMotion": "#AED581",  # lime
    "Sentiment":   "#26C6DA",  # bright teal
    "EventProx":   "#FF7043",  # coral
}


# ── Matplotlib style context ───────────────────────────────────────────────────

_OCEAN_RC = {
    # Figure
    "figure.facecolor":   OCEAN_DARK,
    "figure.edgecolor":   OCEAN_DARK,
    "figure.figsize":     (14, 8),
    "figure.dpi":         120,

    # Axes
    "axes.facecolor":     OCEAN_SURFACE,
    "axes.edgecolor":     OCEAN_GRID,
    "axes.labelcolor":    OCEAN_TEXT,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.prop_cycle":    mpl.cycler(color=SERIES_COLORS),
    "axes.grid":          True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,

    # Grid
    "grid.color":         OCEAN_GRID,
    "grid.alpha":         0.4,
    "grid.linewidth":     0.5,

    # Ticks
    "xtick.color":        OCEAN_MUTED,
    "ytick.color":        OCEAN_MUTED,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,

    # Text
    "text.color":         OCEAN_TEXT,
    "font.family":        "sans-serif",

    # Legend
    "legend.facecolor":   OCEAN_SURFACE,
    "legend.edgecolor":   OCEAN_GRID,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,

    # Savefig
    "savefig.facecolor":  OCEAN_DARK,
    "savefig.edgecolor":  OCEAN_DARK,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.3,
}


@contextmanager
def ocean_style():
    """Context manager: apply Midnight Ocean theme for all plots inside."""
    with mpl.rc_context(_OCEAN_RC):
        yield


# Back-compat alias so charts.py imports work without edits
nyx_style = ocean_style


# ── Helper functions ────────────────────────────────────────────────────────────

def color_for_value(val: float, zero_color: str = NEUTRAL) -> str:
    """Return win/loss/neutral colour based on sign."""
    if val > 0:
        return WIN_COLOR
    elif val < 0:
        return LOSS_COLOR
    return zero_color


def gradient_fill(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    baseline: float = 0.0,
    color_above: str = OCEAN_CYAN,
    color_below: str = OCEAN_CORAL,
    alpha: float = 0.25,
    **kwargs,
):
    """Fill between y and baseline with different colours above/below."""
    ax.fill_between(x, baseline, y, where=y >= baseline,
                    color=color_above, alpha=alpha, **kwargs)
    ax.fill_between(x, baseline, y, where=y < baseline,
                    color=color_below, alpha=alpha, **kwargs)


def add_watermark(fig, text: str = "RKHS", alpha: float = 0.04):
    """Subtle watermark across the figure."""
    fig.text(0.5, 0.5, text, fontsize=80, fontweight="bold",
             color=OCEAN_TEAL, alpha=alpha,
             ha="center", va="center", rotation=30,
             transform=fig.transFigure)


def percentile_band_colors(n_bands: int = 5) -> List[str]:
    """Return a list of progressively transparent teals for MC fan bands."""
    base = np.array(mpl.colors.to_rgba(OCEAN_TEAL))
    colors = []
    for i in range(n_bands):
        c = base.copy()
        c[3] = 0.10 + 0.12 * i  # alpha ramps from faint to moderate
        colors.append(c)
    return colors


def format_pct(val: float, decimals: int = 1) -> str:
    """Format a fraction as a percentage string."""
    return f"{val * 100:.{decimals}f}%"


def format_dollars(val: float) -> str:
    """Format dollars with comma separator."""
    return f"${val:,.0f}"


# ── Back-compat aliases (so charts.py Nyx imports resolve to Ocean colours) ───

NYX_PURPLE   = OCEAN_TEAL     # primary accent
NYX_PINK     = OCEAN_CORAL    # alerts / losses
NYX_BLUE     = OCEAN_BLUE     # neutral / info
NYX_LAVENDER = OCEAN_CYAN     # light accent
NYX_ICE      = OCEAN_GOLD     # secondary
NYX_ORCHID   = OCEAN_SEA      # tertiary
NYX_DARK     = OCEAN_DARK     # background
NYX_SURFACE  = OCEAN_SURFACE  # panel
NYX_GRID     = OCEAN_GRID     # grid
NYX_TEXT     = OCEAN_TEXT      # text
NYX_MUTED    = OCEAN_MUTED    # muted
