"""
Backtesting Visualization Dashboard — Two-Level Architecture
==============================================================
Traders@SMU — Quantitative Strategies Group
Week 5: Backtesting Framework

Full two-level architecture:
    K_total = K_fast + β · K_fast · K_slow
    K_fast  = α₁·K_LOB + α₂·K_VPIN + α₃·K_Lambda + α₄·K_VannaCharm
              (dollar bar resolution)
    K_slow  = α₅·K_VRP + α₆·K_GEX + α₇·K_Sentiment
              (daily resolution)
    Gate    = Event Proximity (optional multiplicative 0/1 mask on K_total)
    Note    : K_Lambda subject to CKA redundancy gate vs K_VPIN (drop if CKA >= 0.5)

Generates publication-quality figures for:
    1.  Walk-forward fold structure (train/embargo/test)
    2.  OOS predictions vs actuals (two-level model)
    3.  s-score time series with entry/exit zones
    4.  Cumulative PnL curve with drawdown
    5.  OU parameter diagnostics (half-life, mean-reversion fit)
    6.  Per-fold performance comparison
    7.  CKA heatmap — ALL kernel layers (fast + slow + cross)
    8.  Effective dimensionality across ALL layers
    9.  Eigenvalue spectrum of Gram matrices
    10. CPCV / PBO distribution
    11. Hyperparameter CV surface
    12. Signal diagnostics — per-layer SNR + autocorrelation
    13. Architecture diagram — K_total two-level decomposition
    14. Fast vs slow decomposition — contribution timeseries + gate overlay
    15. Cross-level analysis — fast×slow interaction + MKL weights

Usage:
    python visualize_backtest.py          # runs full demo with synthetic data

    # Or import individual figure functions:
    from backtesting.visualize_backtest import fig_walk_forward_splits, fig_pnl_curve, ...

Style conventions match existing visualize.py / visualize_3d.py.
"""

import os
import sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Style: match existing visualize.py conventions
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# Layer color palette (shared with other viz modules)
C_LOB = "#27ae60"
C_VPIN = "#e67e22"
C_LAMBDA = "#8e44ad"
C_VANNA = "#e74c3c"
C_VRP = "#3498db"
C_GEX = "#9b59b6"
C_SENT = "#16a085"
C_COMBINED = "#2980b9"
C_ACCENT = "#1a5276"
C_RED = "#c0392b"
C_GREEN = "#27ae60"
C_GRAY = "#555555"
C_LIGHT_GRAY = "#bdc3c7"

# Walk-forward palette
C_TRAIN = "#3498db"
C_EMBARGO = "#e74c3c"
C_TEST = "#2ecc71"

FIG_DIR = os.path.join(_PARENT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {path}")


# ============================================================================
# 1. Walk-Forward Fold Structure
# ============================================================================

def fig_walk_forward_splits(
    backtest_result,
    n_total: int,
    embargo_bars: int = 50,
    title: str = "Walk-Forward Fold Structure",
    save_name: str = "wf_01_fold_structure.png",
):
    """
    Visualize train/embargo/test windows for each fold as horizontal bars.

    Shows the expanding training window, purge gap, and test region.
    """
    folds = backtest_result.fold_results
    n_folds = len(folds)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.7 * n_folds + 1)))

    for i, fr in enumerate(folds):
        y = n_folds - 1 - i  # top to bottom

        # Train bar
        ax.barh(y, fr.train_end - fr.train_start, left=fr.train_start,
                height=0.6, color=C_TRAIN, alpha=0.7, edgecolor="white")

        # Embargo bar
        emb_start = fr.train_end
        emb_end = fr.test_start
        ax.barh(y, emb_end - emb_start, left=emb_start,
                height=0.6, color=C_EMBARGO, alpha=0.5, edgecolor="white")

        # Test bar
        ax.barh(y, fr.test_end - fr.test_start, left=fr.test_start,
                height=0.6, color=C_TEST, alpha=0.7, edgecolor="white")

        ax.text(fr.train_start + 5, y, f"Train ({fr.train_end - fr.train_start})",
                va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(fr.test_start + 5, y, f"Test ({fr.test_end - fr.test_start})",
                va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {n_folds - 1 - i}" for i in range(n_folds)])
    ax.set_xlabel("Bar Index")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, n_total)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_TRAIN, alpha=0.7, label="Train (expanding)"),
        Patch(facecolor=C_EMBARGO, alpha=0.5, label=f"Embargo ({embargo_bars} bars)"),
        Patch(facecolor=C_TEST, alpha=0.7, label="Test (OOS)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 2. OOS Predictions vs Actuals
# ============================================================================

def fig_oos_predictions(
    backtest_result,
    save_name: str = "wf_02_oos_predictions.png",
):
    """
    Scatter plot of OOS predicted vs actual values, plus residual histogram.
    """
    y_pred = backtest_result.y_pred_oos
    y_true = backtest_result.y_true_oos
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel 1: Scatter ---
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, color=C_ACCENT, rasterized=True)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color=C_RED, lw=1.5, label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("OOS: Predicted vs Actual", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # R² annotation
    r2 = backtest_result.oos_r2
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Panel 2: Residual time series ---
    ax = axes[1]
    idx = backtest_result.oos_indices
    ax.plot(idx, residuals, lw=0.5, color=C_ACCENT, alpha=0.6)
    ax.axhline(0, color=C_RED, lw=1, ls="--")
    ax.fill_between(idx, residuals, 0, alpha=0.15, color=C_ACCENT)
    ax.set_xlabel("Bar Index")
    ax.set_ylabel("Residual (actual - pred)")
    ax.set_title("OOS Residuals Over Time", fontweight="bold")
    ax.grid(alpha=0.3)

    # --- Panel 3: Residual histogram ---
    ax = axes[2]
    ax.hist(residuals, bins=60, color=C_ACCENT, alpha=0.7, edgecolor="white",
            density=True)
    ax.axvline(0, color=C_RED, lw=1.5, ls="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.text(0.95, 0.92, f"mean={np.mean(residuals):.4f}\nstd={np.std(residuals):.4f}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 3. s-Score Time Series with Entry/Exit Zones
# ============================================================================

def fig_sscore_signals(
    s_scores: np.ndarray,
    positions: np.ndarray,
    prices: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    save_name: str = "wf_03_sscore_signals.png",
):
    """
    Three-panel figure: price, s-score with thresholds, and positions.
    """
    T = len(s_scores)
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})

    # --- Panel 1: Price series ---
    ax = axes[0]
    ax.plot(t, prices, lw=0.8, color=C_ACCENT)
    ax.set_ylabel("Price")
    ax.set_title("Price Series with Mean-Reversion Signals", fontweight="bold")
    ax.grid(alpha=0.3)

    # Color background by position
    for i in range(1, T):
        if positions[i] > 0.01:
            ax.axvspan(i - 1, i, alpha=0.08, color=C_GREEN)
        elif positions[i] < -0.01:
            ax.axvspan(i - 1, i, alpha=0.08, color=C_RED)

    # --- Panel 2: s-score ---
    ax = axes[1]
    valid = ~np.isnan(s_scores)
    ax.plot(t[valid], s_scores[valid], lw=0.7, color=C_ACCENT)
    ax.axhline(entry_threshold, color=C_RED, ls="--", lw=1, alpha=0.7,
               label=f"Entry (±{entry_threshold})")
    ax.axhline(-entry_threshold, color=C_RED, ls="--", lw=1, alpha=0.7)
    ax.axhline(exit_threshold, color=C_GREEN, ls=":", lw=1, alpha=0.7,
               label=f"Exit (±{exit_threshold})")
    ax.axhline(-exit_threshold, color=C_GREEN, ls=":", lw=1, alpha=0.7)
    ax.axhline(0, color=C_GRAY, lw=0.5)

    # Fill entry zones
    ax.fill_between(t, entry_threshold, s_scores,
                    where=(valid & (s_scores > entry_threshold)),
                    alpha=0.2, color=C_RED, label="Short zone")
    ax.fill_between(t, -entry_threshold, s_scores,
                    where=(valid & (s_scores < -entry_threshold)),
                    alpha=0.2, color=C_GREEN, label="Long zone")

    ax.set_ylabel("s-score")
    ax.set_title("Hilbert Space s-Score (OU Mean Reversion)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # --- Panel 3: Position ---
    ax = axes[2]
    ax.fill_between(t, 0, positions, where=(positions > 0),
                    color=C_GREEN, alpha=0.5, label="Long")
    ax.fill_between(t, 0, positions, where=(positions < 0),
                    color=C_RED, alpha=0.5, label="Short")
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.set_ylabel("Position")
    ax.set_xlabel("Bar Index")
    ax.set_title("Position Size", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-1.3, 1.3)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 4. Cumulative PnL with Drawdown
# ============================================================================

def fig_pnl_curve(
    pnl: np.ndarray,
    save_name: str = "wf_04_pnl_curve.png",
):
    """
    Cumulative PnL and underwater (drawdown) chart.
    """
    valid = ~np.isnan(pnl)
    pnl_clean = pnl[valid]
    cum_pnl = np.cumsum(pnl_clean)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - running_max

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # --- Cumulative PnL ---
    ax = axes[0]
    t = np.arange(len(cum_pnl))
    ax.plot(t, cum_pnl, lw=1.2, color=C_ACCENT)
    ax.fill_between(t, 0, cum_pnl, where=(cum_pnl >= 0),
                    alpha=0.15, color=C_GREEN)
    ax.fill_between(t, 0, cum_pnl, where=(cum_pnl < 0),
                    alpha=0.15, color=C_RED)
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Strategy Cumulative PnL", fontweight="bold")
    ax.grid(alpha=0.3)

    # Annotate final PnL and max drawdown
    ax.text(0.02, 0.92,
            f"Final PnL: {cum_pnl[-1]:.4f}\n"
            f"Max Drawdown: {drawdown.min():.4f}\n"
            f"Sharpe: {np.mean(pnl_clean) / max(np.std(pnl_clean), 1e-12) * np.sqrt(252):.2f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    # --- Drawdown ---
    ax = axes[1]
    ax.fill_between(t, 0, drawdown, color=C_RED, alpha=0.4)
    ax.plot(t, drawdown, lw=0.7, color=C_RED)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Bar Index")
    ax.set_title("Underwater Chart", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 5. OU Parameter Diagnostics
# ============================================================================

def fig_ou_diagnostics(
    distances: np.ndarray,
    ou_params,
    rolling_window: int = 100,
    save_name: str = "wf_05_ou_diagnostics.png",
):
    """
    OU fit diagnostics: distance series, AR(1) regression, and ACF.
    """
    from signal_definition import estimate_ou_params

    valid = ~np.isnan(distances)
    d = distances[valid]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel 1: Distance from mean embedding ---
    ax = axes[0]
    t = np.arange(len(d))
    ax.plot(t, d, lw=0.6, color=C_ACCENT, alpha=0.7)
    mu = ou_params.mu
    ax.axhline(mu, color=C_RED, ls="--", lw=1.5, label=f"OU μ = {mu:.3f}")
    ax.fill_between(t, mu - ou_params.sigma, mu + ou_params.sigma,
                    alpha=0.1, color=C_RED, label=f"±σ = {ou_params.sigma:.3f}")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Hilbert Distance")
    ax.set_title("Distance from Mean Embedding", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 2: AR(1) regression (dX vs X) ---
    ax = axes[1]
    dX = np.diff(d)
    X = d[:-1]
    ax.scatter(X, dX, s=3, alpha=0.2, color=C_ACCENT, rasterized=True)

    # OLS fit line
    A = np.column_stack([np.ones(len(X)), X])
    params_fit, _, _, _ = np.linalg.lstsq(A, dX, rcond=None)
    x_range = np.linspace(X.min(), X.max(), 100)
    ax.plot(x_range, params_fit[0] + params_fit[1] * x_range,
            color=C_RED, lw=2, label=f"slope = {params_fit[1]:.4f}")
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.set_xlabel("d(t)")
    ax.set_ylabel("Δd(t)")
    ax.set_title("AR(1) Regression: Δd vs d", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Annotation
    ax.text(0.05, 0.05,
            f"κ = {ou_params.kappa:.4f}\nt½ = {ou_params.half_life:.1f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # --- Panel 3: ACF ---
    ax = axes[2]
    max_lag = min(50, len(d) // 4)
    acf_vals = []
    d_centered = d - np.mean(d)
    c0 = np.sum(d_centered ** 2)
    for lag in range(max_lag + 1):
        if c0 > 0:
            if lag == 0:
                acf_vals.append(1.0)
            else:
                ck = np.sum(d_centered[:-lag] * d_centered[lag:])
                acf_vals.append(ck / c0)
        else:
            acf_vals.append(0)

    lags = np.arange(max_lag + 1)
    ax.bar(lags, acf_vals, width=0.8, color=C_ACCENT, alpha=0.7, edgecolor="white")
    # 95% CI band
    ci = 1.96 / np.sqrt(len(d))
    ax.axhline(ci, color=C_RED, ls="--", lw=1, alpha=0.5)
    ax.axhline(-ci, color=C_RED, ls="--", lw=1, alpha=0.5)
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation of Distances", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 6. Per-Fold Performance Comparison
# ============================================================================

def fig_per_fold_performance(
    backtest_result,
    save_name: str = "wf_06_per_fold.png",
):
    """
    Bar charts comparing key metrics across walk-forward folds.
    """
    folds = backtest_result.fold_results
    n = len(folds)
    fold_labels = [f"Fold {i}" for i in range(n)]

    # Compute per-fold metrics
    oos_mses = []
    train_losses = []
    w_norms = []
    for fr in folds:
        oos_mses.append(float(np.mean((fr.y_true - fr.y_pred) ** 2)))
        train_losses.append(fr.train_loss)
        w_norms.append(fr.weights_norm)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # --- OOS MSE ---
    ax = axes[0]
    colors = [C_GREEN if m < np.median(oos_mses) else C_RED for m in oos_mses]
    ax.bar(fold_labels, oos_mses, color=colors, alpha=0.7, edgecolor="white")
    ax.axhline(np.mean(oos_mses), color=C_ACCENT, ls="--", lw=1.5,
               label=f"Mean = {np.mean(oos_mses):.4f}")
    ax.set_ylabel("OOS MSE")
    ax.set_title("OOS Error by Fold", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Train vs OOS loss ---
    ax = axes[1]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w / 2, train_losses, w, color=C_TRAIN, alpha=0.7, label="Train")
    ax.bar(x + w / 2, oos_mses, w, color=C_TEST, alpha=0.7, label="OOS")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.set_ylabel("MSE")
    ax.set_title("Train vs OOS Error", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Weight norms ---
    ax = axes[2]
    ax.bar(fold_labels, w_norms, color=C_ACCENT, alpha=0.7, edgecolor="white")
    ax.set_ylabel("||w||₂")
    ax.set_title("KRR Weight Norm by Fold", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 7. CKA Heatmap — Kernel Layer Redundancy
# ============================================================================

def fig_cka_heatmap(
    gram_matrices: dict,
    save_name: str = "wf_07_cka_heatmap.png",
):
    """
    CKA heatmap showing pairwise redundancy between kernel layers.

    Parameters
    ----------
    gram_matrices : dict
        Mapping of layer names → (n, n) Gram matrices.
    """
    from metrics import centered_kernel_alignment

    names = list(gram_matrices.keys())
    n = len(names)
    cka_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cka_matrix[i, j] = centered_kernel_alignment(
                gram_matrices[names[i]], gram_matrices[names[j]]
            )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cka_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = cka_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("CKA: Kernel Layer Redundancy", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="CKA (0=orthogonal, 1=identical)")

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 8. Effective Dimensionality
# ============================================================================

def fig_effective_dimensionality(
    gram_matrices: dict,
    save_name: str = "wf_08_eff_dim.png",
):
    """
    Bar chart of effective dimensionality per kernel layer + eigenvalue spectra.
    """
    from metrics import effective_dimensionality

    names = list(gram_matrices.keys())
    n_layers = len(names)

    fig, axes = plt.subplots(1, 2, figsize=(max(12, n_layers * 1.5 + 4), 5))

    # --- Panel 1: Effective dim bars ---
    ax = axes[0]
    eff_dims = [effective_dimensionality(gram_matrices[name]) for name in names]
    _base_colors = [C_LOB, C_VPIN, C_LAMBDA, C_VANNA, C_VRP,
                    C_GEX, C_SENT, C_COMBINED, C_ACCENT, C_RED]
    layer_colors = [_base_colors[i % len(_base_colors)] for i in range(n_layers)]
    ax.bar(range(n_layers), eff_dims, color=layer_colors, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    for i, (name, ed) in enumerate(zip(names, eff_dims)):
        ax.text(i, ed + 0.3, f"{ed:.1f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_ylabel("Effective Dimensionality")
    ax.set_title("Eigenvalue Concentration by Layer", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: Eigenvalue spectra ---
    ax = axes[1]
    for i, name in enumerate(names):
        K = gram_matrices[name]
        eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
        eigs = np.maximum(eigs, 0)
        eigs_norm = eigs / max(eigs.sum(), 1e-12)
        ax.plot(eigs_norm[:50], lw=1.5, color=layer_colors[i],
                label=f"{name} (d_eff={eff_dims[i]:.1f})", alpha=0.8)

    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Normalized Eigenvalue")
    ax.set_title("Eigenvalue Spectrum (top 50)", fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 9. Eigenvalue Spectrum + PSD Verification
# ============================================================================

def fig_eigenvalue_psd(
    gram_matrices: dict,
    save_name: str = "wf_09_eigenvalues_psd.png",
):
    """
    Eigenvalue bar charts with PSD verification for each kernel.
    """
    names = list(gram_matrices.keys())
    n = len(names)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    _base_colors = [C_LOB, C_VPIN, C_LAMBDA, C_VANNA, C_VRP,
                    C_GEX, C_SENT, C_COMBINED, C_ACCENT, C_RED]
    layer_colors = _base_colors

    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        K = gram_matrices[name]
        eigs = np.sort(np.linalg.eigvalsh(K))
        min_eig = eigs[0]
        is_psd = min_eig >= -1e-10

        color = layer_colors[idx % len(layer_colors)]
        ax.bar(range(len(eigs)), eigs, color=color, alpha=0.7, edgecolor="white")
        ax.axhline(0, color=C_RED, lw=1, ls="--")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Index")
        ax.set_ylabel("Eigenvalue")

        status = "PSD" if is_psd else "NOT PSD"
        status_color = C_GREEN if is_psd else C_RED
        ax.text(0.95, 0.92, f"{status}\nλ_min={min_eig:.2e}",
                transform=ax.transAxes, ha="right", fontsize=9,
                fontweight="bold", color=status_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Eigenvalue Spectra & PSD Verification", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 10. CPCV / PBO Distribution
# ============================================================================

def fig_cpcv_pbo(
    pbo: float,
    details: dict,
    save_name: str = "wf_10_cpcv_pbo.png",
):
    """
    Visualize CPCV results: OOS rank of IS-best strategy across splits.
    """
    is_scores = details["is_scores"]
    oos_scores = details["oos_scores"]
    n_splits, n_strategies = is_scores.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel 1: IS-best strategy's OOS rank ---
    ax = axes[0]
    oos_ranks = []
    for s in range(n_splits):
        is_best = np.argmax(is_scores[s])
        oos_of_best = oos_scores[s, is_best]
        rank = np.sum(oos_scores[s] > oos_of_best)  # 0 = best
        oos_ranks.append(rank)

    ax.hist(oos_ranks, bins=max(n_strategies, 5), color=C_ACCENT, alpha=0.7,
            edgecolor="white")
    ax.axvline(n_strategies / 2, color=C_RED, ls="--", lw=1.5,
               label=f"Median rank = {n_strategies / 2:.0f}")
    ax.set_xlabel("OOS Rank of IS-Best Strategy")
    ax.set_ylabel("Count (splits)")
    ax.set_title("IS-Best Strategy OOS Rank Distribution", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # PBO annotation
    pbo_color = C_GREEN if pbo < 0.5 else C_RED
    ax.text(0.95, 0.92, f"PBO = {pbo:.3f}",
            transform=ax.transAxes, ha="right", fontsize=12, fontweight="bold",
            color=pbo_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    # --- Panel 2: IS vs OOS scores (each split as a point) ---
    ax = axes[1]
    for k in range(n_strategies):
        ax.scatter(is_scores[:, k], oos_scores[:, k], s=15, alpha=0.5,
                   label=f"Strategy {k}")
    # 45-degree line
    all_scores = np.concatenate([is_scores.ravel(), oos_scores.ravel()])
    lims = [all_scores.min(), all_scores.max()]
    ax.plot(lims, lims, "--", color=C_GRAY, lw=1)
    ax.set_xlabel("IS Score")
    ax.set_ylabel("OOS Score")
    ax.set_title("IS vs OOS Performance", fontweight="bold")
    if n_strategies <= 5:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 3: Strategy performance box plots ---
    ax = axes[2]
    bp = ax.boxplot(
        [oos_scores[:, k] for k in range(n_strategies)],
        tick_labels=[f"S{k}" for k in range(n_strategies)],
        patch_artist=True,
    )
    for i, patch in enumerate(bp["boxes"]):
        c = [C_ACCENT, C_RED, C_GREEN, C_VPIN, C_LAMBDA][i % 5]
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_ylabel("OOS Score")
    ax.set_title("OOS Score Distribution by Strategy", fontweight="bold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 11. Hyperparameter CV Surface
# ============================================================================

def fig_cv_surface(
    cv_result,
    save_name: str = "wf_11_cv_surface.png",
):
    """
    Heatmap of inner CV scores across (length_scale, reg_lambda) grid.
    """
    all_scores = cv_result.all_scores
    keys = sorted(all_scores.keys())

    length_scales = sorted(set(k[0] for k in keys))
    reg_lambdas = sorted(set(k[1] for k in keys))
    n_ls = len(length_scales)
    n_lam = len(reg_lambdas)

    score_grid = np.zeros((n_lam, n_ls))
    for i, lam in enumerate(reg_lambdas):
        for j, ls in enumerate(length_scales):
            score_grid[i, j] = all_scores.get((ls, lam), np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(score_grid, cmap="YlGnBu", aspect="auto",
                   origin="lower")

    ax.set_xticks(range(n_ls))
    ax.set_xticklabels([f"{ls:.2g}" for ls in length_scales])
    ax.set_yticks(range(n_lam))
    ax.set_yticklabels([f"{lam:.1e}" for lam in reg_lambdas])
    ax.set_xlabel("Length Scale (ℓ)")
    ax.set_ylabel("Regularization (λ)")
    ax.set_title("Inner CV Score: (ℓ, λ) Grid Search", fontweight="bold")

    # Annotate cells
    for i in range(n_lam):
        for j in range(n_ls):
            val = score_grid[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if val < np.nanmedian(score_grid) else "black")

    # Mark best
    best_ls = cv_result.best_length_scale
    best_lam = cv_result.best_reg_lambda
    if best_ls in length_scales and best_lam in reg_lambdas:
        bj = length_scales.index(best_ls)
        bi = reg_lambdas.index(best_lam)
        ax.plot(bj, bi, "r*", markersize=18, markeredgecolor="white",
                markeredgewidth=1.5)
        ax.text(bj + 0.3, bi + 0.3, "BEST", color=C_RED, fontsize=9,
                fontweight="bold")

    fig.colorbar(im, ax=ax, label="Score (higher = better)")
    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 12. Signal Diagnostics — Per-Layer SNR + Autocorrelation
# ============================================================================

def fig_signal_diagnostics(
    layer_results: dict,
    save_name: str = "wf_12_signal_diagnostics.png",
):
    """
    Per-layer signal quality: distance time series, SNR bars, ACF comparison.

    Parameters
    ----------
    layer_results : dict
        Output of SignalDiagnostics.per_layer_distances().
        Maps name → {'distances': array, 'snr_vs_combined': float}
    """
    names = list(layer_results.keys())
    n_layers = len(names)
    _base_colors = [C_LOB, C_VPIN, C_LAMBDA, C_VANNA, C_VRP,
                    C_GEX, C_SENT, C_COMBINED, C_ACCENT, C_RED]
    layer_colors = [_base_colors[i % len(_base_colors)] for i in range(n_layers)]

    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # --- Top-left: Distance time series per layer ---
    ax = fig.add_subplot(gs[0, 0])
    for i, name in enumerate(names):
        d = layer_results[name]["distances"]
        ax.plot(d, lw=0.6, color=layer_colors[i], alpha=0.7, label=name)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Hilbert Distance")
    ax.set_title("Per-Layer Distance from Mean Embedding", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Top-right: SNR bar chart ---
    ax = fig.add_subplot(gs[0, 1])
    snrs = [layer_results[name]["snr_vs_combined"] for name in names]
    snrs_display = [min(s, 100) for s in snrs]  # cap for display
    ax.bar(names, snrs_display, color=layer_colors, alpha=0.8, edgecolor="white")
    for i, s in enumerate(snrs):
        label = f"{s:.2f}" if s < 100 else f"{s:.1f}"
        ax.text(i, snrs_display[i] + 0.05, label, ha="center", fontsize=9,
                fontweight="bold")
    ax.set_ylabel("SNR (Var_signal / Var_noise)")
    ax.set_title("Signal-to-Noise Ratio by Layer", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- Bottom-left: ACF comparison ---
    ax = fig.add_subplot(gs[1, 0])
    max_lag = 30
    for i, name in enumerate(names):
        d = layer_results[name]["distances"]
        d_c = d - np.mean(d)
        c0 = np.sum(d_c ** 2)
        acf_vals = []
        for lag in range(max_lag + 1):
            if c0 > 0 and lag > 0:
                ck = np.sum(d_c[:-lag] * d_c[lag:])
                acf_vals.append(ck / c0)
            elif lag == 0:
                acf_vals.append(1.0)
            else:
                acf_vals.append(0.0)
        ax.plot(range(max_lag + 1), acf_vals, lw=1.5, color=layer_colors[i],
                label=name, alpha=0.8)

    ci = 1.96 / np.sqrt(len(layer_results[names[0]]["distances"]))
    ax.axhline(ci, color=C_RED, ls="--", lw=1, alpha=0.5)
    ax.axhline(-ci, color=C_RED, ls="--", lw=1, alpha=0.5)
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title("Autocorrelation by Layer", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Bottom-right: Summary stats table ---
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    cell_text = []
    for name in names:
        d = layer_results[name]["distances"]
        snr = layer_results[name]["snr_vs_combined"]
        cell_text.append([
            name,
            f"{np.mean(d):.3f}",
            f"{np.std(d):.3f}",
            f"{snr:.2f}",
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Layer", "Mean Dist", "Std Dist", "SNR"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header
    for j in range(4):
        table[(0, j)].set_facecolor(C_ACCENT)
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Layer Summary", fontweight="bold", pad=20)

    _save(fig, save_name)
    return fig


# ============================================================================
# 13. Architecture Diagram — Two-Level K_total Decomposition
# ============================================================================

def fig_architecture_diagram(
    alpha_fast: np.ndarray,
    alpha_slow: np.ndarray,
    beta: float,
    fast_names: list,
    slow_names: list,
    save_name: str = "wf_13_architecture.png",
):
    """
    Visual diagram of the two-level kernel architecture showing
    K_total = K_fast + β · K_fast · K_slow with learned weights.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def _box(x, y, w, h, text, color, fontsize=10, alpha=0.8):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", alpha=alpha, linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    def _arrow(x1, y1, x2, y2, label="", color=C_GRAY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my + 0.15, label, fontsize=8, color=color,
                    fontstyle="italic")

    # --- Title ---
    ax.text(7, 6.6, r"$K_{total} = K_{fast} + \beta \cdot K_{fast} \cdot K_{slow}$",
            ha="center", fontsize=14, fontweight="bold", color=C_ACCENT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf2f8", alpha=0.9))

    # --- K_total ---
    _box(5.5, 5.2, 3, 0.8, "K_total (combined)", C_ACCENT, fontsize=11)

    # --- K_fast and K_slow ---
    _box(1.5, 3.5, 3, 0.8, f"K_fast (bar-level)", C_TRAIN, fontsize=10)
    _box(9.5, 3.5, 3, 0.8, f"K_slow (daily)", "#d35400", fontsize=10)

    # Product term
    _box(5.5, 3.5, 3, 0.8, f"β={beta:.2f} · K_fast · K_slow", C_RED, fontsize=9)

    # Arrows: K_fast → K_total, product → K_total
    _arrow(3.0, 4.3, 5.8, 5.2, "+")
    _arrow(7.0, 4.3, 7.0, 5.2, "+")
    # Arrows: K_fast → product, K_slow → product
    _arrow(4.5, 3.9, 5.5, 3.9, "")
    _arrow(9.5, 3.9, 8.5, 3.9, "")

    # --- Gate ---
    _box(5.5, 2.4, 3, 0.6, "Event Gate (0/1 mask)", C_LIGHT_GRAY, fontsize=9)
    _arrow(7.0, 3.0, 7.0, 3.5, "×", C_GRAY)

    # --- Fast sub-kernels ---
    n_fast = len(fast_names)
    fast_colors = [C_LOB, C_VPIN, C_LAMBDA, C_VANNA][:n_fast]
    fast_width = 6.0  # total width for fast sub-kernels
    fast_box_w = min(1.6, fast_width / n_fast - 0.1)
    fast_spacing = fast_width / n_fast
    x_start_fast = 0.2
    for i, (name, alpha) in enumerate(zip(fast_names, alpha_fast)):
        x = x_start_fast + i * fast_spacing
        c = fast_colors[i % len(fast_colors)]
        _box(x, 1.5, fast_box_w, 0.7, f"{name}\nα={alpha:.2f}", c, fontsize=8)
        _arrow(x + fast_box_w / 2, 2.2, 3.0, 3.5)

    # --- Slow sub-kernels ---
    n_slow = len(slow_names)
    slow_colors = [C_VRP, C_GEX, C_SENT][:n_slow]
    x_start = 7.8
    for i, (name, alpha) in enumerate(zip(slow_names, alpha_slow)):
        x = x_start + i * 2.0
        c = slow_colors[i % len(slow_colors)]
        _box(x, 1.5, 1.8, 0.7, f"{name}\nα={alpha:.2f}", c, fontsize=8)
        _arrow(x + 0.9, 2.2, 11.0, 3.5)

    # --- Resolution labels ---
    ax.text(3.0, 0.8, "Dollar Bars (event-time)",
            ha="center", fontsize=9, color=C_TRAIN, fontstyle="italic")
    ax.text(11.0, 0.8, "Daily (calendar-time)",
            ha="center", fontsize=9, color="#d35400", fontstyle="italic")

    # Divider
    ax.plot([6.8, 6.8], [0.5, 2.5], "--", color=C_LIGHT_GRAY, lw=1, alpha=0.5)
    ax.text(7.0, 0.5, "resolution boundary", fontsize=7, color=C_LIGHT_GRAY,
            ha="center", fontstyle="italic")

    ax.set_title("Multi-Layer RKHS Kernel Architecture", fontsize=14,
                 fontweight="bold", pad=20)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 14. Fast vs Slow Decomposition with Gate Overlay
# ============================================================================

def fig_fast_slow_decomposition(
    fast_distances: np.ndarray,
    slow_distances_aligned: np.ndarray,
    combined_distances: np.ndarray,
    gate_mask: np.ndarray,
    prices: np.ndarray,
    save_name: str = "wf_14_fast_slow_decomposition.png",
):
    """
    Four-panel figure showing fast signal, slow signal, combined, and gate.

    Parameters
    ----------
    fast_distances : array (n_bars,)
        Hilbert distances from K_fast mean embedding.
    slow_distances_aligned : array (n_bars,)
        Slow distances aligned to bar resolution via MultiResolutionAligner.
    combined_distances : array (n_bars,)
        Distances from the combined K_total model.
    gate_mask : array (n_bars,)
        Event proximity mask (0 or 1 per bar).
    prices : array (n_bars,)
    """
    T = len(fast_distances)
    t = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5, 0.8]})

    # --- Panel 1: Price with gate shading ---
    ax = axes[0]
    ax.plot(t, prices[:T], lw=0.8, color=C_ACCENT)
    # Shade gated-off regions
    for i in range(1, T):
        if gate_mask[i] < 0.5:
            ax.axvspan(i - 1, i, alpha=0.07, color=C_RED)
    ax.set_ylabel("Price")
    ax.set_title("Price Series — red shading = gate OFF (near events)", fontweight="bold")
    ax.grid(alpha=0.3)

    # --- Panel 2: Fast vs Slow distances ---
    ax = axes[1]
    ax.plot(t, fast_distances, lw=0.6, color=C_TRAIN, alpha=0.8, label="K_fast (bar)")
    ax.plot(t, slow_distances_aligned, lw=1.2, color="#d35400", alpha=0.8,
            label="K_slow (daily, aligned)")
    ax.set_ylabel("Hilbert Distance")
    ax.set_title("Fast vs Slow Layer Distances from Mean Embedding", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # --- Panel 3: Combined distance ---
    ax = axes[2]
    ax.plot(t, combined_distances, lw=0.7, color=C_ACCENT, label="K_total (combined)")
    # Show what it would be without the gate
    ungated = combined_distances.copy()
    gated = combined_distances * gate_mask
    ax.fill_between(t, gated, combined_distances, where=(gate_mask < 0.5),
                    alpha=0.3, color=C_RED, label="Gated out")
    ax.set_ylabel("Combined Distance")
    ax.set_title("Combined K_total Distance (with gate effect)", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # --- Panel 4: Gate mask ---
    ax = axes[3]
    ax.fill_between(t, 0, gate_mask, color=C_GREEN, alpha=0.5, step="mid")
    ax.fill_between(t, 0, 1 - gate_mask, color=C_RED, alpha=0.3, step="mid")
    ax.set_ylabel("Gate")
    ax.set_xlabel("Bar Index")
    ax.set_title("Event Proximity Gate (1=active, 0=suppressed near events)",
                 fontweight="bold")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["OFF", "ON"])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, save_name)
    return fig


# ============================================================================
# 15. Cross-Level Analysis — Fast×Slow Interaction + MKL Weights
# ============================================================================

def fig_cross_level_analysis(
    gram_fast_subs: dict,
    gram_slow_subs: dict,
    gram_fast: np.ndarray,
    gram_slow: np.ndarray,
    gram_total: np.ndarray,
    alpha_fast: np.ndarray,
    alpha_slow: np.ndarray,
    beta: float,
    fast_names: list,
    slow_names: list,
    save_name: str = "wf_15_cross_level.png",
):
    """
    Cross-level diagnostics: CKA between fast and slow sub-kernels,
    MKL weight bar chart, and K_total composition pie chart.
    """
    from metrics import centered_kernel_alignment, effective_dimensionality

    all_names = fast_names + slow_names
    all_grams = {**gram_fast_subs, **gram_slow_subs}
    n = len(all_names)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # --- Panel 1: Full cross-level CKA (top-left, spans 2 cols) ---
    ax = fig.add_subplot(gs[0, :2])
    cka_mat = np.zeros((n, n))
    for i, ni in enumerate(all_names):
        for j, nj in enumerate(all_names):
            cka_mat[i, j] = centered_kernel_alignment(all_grams[ni], all_grams[nj])

    im = ax.imshow(cka_mat, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(all_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = cka_mat[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

    # Draw border between fast and slow
    n_fast = len(fast_names)
    ax.axhline(n_fast - 0.5, color="white", lw=3)
    ax.axvline(n_fast - 0.5, color="white", lw=3)
    ax.text(n_fast / 2 - 0.5, -0.8, "FAST", fontsize=10, color=C_TRAIN,
            fontweight="bold", ha="center")
    ax.text(n_fast + len(slow_names) / 2 - 0.5, -0.8, "SLOW", fontsize=10,
            color="#d35400", fontweight="bold", ha="center")

    ax.set_title("Cross-Level CKA: Fast × Slow Sub-Kernels", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # --- Panel 2: MKL weights (top-right) ---
    ax = fig.add_subplot(gs[0, 2])
    all_weights = list(alpha_fast) + list(alpha_slow) + [beta]
    all_labels = [f"α({n})" for n in fast_names] + \
                 [f"α({n})" for n in slow_names] + ["β (cross)"]
    _fast_colors = [C_LOB, C_VPIN, C_LAMBDA, C_VANNA][:len(fast_names)]
    _slow_colors = [C_VRP, C_GEX, C_SENT][:len(slow_names)]
    all_colors = _fast_colors + _slow_colors + [C_RED]

    bars = ax.barh(range(len(all_weights)), all_weights,
                   color=all_colors, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(all_weights)))
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_xlabel("Weight")
    ax.set_title("MKL Weights", fontweight="bold")
    for i, v in enumerate(all_weights):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # --- Panel 3: Effective dim comparison (bottom-left) ---
    ax = fig.add_subplot(gs[1, 0])
    ed_labels = list(all_names) + ["K_fast", "K_slow", "K_total"]
    ed_grams = list(all_grams.values()) + [gram_fast, gram_slow, gram_total]
    ed_vals = [effective_dimensionality(K) for K in ed_grams]
    ed_colors = ([C_LOB, C_VPIN, C_LAMBDA, C_VANNA][:len(fast_names)] +
                 [C_VRP, C_GEX, C_SENT][:len(slow_names)] +
                 [C_TRAIN, "#d35400", C_ACCENT])

    ax.bar(range(len(ed_vals)), ed_vals, color=ed_colors, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(len(ed_vals)))
    ax.set_xticklabels(ed_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Effective Dim")
    ax.set_title("Effective Dimensionality: All Layers", fontweight="bold")
    for i, v in enumerate(ed_vals):
        ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 4: CKA between K_fast, K_slow, K_total (bottom-center) ---
    ax = fig.add_subplot(gs[1, 1])
    level_names = ["K_fast", "K_slow", "K_total"]
    level_grams = [gram_fast, gram_slow, gram_total]
    level_cka = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            level_cka[i, j] = centered_kernel_alignment(level_grams[i], level_grams[j])

    im2 = ax.imshow(level_cka, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(level_names, fontsize=10)
    ax.set_yticklabels(level_names, fontsize=10)
    for i in range(3):
        for j in range(3):
            val = level_cka[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    ax.set_title("CKA: Fast vs Slow vs Total", fontweight="bold")
    fig.colorbar(im2, ax=ax, shrink=0.8)

    # --- Panel 5: Composition breakdown (bottom-right) ---
    ax = fig.add_subplot(gs[1, 2])

    # Frobenius norm contribution of each term to K_total
    norm_fast = np.linalg.norm(gram_fast, "fro")
    norm_product = beta * np.linalg.norm(gram_fast * gram_slow, "fro")
    total_norm = norm_fast + norm_product

    if total_norm > 0:
        fracs = [norm_fast / total_norm, norm_product / total_norm]
    else:
        fracs = [0.5, 0.5]

    wedges, texts, autotexts = ax.pie(
        fracs, labels=["K_fast\n(additive)", "β·K_fast·K_slow\n(multiplicative)"],
        colors=[C_TRAIN, C_RED], autopct="%1.1f%%",
        startangle=90, textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_color("white")
    ax.set_title("K_total Composition\n(Frobenius norm share)", fontweight="bold")

    _save(fig, save_name)
    return fig


# ============================================================================
# Master Dashboard (runs all figures from synthetic data)
# ============================================================================

def run_dashboard():
    """
    Generate all 15 figures using the full two-level architecture:
        K_total = K_fast + β · K_fast · K_slow
        K_fast  = α₁·K_LOB + α₂·K_VPIN + α₃·K_Lambda + α₄·K_VannaCharm
                  (dollar bar resolution)
        K_slow  = α₅·K_VRP + α₆·K_GEX + α₇·K_Sentiment
                  (daily resolution)
        Gate    = Event Proximity (optional 0/1 mask on K_total)

    K_Lambda is subject to a CKA redundancy gate against K_VPIN;
    if CKA >= 0.5, K_Lambda is dropped from the fast layer.
    """
    from lob_kernel import (
        generate_synthetic_lob,
        VolumeProfileKernel,
        BookShapeKernel,
        DepthImbalanceKernel,
        RKHSLayer,
    )
    from walk_forward import (
        PurgedWalkForward, CPCVEvaluator,
        MultiResolutionAligner, TwoLevelKernelCombiner,
        CKARedundancyGate,
    )
    from signal_definition import (
        OUSignalGenerator, generate_positions, SignalDiagnostics,
    )
    from hyperparameter_cv import PurgedKFold, inner_cv_grid_search
    from test_pipeline import (
        generate_ou_price_series,
        generate_synthetic_lob_from_prices,
        generate_synthetic_daily_features,
        SyntheticFeatureKernel,
    )

    print("\n" + "=" * 60)
    print("Two-Level RKHS Backtesting Dashboard")
    print("K_total = K_fast + β · K_fast · K_slow")
    print("K_fast  = LOB + VPIN + Lambda(CKA-gated) + VannaCharm")
    print("K_slow  = VRP + GEX + Sentiment")
    print("=" * 60 + "\n")

    np.random.seed(42)

    # ================================================================
    # Generate synthetic data for all layers
    # ================================================================
    print("  Generating synthetic data (all layers)...")
    n_bars = 3000
    n_days = 150
    bars_per_day = n_bars // n_days
    true_kappa = 0.05

    prices = generate_ou_price_series(
        n=n_bars, kappa=true_kappa, mu=100.0, sigma=0.3,
    )
    snapshots = generate_synthetic_lob_from_prices(prices, n_levels=5)
    returns = np.diff(prices) / prices[:-1]
    y = returns

    bar_timestamps = np.arange(n_bars, dtype=np.float64)
    daily_timestamps = np.arange(0, n_bars, bars_per_day, dtype=np.float64)

    # ================================================================
    # FAST LAYER: K_fast = α₁·K_LOB + α₂·K_VPIN + α₃·K_Lambda
    #             + α₄·K_VannaCharm  (bar resolution)
    # ================================================================
    print("  Building fast layer (K_LOB + K_VPIN + K_Lambda + K_VannaCharm)...")

    # K_LOB sub-kernels
    vol_k = VolumeProfileKernel(
        n_levels=5, sigma=0.5, use_rff=True, nu=1.5, n_rff=200,
    )
    shape_k = BookShapeKernel(
        n_levels=5, sigma="auto", use_rff=True, nu=1.5, n_rff=200,
    )
    depth_k = DepthImbalanceKernel(
        n_levels=5, sigma=0.8, use_rff=True, nu=1.5, n_rff=200,
    )
    lob_layer = RKHSLayer(
        kernels=[vol_k, shape_k, depth_k],
        weights=[1.0, 1.0, 0.8],
        name="K_LOB",
    )

    # K_VPIN — synthetic 4D features (VPIN, change, EWMA, duration)
    rng = np.random.RandomState(44)
    vpin_features = np.zeros((n_bars, 4))
    vpin_features[0] = [0.3, 0.0, 0.3, 1.0]
    for t in range(1, n_bars):
        vpin_features[t, 0] = np.clip(
            0.95 * vpin_features[t - 1, 0] + 0.02 * rng.randn(), 0, 1
        )
        vpin_features[t, 1] = vpin_features[t, 0] - vpin_features[t - 1, 0]
        vpin_features[t, 2] = 0.9 * vpin_features[t - 1, 2] + 0.1 * vpin_features[t, 0]
        vpin_features[t, 3] = np.clip(1.0 + 0.3 * rng.randn(), 0.1, 5.0)
    vpin_kernel = SyntheticFeatureKernel(
        feature_dim=4, n_rff=200, seed=44, kernel_name="K_VPIN",
    )

    # K_Lambda — synthetic Kyle's Lambda features
    # (price impact slope, volume imbalance, trade sign autocorrelation)
    rng_lam = np.random.RandomState(48)
    lambda_features = np.zeros((n_bars, 3))
    lambda_features[0] = [0.5, 0.0, 0.1]
    for t in range(1, n_bars):
        lambda_features[t, 0] = np.clip(
            0.92 * lambda_features[t - 1, 0] + 0.04 * rng_lam.randn(), 0, 2
        )
        lambda_features[t, 1] = 0.85 * lambda_features[t - 1, 1] + 0.15 * rng_lam.randn()
        lambda_features[t, 2] = np.clip(
            0.9 * lambda_features[t - 1, 2] + 0.05 * rng_lam.randn(), -1, 1
        )
    lambda_kernel = SyntheticFeatureKernel(
        feature_dim=3, n_rff=200, seed=48, kernel_name="K_Lambda",
    )

    # K_VannaCharm — synthetic Vanna/Charm features
    # (vanna exposure, charm decay, delta hedging pressure)
    rng_vc = np.random.RandomState(49)
    vanna_features = np.zeros((n_bars, 3))
    vanna_features[0] = [0.0, -0.1, 0.2]
    for t in range(1, n_bars):
        vanna_features[t, 0] = 0.88 * vanna_features[t - 1, 0] + 0.08 * rng_vc.randn()
        vanna_features[t, 1] = -0.05 + 0.9 * (vanna_features[t - 1, 1] + 0.05) + 0.03 * rng_vc.randn()
        vanna_features[t, 2] = 0.93 * vanna_features[t - 1, 2] + 0.05 * rng_vc.randn()
    vanna_kernel = SyntheticFeatureKernel(
        feature_dim=3, n_rff=200, seed=49, kernel_name="K_VannaCharm",
    )

    # --- CKA redundancy gate: check Lambda vs VPIN ---
    print("  Running CKA redundancy gate (K_Lambda vs K_VPIN)...")
    cka_gate = CKARedundancyGate(threshold=0.5, n_sample=200, seed=42)

    # Build candidate fast kernels and names (all 4)
    all_fast_kernels = [lob_layer, vpin_kernel, lambda_kernel, vanna_kernel]
    all_fast_names = ["K_LOB", "K_VPIN", "K_Lambda", "K_VannaCharm"]
    all_fast_data = [snapshots, vpin_features, lambda_features, vanna_features]

    # Check Lambda (idx 2) vs VPIN (idx 1) — each uses its own features
    keep_lambda, cka_val = cka_gate.check(
        lambda_kernel, vpin_kernel,
        candidate_data=lambda_features,
        reference_data=vpin_features,
    )
    print(f"    CKA(K_Lambda, K_VPIN) = {cka_val:.3f}, threshold = 0.5")
    print(f"    Keep K_Lambda: {keep_lambda}")

    # Build final fast kernel list (Lambda kept or dropped)
    if keep_lambda:
        fast_kernels = all_fast_kernels
        fast_names = all_fast_names
        fast_data_list = all_fast_data
        alpha_fast = np.array([0.35, 0.25, 0.20, 0.20])
    else:
        fast_kernels = [lob_layer, vpin_kernel, vanna_kernel]
        fast_names = ["K_LOB", "K_VPIN", "K_VannaCharm"]
        fast_data_list = [snapshots, vpin_features, vanna_features]
        alpha_fast = np.array([0.40, 0.30, 0.30])

    print(f"    Fast layer: {fast_names}")
    print(f"    α_fast: {alpha_fast}")

    # ================================================================
    # SLOW LAYER: K_slow = α₅·K_VRP + α₆·K_GEX + α₇·K_Sentiment (daily)
    # ================================================================
    print("  Building slow layer (K_VRP + K_GEX + K_Sentiment)...")

    daily_features = generate_synthetic_daily_features(
        n_days, n_features=9, seed=43,
    )  # 9 features → 3 per slow kernel

    # Three slow sub-kernels operating on different feature subsets
    vrp_kernel = SyntheticFeatureKernel(
        feature_dim=3, n_rff=150, seed=45, kernel_name="K_VRP",
    )
    gex_kernel = SyntheticFeatureKernel(
        feature_dim=3, n_rff=150, seed=46, kernel_name="K_GEX",
    )
    sent_kernel = SyntheticFeatureKernel(
        feature_dim=3, n_rff=150, seed=47, kernel_name="K_Sentiment",
    )

    alpha_slow = np.array([0.4, 0.35, 0.25])
    slow_names = ["K_VRP", "K_GEX", "K_Sentiment"]

    # Split daily features across slow kernels
    vrp_data = daily_features[:, 0:3]
    gex_data = daily_features[:, 3:6]
    sent_data = daily_features[:, 6:9]

    # ================================================================
    # EVENT PROXIMITY GATE (optional 0/1 mask on K_total)
    # ================================================================
    print("  Building event proximity gate (applied to K_total)...")

    # Simulate: gate OFF for 5 bars around synthetic "events" every ~100 bars
    gate_mask = np.ones(n_bars)
    event_bars = np.arange(50, n_bars, 100)  # events every 100 bars
    proximity_window = 5
    for eb in event_bars:
        start = max(0, eb - proximity_window)
        end = min(n_bars, eb + proximity_window + 1)
        gate_mask[start:end] = 0.0

    n_events = len(event_bars)
    pct_gated = 100.0 * (1 - gate_mask.mean())
    print(f"    {n_events} events, {pct_gated:.1f}% of bars gated off")

    # ================================================================
    # TWO-LEVEL COMBINER: K_total = K_fast + β · K_fast · K_slow
    # ================================================================
    print("  Assembling two-level combiner...")
    beta = 0.8

    # Build combined fast feature matrix (LOB + VPIN + Lambda + VannaCharm)
    lob_raw = np.array([vol_k.extract_features(s) for s in snapshots])  # (n, 10)
    fast_raw_parts = [lob_raw, vpin_features]
    if keep_lambda:
        fast_raw_parts.append(lambda_features)
    fast_raw_parts.append(vanna_features)
    X_fast_combined = np.concatenate(fast_raw_parts, axis=1)  # (n, 10+4+3+3=20)

    fast_combined = SyntheticFeatureKernel(
        feature_dim=X_fast_combined.shape[1],
        n_rff=250, seed=42, kernel_name="K_fast_combined",
    )
    slow_combined = SyntheticFeatureKernel(
        feature_dim=9,  # VRP(3D) + GEX(3D) + Sentiment(3D) combined raw
        n_rff=200, seed=43, kernel_name="K_slow_combined",
    )

    aligner = MultiResolutionAligner(bar_timestamps, daily_timestamps)
    combiner = TwoLevelKernelCombiner(
        fast_layer=fast_combined,
        slow_layer=slow_combined,
        aligner=aligner,
        beta=beta,
        product_dim=150,
        event_gate=lambda idx: float(gate_mask[min(idx, n_bars - 1)]),
        seed=42,
    )

    # ================================================================
    # RUN TWO-LEVEL WALK-FORWARD BACKTEST
    # ================================================================
    print("  Running two-level walk-forward backtest...")
    engine = PurgedWalkForward(
        n_splits=5, embargo_bars=30, min_train_bars=500,
        decay_half_life=1000.0, reg_lambda=1e-2,
    )
    result = engine.run_two_level(
        fast_data=X_fast_combined[:-1],
        slow_data=daily_features,
        y=y,
        combiner=combiner,
    )

    # ================================================================
    # SIGNAL GENERATION on combined features
    # ================================================================
    print("  Generating signals from combined model...")
    phi_fast = fast_combined.feature_map(X_fast_combined)     # (n, D_fast)
    phi_slow = slow_combined.feature_map(daily_features)      # (n_days, D_slow)
    # Align slow to bars
    daily_idx = aligner.get_daily_indices(np.arange(n_bars))
    phi_slow_aligned = phi_slow[daily_idx]                    # (n, D_slow)

    # Combined feature map (approximate — without full tensor sketch for viz)
    phi_combined = np.concatenate([
        phi_fast,
        np.sqrt(beta) * phi_fast[:, :min(phi_fast.shape[1], phi_slow_aligned.shape[1])]
        * phi_slow_aligned[:, :min(phi_fast.shape[1], phi_slow_aligned.shape[1])],
    ], axis=1)

    gen = OUSignalGenerator(rolling_window=100, ou_window=500)
    signal_combined = gen.generate(phi_combined, dt=1.0)
    signal_fast = gen.generate(phi_fast, dt=1.0)
    signal_slow = gen.generate(phi_slow_aligned, dt=1.0)

    positions = generate_positions(
        signal_combined.s_scores, entry_threshold=1.5, exit_threshold=0.3,
    )

    # ================================================================
    # COMPUTE GRAM MATRICES for all layers (on subset for viz)
    # ================================================================
    print("  Computing Gram matrices for all layers...")
    n_sub = 200
    sub_snaps = snapshots[:n_sub]
    sub_vpin = vpin_features[:n_sub]
    sub_lambda = lambda_features[:n_sub]
    sub_vanna = vanna_features[:n_sub]
    sub_vrp = vrp_data[:min(n_sub // bars_per_day + 1, n_days)]
    sub_gex = gex_data[:min(n_sub // bars_per_day + 1, n_days)]
    sub_sent = sent_data[:min(n_sub // bars_per_day + 1, n_days)]

    # Fast sub-kernel Grams
    K_lob = lob_layer.gram_matrix(sub_snaps)
    phi_vpin_sub = vpin_kernel.feature_map(sub_vpin)
    K_vpin = phi_vpin_sub @ phi_vpin_sub.T
    phi_lambda_sub = lambda_kernel.feature_map(sub_lambda)
    K_lambda = phi_lambda_sub @ phi_lambda_sub.T
    phi_vanna_sub = vanna_kernel.feature_map(sub_vanna)
    K_vanna = phi_vanna_sub @ phi_vanna_sub.T

    # Slow sub-kernel Grams (at daily resolution, then align)
    # Align slow Grams to bar resolution for cross-level CKA
    sub_daily_idx = aligner.get_daily_indices(np.arange(n_sub))
    sub_daily_idx = np.clip(sub_daily_idx, 0, len(sub_vrp) - 1)
    phi_vrp_bar = vrp_kernel.feature_map(sub_vrp)[sub_daily_idx]
    phi_gex_bar = gex_kernel.feature_map(sub_gex)[sub_daily_idx]
    phi_sent_bar = sent_kernel.feature_map(sub_sent)[sub_daily_idx]
    K_vrp = phi_vrp_bar @ phi_vrp_bar.T
    K_gex = phi_gex_bar @ phi_gex_bar.T
    K_sent = phi_sent_bar @ phi_sent_bar.T

    # Combined layer Grams (dynamic based on which fast kernels survived gate)
    gram_fast_subs = {"K_LOB": K_lob, "K_VPIN": K_vpin}
    if keep_lambda:
        gram_fast_subs["K_Lambda"] = K_lambda
    gram_fast_subs["K_VannaCharm"] = K_vanna

    K_fast = sum(
        alpha_fast[i] * list(gram_fast_subs.values())[i]
        for i in range(len(alpha_fast))
    )

    gram_slow_subs = {"K_VRP": K_vrp, "K_GEX": K_gex, "K_Sentiment": K_sent}
    K_slow = alpha_slow[0] * K_vrp + alpha_slow[1] * K_gex + alpha_slow[2] * K_sent
    K_total = K_fast + beta * (K_fast * K_slow)  # Schur product

    gram_all = {
        **gram_fast_subs, **gram_slow_subs,
        "K_fast": K_fast, "K_slow": K_slow, "K_total": K_total,
    }

    # ================================================================
    # GENERATE ALL 15 FIGURES
    # ================================================================

    # --- Fig 1: Walk-forward fold structure ---
    print("\n  Figure 1: Walk-forward fold structure")
    fig_walk_forward_splits(result, n_total=n_bars - 1, embargo_bars=30)

    # --- Fig 2: OOS predictions (from two-level model) ---
    print("  Figure 2: OOS predictions vs actuals (two-level)")
    fig_oos_predictions(result)

    # --- Fig 3: s-score signals (from combined K_total) ---
    print("  Figure 3: s-score signals (K_total combined)")
    fig_sscore_signals(signal_combined.s_scores, positions, prices)

    # --- Fig 4: PnL curve ---
    print("  Figure 4: Cumulative PnL + drawdown")
    pnl = positions[:-1] * returns
    fig_pnl_curve(pnl)

    # --- Fig 5: OU diagnostics ---
    print("  Figure 5: OU parameter diagnostics")
    fig_ou_diagnostics(signal_combined.distances, signal_combined.ou_params)

    # --- Fig 6: Per-fold performance ---
    print("  Figure 6: Per-fold performance comparison")
    fig_per_fold_performance(result)

    # --- Fig 7: CKA heatmap (ALL layers) ---
    print("  Figure 7: CKA heatmap (all sub-kernels + combined)")
    fig_cka_heatmap(gram_all)

    # --- Fig 8: Effective dimensionality (ALL layers) ---
    print("  Figure 8: Effective dimensionality (all layers)")
    fig_effective_dimensionality(gram_all)

    # --- Fig 9: Eigenvalue PSD ---
    print("  Figure 9: Eigenvalue spectra + PSD (all layers)")
    fig_eigenvalue_psd(gram_all)

    # --- Fig 10: CPCV/PBO ---
    print("  Figure 10: CPCV / PBO")
    lob_feat_dim = lob_raw.shape[1]
    pbo_kernel_a = SyntheticFeatureKernel(
        feature_dim=lob_feat_dim, n_rff=150, length_scale=1.0, seed=42,
        kernel_name="Good",
    )
    pbo_kernel_b = SyntheticFeatureKernel(
        feature_dim=lob_feat_dim, n_rff=150, length_scale=0.01, seed=42,
        kernel_name="Overfit",
    )
    evaluator = CPCVEvaluator(n_groups=6, n_test_groups=2, embargo_bars=20)
    pbo, details = evaluator.compute_pbo(
        lob_raw[:2000], y[:2000], [pbo_kernel_a, pbo_kernel_b],
        metric_fn=lambda yt, yp: -float(np.mean((yt - yp) ** 2)),
        reg_lambda=1e-2,
    )
    fig_cpcv_pbo(pbo, details)

    # --- Fig 11: CV surface ---
    print("  Figure 11: Hyperparameter CV surface")
    cv_kernel = SyntheticFeatureKernel(
        feature_dim=lob_feat_dim, n_rff=150, seed=42,
    )
    cv = PurgedKFold(n_folds=3, embargo=20)
    cv_result = inner_cv_grid_search(
        kernel=cv_kernel, data=lob_raw[:1000], y=y[:1000],
        length_scales=[0.1, 0.5, 1.0, 2.0, 5.0],
        reg_lambdas=[1e-4, 1e-3, 1e-2, 1e-1],
        cv=cv,
        metric_fn=lambda yt, yp: -float(np.mean((yt - yp) ** 2)),
    )
    fig_cv_surface(cv_result)

    # --- Fig 12: Signal diagnostics (fast + slow sub-kernels) ---
    print("  Figure 12: Signal diagnostics (all sub-kernels)")
    # Build per-layer distances for ALL sub-kernels
    layer_distance_results = {}

    # Fast sub-kernels (bar resolution)
    fast_diag_items = [
        ("K_LOB", lob_layer, snapshots[:500]),
        ("K_VPIN", vpin_kernel, vpin_features[:500]),
    ]
    if keep_lambda:
        fast_diag_items.append(("K_Lambda", lambda_kernel, lambda_features[:500]))
    fast_diag_items.append(("K_VannaCharm", vanna_kernel, vanna_features[:500]))

    for name, kernel, data in fast_diag_items:
        phi = kernel.feature_map(data)
        T_sub = phi.shape[0]
        dists = np.zeros(T_sub)
        for t in range(T_sub):
            start = max(0, t - 50 + 1)
            mu = np.mean(phi[start:t + 1], axis=0)
            dists[t] = np.linalg.norm(phi[t] - mu)
        layer_distance_results[name] = {
            "distances": dists, "snr_vs_combined": 0.0,
        }

    # Slow sub-kernels (daily → aligned to bar resolution)
    for name, kernel, data in [
        ("K_VRP", vrp_kernel, vrp_data),
        ("K_GEX", gex_kernel, gex_data),
        ("K_Sent", sent_kernel, sent_data),
    ]:
        phi_daily = kernel.feature_map(data)
        sub_idx = aligner.get_daily_indices(np.arange(500))
        sub_idx = np.clip(sub_idx, 0, len(phi_daily) - 1)
        phi_bar = phi_daily[sub_idx]
        T_sub = phi_bar.shape[0]
        dists = np.zeros(T_sub)
        for t in range(T_sub):
            start = max(0, t - 50 + 1)
            mu = np.mean(phi_bar[start:t + 1], axis=0)
            dists[t] = np.linalg.norm(phi_bar[t] - mu)
        layer_distance_results[name] = {
            "distances": dists, "snr_vs_combined": 0.0,
        }

    # Compute SNR for each layer
    combined_d = sum(v["distances"] for v in layer_distance_results.values())
    for name, info in layer_distance_results.items():
        d = info["distances"]
        residual = combined_d - d
        var_sig = np.var(d)
        var_noise = np.var(residual)
        info["snr_vs_combined"] = var_sig / max(var_noise, 1e-12)

    fig_signal_diagnostics(layer_distance_results)

    # --- Fig 13: Architecture diagram ---
    print("  Figure 13: Architecture diagram (two-level)")
    fig_architecture_diagram(
        alpha_fast=alpha_fast, alpha_slow=alpha_slow, beta=beta,
        fast_names=fast_names, slow_names=slow_names,
    )

    # --- Fig 14: Fast vs slow decomposition with gate ---
    print("  Figure 14: Fast vs slow decomposition + gate overlay")
    fig_fast_slow_decomposition(
        fast_distances=signal_fast.distances,
        slow_distances_aligned=signal_slow.distances,
        combined_distances=signal_combined.distances,
        gate_mask=gate_mask,
        prices=prices,
    )

    # --- Fig 15: Cross-level analysis ---
    print("  Figure 15: Cross-level analysis (fast × slow + MKL weights)")
    fig_cross_level_analysis(
        gram_fast_subs=gram_fast_subs,
        gram_slow_subs=gram_slow_subs,
        gram_fast=K_fast,
        gram_slow=K_slow,
        gram_total=K_total,
        alpha_fast=alpha_fast,
        alpha_slow=alpha_slow,
        beta=beta,
        fast_names=fast_names,
        slow_names=slow_names,
    )

    print(f"\n  All 15 figures saved to {FIG_DIR}/")
    print("  Done!\n")


if __name__ == "__main__":
    run_dashboard()
