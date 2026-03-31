"""
RKHS chart library — Midnight Ocean themed visualisations for every pipeline layer.

Tier 1 (Critical — bootstrap MC):
  - bootstrap_histograms()     - distribution of each metric with CI shading
  - equity_curve_drawdown()    - dual-panel equity curve + underwater plot
  - mc_fan_chart()             - spaghetti plot with percentile bands

Tier 2 (Portfolio analytics):
  - correlation_heatmap()      - strategy-pair correlation matrix
  - trade_pnl_box()            - P&L distribution by exit reason
  - vol_regime_timeline()      - annotated vol-regime bar chart
  - alpha_curve()              - strategy vs SPY benchmark with alpha spread
  - stress_test_chart()        - scenario analysis: equity impact across crises
  - strategy_scorecard_chart() - per-strategy health dashboard with sparklines
  - health_timeline()          - swimlane chart of health state transitions

Tier 3 (Operational):
  - slippage_scatter()         - estimate vs. actual slippage
  - risk_utilisation_bars()    - per-trade risk fraction utilisation
  - signal_funnel()            - pipeline conversion: signals → selections → executions
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analytics.bootstrap_mc import BootstrapOutput, BootstrapResult
from analytics.stress_test import StressTestOutput
from analytics.alpha import AlphaStats
from analytics.scorecard import PortfolioScorecard
from reporting.theme import (
    nyx_style, NYX_PURPLE, NYX_PINK, NYX_BLUE, NYX_LAVENDER, NYX_ICE,
    NYX_ORCHID, NYX_DARK, NYX_SURFACE, NYX_GRID, NYX_TEXT, NYX_MUTED,
    WIN_COLOR, LOSS_COLOR, NEUTRAL, SERIES_COLORS,
    color_for_value, gradient_fill, add_watermark, percentile_band_colors,
    format_pct, format_dollars,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Bootstrap Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_histograms(
    output: BootstrapOutput,
    metrics: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Distribution histograms for each bootstrap metric with CI shading.

    Shows observed value as a vertical line, 90% CI as shaded region,
    and P(negative) annotation.
    """
    if metrics is None:
        metrics = [k for k in output.results if k != "n_trades"]

    n = len(metrics)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    with nyx_style():
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, key in enumerate(metrics):
            ax = axes[i]
            r = output.results[key]

            # Regenerate distribution from stored paths if available,
            # otherwise use a synthetic normal approximation for the histogram
            if output.simulated_equities:
                # Recompute this metric from each sim
                vals = _extract_metric_distribution(output, key)
            else:
                # Approximate with normal (mean, inferred std from CIs)
                std_approx = (r.ci_95 - r.ci_5) / 3.29  # 90% CI ≈ 3.29 sigma
                vals = np.random.default_rng(42).normal(r.mean, max(std_approx, 1e-9), 5000)

            ax.hist(vals, bins=60, color=NYX_LAVENDER, alpha=0.7, edgecolor="none")

            # CI shading
            ax.axvspan(r.ci_5, r.ci_95, alpha=0.15, color=NYX_PURPLE, label="90% CI")

            # Observed line
            ax.axvline(r.observed, color=NYX_PINK, linewidth=2, linestyle="--",
                       label=f"Observed: {r.observed:.4f}")

            # Mean line
            ax.axvline(r.mean, color=NYX_ICE, linewidth=1.5, linestyle=":",
                       label=f"Mean: {r.mean:.4f}")

            # P(neg) annotation
            if r.prob_negative > 0:
                ax.annotate(
                    f"P(neg) = {r.prob_negative:.1%}",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    fontsize=8, color=NYX_PINK, va="top",
                )

            _label = key.replace("_", " ").title()
            ax.set_title(_label, fontsize=11)
            ax.legend(fontsize=7, loc="upper right")
            ax.set_ylabel("")
            ax.set_xlabel("")

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Bootstrap Monte Carlo Distributions", fontsize=16,
                     fontweight="bold", y=1.02)
        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


def equity_curve_drawdown(
    output: BootstrapOutput,
    strategy_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Dual-panel: equity curve (top) + underwater drawdown (bottom).

    Uses the observed equity from the bootstrap output.
    """
    equity = output.observed_equity
    n = len(equity)
    trades = np.arange(n)

    # Compute drawdown
    peak = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / np.where(peak > 0, peak, 1.0)

    with nyx_style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                         sharex=True)

        # ── Equity curve ──
        ax1.plot(trades, equity, color=NYX_LAVENDER, linewidth=1.5, label="Equity")
        ax1.fill_between(trades, equity[0], equity, alpha=0.1, color=NYX_LAVENDER)
        ax1.axhline(equity[0], color=NYX_MUTED, linewidth=0.8, linestyle="--", alpha=0.5)

        # Annotate final value
        final = equity[-1]
        total_ret = (final / equity[0]) - 1
        color = WIN_COLOR if total_ret >= 0 else LOSS_COLOR
        ax1.annotate(
            f"  {format_dollars(final)}  ({format_pct(total_ret)})",
            xy=(n - 1, final), fontsize=10, color=color, fontweight="bold",
            va="bottom" if total_ret >= 0 else "top",
        )

        title = f"Equity Curve — {strategy_name}" if strategy_name else "Equity Curve"
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel("Equity ($)")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=9)

        # ── Drawdown (underwater) ──
        ax2.fill_between(trades, 0, dd_pct, color=NYX_PINK, alpha=0.5)
        ax2.plot(trades, dd_pct, color=NYX_PINK, linewidth=0.8)

        # Annotate max drawdown
        max_dd_idx = np.argmin(dd_pct)
        max_dd_val = dd_pct[max_dd_idx]
        ax2.annotate(
            f"  Max DD: {format_pct(max_dd_val)}",
            xy=(max_dd_idx, max_dd_val), fontsize=9, color=NYX_PINK,
            fontweight="bold", va="top",
        )

        ax2.set_title("Drawdown (Underwater Plot)", fontsize=11)
        ax2.set_ylabel("Drawdown %")
        ax2.set_xlabel("Trade #")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.set_ylim(min(dd_pct) * 1.15, 0.02)

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


def mc_fan_chart(
    output: BootstrapOutput,
    max_paths: int = 200,
    percentiles: Tuple[int, ...] = (5, 25, 50, 75, 95),
    strategy_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Monte Carlo spaghetti fan chart with percentile bands.

    Requires store_paths=True in the bootstrap call.
    """
    if not output.simulated_equities:
        raise ValueError(
            "No simulated paths stored. Re-run bootstrap_trade_series with store_paths=True."
        )

    equities = output.simulated_equities
    n_sims = len(equities)
    n_points = len(equities[0])
    trades = np.arange(n_points)

    # Stack into matrix: (n_sims, n_points)
    eq_matrix = np.array(equities)

    with nyx_style():
        fig, ax = plt.subplots(figsize=(14, 8))

        # Spaghetti — thin, semi-transparent individual paths
        sample_idx = np.random.default_rng(42).choice(n_sims, size=min(max_paths, n_sims), replace=False)
        for idx in sample_idx:
            ax.plot(trades, eq_matrix[idx], color=NYX_LAVENDER, alpha=0.03, linewidth=0.5)

        # Percentile bands (filled)
        band_colors = percentile_band_colors(len(percentiles) // 2)
        pctls = np.percentile(eq_matrix, percentiles, axis=0)

        n_bands = len(percentiles) // 2
        for b in range(n_bands):
            lower = pctls[b]
            upper = pctls[-(b + 1)]
            bc = band_colors[b] if b < len(band_colors) else band_colors[-1]
            label = f"{percentiles[b]}–{percentiles[-(b+1)]}th pctl"
            ax.fill_between(trades, lower, upper, color=bc, label=label)

        # Median
        median_idx = len(percentiles) // 2
        ax.plot(trades, pctls[median_idx], color=NYX_ICE, linewidth=2,
                label=f"Median ({percentiles[median_idx]}th)")

        # Observed
        ax.plot(trades, output.observed_equity, color=NYX_PINK, linewidth=2,
                linestyle="--", label="Observed")

        # Start capital line
        ax.axhline(output.observed_equity[0], color=NYX_MUTED, linewidth=0.8,
                    linestyle=":", alpha=0.5)

        title = f"MC Fan Chart — {strategy_name}" if strategy_name else "Monte Carlo Fan Chart"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8, loc="upper left")

        # Annotate terminal stats
        terminal = eq_matrix[:, -1]
        ax.annotate(
            f"Terminal equity\n"
            f"  Median: {format_dollars(np.median(terminal))}\n"
            f"  5th: {format_dollars(np.percentile(terminal, 5))}\n"
            f"  95th: {format_dollars(np.percentile(terminal, 95))}\n"
            f"  P(ruin): {(terminal < output.observed_equity[0] * 0.5).mean():.1%}",
            xy=(0.98, 0.02), xycoords="axes fraction",
            fontsize=9, color=NYX_TEXT, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", fc=NYX_SURFACE, ec=NYX_GRID, alpha=0.9),
        )

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Portfolio Analytics
# ═══════════════════════════════════════════════════════════════════════════════

def correlation_heatmap(
    corr_df: pd.DataFrame,
    pval_df: Optional[pd.DataFrame] = None,
    title: str = "Strategy Correlation Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Heatmap of pairwise strategy correlations.

    Optionally annotates p-values with significance markers.
    """
    n = corr_df.shape[0]

    with nyx_style():
        fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(6, n * 1.2)))

        # Custom colormap: pink (neg) → dark purple (zero) → ice blue (pos)
        from matplotlib.colors import LinearSegmentedColormap
        nyx_cmap = LinearSegmentedColormap.from_list(
            "nyx_corr",
            [NYX_PINK, NYX_DARK, NYX_SURFACE, NYX_ICE],
            N=256,
        )

        im = ax.imshow(corr_df.values, cmap=nyx_cmap, vmin=-1, vmax=1, aspect="auto")

        # Tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr_df.index, fontsize=9)

        # Annotate values
        for i in range(n):
            for j in range(n):
                val = corr_df.iloc[i, j]
                if np.isnan(val):
                    text = "n/a"
                    color = NYX_MUTED
                else:
                    text = f"{val:.2f}"
                    color = NYX_TEXT if abs(val) < 0.6 else NYX_DARK

                    # Add significance stars from p-values
                    if pval_df is not None and not np.isnan(pval_df.iloc[i, j]):
                        p = pval_df.iloc[i, j]
                        if p < 0.01:
                            text += " ***"
                        elif p < 0.05:
                            text += " **"
                        elif p < 0.10:
                            text += " *"

                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Pearson ρ", fontsize=10, color=NYX_TEXT)
        cbar.ax.yaxis.set_tick_params(color=NYX_MUTED)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=NYX_MUTED)

        ax.set_title(title, fontsize=14)
        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


def trade_pnl_box(
    trades_df: pd.DataFrame,
    pnl_col: str = "pnl_pct",
    group_col: str = "exit_reason",
    title: str = "P&L Distribution by Exit Reason",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Box + swarm plot of per-trade P&L grouped by exit reason."""
    groups = sorted(trades_df[group_col].unique())
    data = [trades_df.loc[trades_df[group_col] == g, pnl_col].astype(float).values for g in groups]

    with nyx_style():
        fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2), 6))

        bp = ax.boxplot(
            data,
            labels=groups,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor=NYX_MUTED, alpha=0.5),
            medianprops=dict(color=NYX_PINK, linewidth=2),
            whiskerprops=dict(color=NYX_LAVENDER),
            capprops=dict(color=NYX_LAVENDER),
        )

        # Color each box
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(SERIES_COLORS[i % len(SERIES_COLORS)])
            patch.set_alpha(0.6)
            patch.set_edgecolor(NYX_TEXT)

        # Scatter individual trades on top
        for i, (g, d) in enumerate(zip(groups, data)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(d))
            ax.scatter(
                np.full_like(d, i + 1) + jitter, d,
                s=12, alpha=0.5, color=SERIES_COLORS[i % len(SERIES_COLORS)],
                edgecolors="none", zorder=3,
            )

        ax.axhline(0, color=NYX_MUTED, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("P&L (% of risk)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

        # Annotate counts
        for i, (g, d) in enumerate(zip(groups, data)):
            ax.annotate(f"n={len(d)}", xy=(i + 1, ax.get_ylim()[1]),
                        ha="center", va="top", fontsize=8, color=NYX_MUTED)

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


def vol_regime_timeline(
    features_df: pd.DataFrame,
    date_col: str = "as_of_date",
    regime_col: str = "vol_regime",
    rv_col: str = "rv20",
    title: str = "Volatility Regime Timeline",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Annotated vol-regime bar chart with RV20 overlay."""
    regime_colors = {
        "low": NYX_ICE,
        "normal": NYX_LAVENDER,
        "high": NYX_PINK,
        "insufficient_data": NYX_MUTED,
    }

    df = features_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    with nyx_style():
        fig, ax = plt.subplots(figsize=(14, 5))

        # Bar background for each regime
        for i, row in df.iterrows():
            colour = regime_colors.get(row[regime_col], NYX_MUTED)
            ax.axvspan(
                row[date_col] - pd.Timedelta(days=0.5),
                row[date_col] + pd.Timedelta(days=0.5),
                color=colour, alpha=0.25,
            )

        # RV20 line
        if rv_col in df.columns:
            rv_vals = pd.to_numeric(df[rv_col], errors="coerce")
            ax.plot(df[date_col], rv_vals, color=NYX_ORCHID, linewidth=1.5, label="RV20")
            ax.set_ylabel("Realised Volatility (annualised)")

        # Legend patches for regimes
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=c, alpha=0.4, label=label.title())
            for label, c in regime_colors.items()
            if label in df[regime_col].unique()
        ]
        ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0],
                  fontsize=8, loc="upper left")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Date")
        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


def alpha_curve(
    cumulative_df: pd.DataFrame,
    rolling_alpha_s: pd.Series,
    stats: AlphaStats,
    strategy_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Dual-panel alpha curve: cumulative returns vs SPY (top) + rolling alpha (bottom).

    Args:
        cumulative_df: Output of ``build_cumulative_curves()`` with columns
            ``strategy_cum``, ``benchmark_cum``, ``alpha_spread``.  DatetimeIndex.
        rolling_alpha_s: Output of ``rolling_alpha()``.  Same index.
        stats: :class:`AlphaStats` with annualised alpha, beta, IR.
        strategy_name: For chart title.
        save_path: If provided, save PNG at this path.
    """
    with nyx_style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), height_ratios=[3, 1], sharex=True,
        )

        dates = cumulative_df.index

        # ── Top panel: three lines + shaded alpha spread ──
        ax1.plot(dates, cumulative_df["strategy_cum"],
                 color=NYX_LAVENDER, linewidth=1.8, label="Strategy")
        ax1.plot(dates, cumulative_df["benchmark_cum"],
                 color=NYX_ICE, linewidth=1.4, label="SPY Buy & Hold")

        # Shaded alpha spread: mint above zero, pink below
        gradient_fill(
            ax1, dates,
            cumulative_df["alpha_spread"].values,
            baseline=0.0,
            color_above=WIN_COLOR,
            color_below=NYX_PINK,
            alpha=0.20,
        )

        # Dashed alpha spread line
        ax1.plot(dates, cumulative_df["alpha_spread"],
                 color=WIN_COLOR, linewidth=1.0, linestyle="--",
                 alpha=0.7, label="Alpha Spread")

        ax1.axhline(0, color=NYX_MUTED, linewidth=0.8, linestyle=":", alpha=0.5)

        # Stats box
        stats_text = (
            f"Ann. Alpha: {stats.annualised_alpha:+.1%}\n"
            f"Beta:       {stats.beta:.2f}\n"
            f"Info Ratio: {stats.information_ratio:+.2f}\n"
            f"Periods:    {stats.n_periods}w"
        )
        ax1.text(
            0.98, 0.95, stats_text,
            transform=ax1.transAxes,
            fontsize=9, fontfamily="monospace",
            color=NYX_TEXT, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", fc=NYX_SURFACE,
                      ec=NYX_GRID, alpha=0.9),
        )

        title = (f"Alpha Curve \u2014 {strategy_name} vs SPY"
                 if strategy_name else "Alpha Curve vs SPY")
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel("Cumulative Return")
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
        ax1.legend(fontsize=9, loc="upper left")

        # ── Bottom panel: rolling alpha ──
        ax2.plot(dates, rolling_alpha_s, color=NYX_ORCHID, linewidth=1.2)
        ax2.axhline(0, color=NYX_MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
        gradient_fill(
            ax2, dates,
            rolling_alpha_s.fillna(0.0).values,
            baseline=0.0,
            color_above=WIN_COLOR,
            color_below=NYX_PINK,
            alpha=0.25,
        )

        ax2.set_title("Rolling Alpha (13-week)", fontsize=11)
        ax2.set_ylabel("Annualised Alpha")
        ax2.set_xlabel("Date")
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
        )

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  \u2726 Saved: {save_path}")

    return fig


def rolling_sharpe_chart(
    rolling_sharpe_s: pd.Series,
    rolling_winrate_s: pd.Series,
    rolling_sortino_s: Optional[pd.Series] = None,
    strategy_name: str = "",
    window: int = 13,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Dual-panel rolling Sharpe (top) + rolling win rate (bottom).

    Args:
        rolling_sharpe_s: Output of ``rolling_sharpe()``.  DatetimeIndex.
        rolling_winrate_s: Output of ``rolling_win_rate()``.  Same index.
        strategy_name: For chart title.
        window: Window size for subtitle annotation.
        save_path: If provided, save PNG at this path.
    """
    with nyx_style():
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 9), height_ratios=[3, 1], sharex=True,
        )

        dates = rolling_sharpe_s.index

        # ── Top panel: rolling Sharpe ──
        ax1.plot(dates, rolling_sharpe_s,
                 color=NYX_ORCHID, linewidth=1.8, label="Rolling Sharpe")

        if rolling_sortino_s is not None:
            ax1.plot(dates, rolling_sortino_s,
                     color=NYX_ICE, linewidth=1.6, linestyle="--",
                     alpha=0.75, label="Rolling Sortino")

        gradient_fill(
            ax1, dates,
            rolling_sharpe_s.fillna(0.0).values,
            baseline=0.0,
            color_above=WIN_COLOR,
            color_below=NYX_PINK,
            alpha=0.20,
        )

        ax1.axhline(0, color=NYX_MUTED, linewidth=0.8, linestyle=":", alpha=0.5)
        ax1.axhline(1.0, color=NYX_ICE, linewidth=0.6, linestyle="--",
                     alpha=0.4, label="Sharpe = 1.0")
        ax1.axhline(2.0, color=NYX_ICE, linewidth=0.6, linestyle="--",
                     alpha=0.25, label="Sharpe = 2.0")

        # Stats box
        valid = rolling_sharpe_s.dropna()
        current_sharpe = float(valid.iloc[-1]) if len(valid) > 0 else 0.0
        mean_sharpe = float(valid.mean()) if len(valid) > 0 else 0.0
        pct_above_1 = float((valid > 1.0).mean()) if len(valid) > 0 else 0.0

        stats_lines = [
            f"Sharpe:",
            f"  Current:  {current_sharpe:+.2f}",
            f"  Mean:     {mean_sharpe:+.2f}",
            f"  % > 1.0:  {pct_above_1:.0%}",
        ]
        if rolling_sortino_s is not None:
            valid_sort = rolling_sortino_s.dropna()
            if len(valid_sort) > 0:
                stats_lines.extend([
                    f"Sortino:",
                    f"  Current:  {float(valid_sort.iloc[-1]):+.2f}",
                    f"  Mean:     {float(valid_sort.mean()):+.2f}",
                ])
        stats_text = "\n".join(stats_lines)
        ax1.text(
            0.98, 0.95, stats_text,
            transform=ax1.transAxes,
            fontsize=9, fontfamily="monospace",
            color=NYX_TEXT, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", fc=NYX_SURFACE,
                      ec=NYX_GRID, alpha=0.9),
        )

        title = (f"Rolling Sharpe & Sortino ({window}w) \u2014 {strategy_name}"
                 if strategy_name
                 else f"Rolling Sharpe & Sortino ({window}w)")
        ax1.set_title(title, fontsize=14)
        ax1.set_ylabel("Annualised Sharpe")
        ax1.legend(fontsize=9, loc="upper left")

        # ── Bottom panel: rolling win rate ──
        ax2.plot(dates, rolling_winrate_s,
                 color=NYX_LAVENDER, linewidth=1.2)
        ax2.axhline(0.5, color=NYX_MUTED, linewidth=0.8, linestyle=":", alpha=0.5)

        gradient_fill(
            ax2, dates,
            rolling_winrate_s.fillna(0.5).values,
            baseline=0.5,
            color_above=WIN_COLOR,
            color_below=NYX_PINK,
            alpha=0.20,
        )

        ax2.set_title(f"Rolling Win Rate ({window}w)", fontsize=11)
        ax2.set_ylabel("Win Rate")
        ax2.set_xlabel("Date")
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
        )

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  \u2726 Saved: {save_path}")

    return fig


def stress_test_chart(
    stress_output: StressTestOutput,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Multi-panel stress test dashboard: equity curves + summary table.

    Top panel: equity curves for each scenario overlaid, showing drawdown
    paths from the initial capital through each crisis.

    Bottom panel: summary table with peak DD, total P&L, survival status.
    """
    results = stress_output.results
    equities = stress_output.scenario_equities
    base_equity = stress_output.base_equity

    if not results:
        fig = plt.figure()
        return fig

    n_scenarios = len(results)

    with nyx_style():
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

        # ── Top panel: equity curves ──
        ax1 = fig.add_subplot(gs[0])

        # Sort scenarios by severity (worst DD first)
        sorted_keys = sorted(
            results.keys(),
            key=lambda k: results[k].peak_drawdown,
        )

        for i, key in enumerate(sorted_keys):
            r = results[key]
            eq = equities[key]
            color = SERIES_COLORS[i % len(SERIES_COLORS)]

            # Prepend the initial capital to show the shock
            full_eq = np.concatenate([[base_equity], eq])
            x = np.arange(len(full_eq))

            ax1.plot(x, full_eq, color=color, linewidth=1.8,
                     label=f"{r.scenario_name} (DD: {r.peak_drawdown:.1%})",
                     alpha=0.9)

            # Mark the trough
            trough_idx = np.argmin(full_eq)
            ax1.scatter([trough_idx], [full_eq[trough_idx]], color=color,
                       s=60, zorder=5, edgecolors="white", linewidth=0.8)

        # Reference lines
        ax1.axhline(base_equity, color=NYX_MUTED, linewidth=1,
                     linestyle="--", alpha=0.6, label=f"Start: {format_dollars(base_equity)}")
        ruin_level = base_equity * (1.0 + stress_output.ruin_threshold)
        ax1.axhline(ruin_level, color=NYX_PINK, linewidth=1.2,
                     linestyle=":", alpha=0.7,
                     label=f"Ruin ({stress_output.ruin_threshold:.0%})")

        ax1.set_title("Stress Test: Portfolio Equity Under Historical Crises",
                       fontsize=14)
        ax1.set_xlabel("Trade # (within scenario)")
        ax1.set_ylabel("Equity ($)")
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )
        ax1.legend(fontsize=8, loc="lower left",
                   bbox_to_anchor=(0, 0), framealpha=0.9)

        # ── Bottom panel: summary table ──
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")

        # Build table data
        col_labels = ["Scenario", "Peak DD", "Total P&L",
                       "End Equity", "Survived", "Worst Trade", "Trades"]
        cell_text = []
        cell_colors = []

        for key in sorted_keys:
            r = results[key]
            survived_str = "✓ Yes" if r.survival else "✗ NO"
            row = [
                r.scenario_name,
                f"{r.peak_drawdown:.1%}",
                f"{r.total_pnl_pct:+.1%}",
                f"{r.ending_equity_pct:.1%}",
                survived_str,
                f"{r.worst_single_trade:+.1%}",
                str(r.n_trades),
            ]
            cell_text.append(row)

            # Color-code survival
            if r.survival:
                cell_colors.append([NYX_SURFACE] * 7)
            else:
                # Highlight failed scenarios
                cell_colors.append(
                    [NYX_SURFACE, NYX_SURFACE, NYX_SURFACE,
                     NYX_SURFACE, "#4A1030", NYX_SURFACE, NYX_SURFACE]
                )

        table = ax2.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=[NYX_GRID] * 7,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(NYX_GRID)
            if row == 0:
                # Header row
                cell.set_text_props(color=NYX_TEXT, fontweight="bold")
                cell.set_facecolor(NYX_GRID)
            else:
                cell.set_text_props(color=NYX_TEXT)

        ax2.set_title("Scenario Summary", fontsize=12, pad=10,
                       color=NYX_TEXT)

        add_watermark(fig)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✦ Saved: {save_path}")

    return fig


def strategy_scorecard_chart(
    portfolio_scorecard: PortfolioScorecard,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Strategy health dashboard — one row per strategy with status, metrics, sparkline.

    Layout per row (using gridspec):
      Col 0: Status badge (coloured circle — green / amber / red)
      Col 1: Strategy name + trade count
      Col 2: Key metrics text block (monospace)
      Col 3: Sparkline of rolling win rate with gradient fill above/below breakeven

    Strategies with 0 trades show "Awaiting trades…".
    """
    scorecards = list(portfolio_scorecard.strategies.values())
    n_rows = len(scorecards)

    if n_rows == 0:
        fig = plt.figure()
        return fig

    STATUS_COLORS = {
        "HEALTHY":  WIN_COLOR,    # green-mint
        "WARNING":  "#FFD180",    # amber
        "CRITICAL": NYX_PINK,     # pink-red
    }

    row_height = 1.8
    fig_height = max(5, n_rows * row_height + 2.5)

    with nyx_style():
        fig = plt.figure(figsize=(16, fig_height))

        # Title area at top
        fig.suptitle(
            f"Strategy Health Scorecard — {portfolio_scorecard.overall_status}",
            fontsize=16, fontweight="bold", y=0.97,
            color=STATUS_COLORS.get(portfolio_scorecard.overall_status, NYX_TEXT),
        )

        # Subtitle with aggregate counts
        subtitle = (
            f"{portfolio_scorecard.n_strategies} strategies  ·  "
            f"{portfolio_scorecard.n_total_trades} trades  ·  "
            f"✓ {portfolio_scorecard.n_healthy} healthy"
        )
        if portfolio_scorecard.n_warning > 0:
            subtitle += f"  ·  ⚠ {portfolio_scorecard.n_warning} warning"
        if portfolio_scorecard.n_critical > 0:
            subtitle += f"  ·  ✗ {portfolio_scorecard.n_critical} critical"
        fig.text(0.5, 0.93, subtitle, ha="center", fontsize=10, color=NYX_MUTED)

        # Grid: n_rows × 4 columns (badge, name, metrics, sparkline)
        gs = fig.add_gridspec(
            n_rows, 4,
            width_ratios=[0.6, 2.5, 4, 5],
            left=0.04, right=0.97, top=0.88, bottom=0.06,
            hspace=0.5, wspace=0.3,
        )

        for i, sc in enumerate(scorecards):
            status_color = STATUS_COLORS.get(sc.status, NEUTRAL)

            # ── Col 0: Status badge ──
            ax_badge = fig.add_subplot(gs[i, 0])
            ax_badge.set_xlim(0, 1)
            ax_badge.set_ylim(0, 1)
            ax_badge.add_patch(plt.Circle(
                (0.5, 0.5), 0.35,
                color=status_color, alpha=0.9,
                transform=ax_badge.transData,
            ))
            # Status initial inside circle
            badge_text = sc.status[0]  # H, W, or C
            ax_badge.text(0.5, 0.5, badge_text, ha="center", va="center",
                          fontsize=14, fontweight="bold",
                          color=NYX_DARK if sc.status != "CRITICAL" else NYX_TEXT)
            ax_badge.set_aspect("equal")
            ax_badge.axis("off")

            # ── Col 1: Strategy name + trade count ──
            ax_name = fig.add_subplot(gs[i, 1])
            ax_name.set_xlim(0, 1)
            ax_name.set_ylim(0, 1)
            ax_name.text(0.05, 0.65, sc.strategy_name,
                         fontsize=12, fontweight="bold", color=NYX_TEXT, va="center")
            trade_label = f"{sc.n_trades} trades" if sc.n_trades != 1 else "1 trade"
            ax_name.text(0.05, 0.30, trade_label,
                         fontsize=9, color=NYX_MUTED, va="center")
            ax_name.axis("off")

            # ── Col 2: Key metrics ──
            ax_metrics = fig.add_subplot(gs[i, 2])
            ax_metrics.set_xlim(0, 1)
            ax_metrics.set_ylim(0, 1)

            if sc.n_trades == 0:
                ax_metrics.text(0.05, 0.5, "Awaiting trades…",
                                fontsize=10, color=NYX_MUTED, va="center",
                                fontstyle="italic")
            else:
                lines = []
                # Win rate
                wr_str = f"{sc.rolling_win_rate:.0%}" if sc.rolling_win_rate is not None else "n/a"
                lines.append(f"WR: {wr_str}")
                # Profit factor
                if sc.rolling_profit_factor == float("inf"):
                    pf_str = "∞"
                elif sc.rolling_profit_factor is not None:
                    pf_str = f"{sc.rolling_profit_factor:.2f}"
                else:
                    pf_str = "n/a"
                lines.append(f"PF: {pf_str}")
                # Avg P&L
                lines.append(f"Avg: {sc.avg_pnl_per_trade:+.1%}")
                # Loss streak
                lines.append(f"Streak: {sc.consecutive_loss_count}L")
                # Sharpe
                if sc.rolling_sharpe is not None:
                    lines.append(f"Sharpe: {sc.rolling_sharpe:.2f}")
                if sc.rolling_sortino is not None:
                    lines.append(f"Sortino: {sc.rolling_sortino:.2f}")
                # Baseline delta
                if sc.live_vs_baseline_delta is not None:
                    lines.append(f"vs Base: {sc.live_vs_baseline_delta:+.0%}")

                metrics_text = "  |  ".join(lines)
                ax_metrics.text(0.02, 0.55, metrics_text,
                                fontsize=8.5, fontfamily="monospace",
                                color=NYX_TEXT, va="center",
                                bbox=dict(boxstyle="round,pad=0.4",
                                          fc=NYX_SURFACE, ec=NYX_GRID, alpha=0.8))

                # Status reasons below metrics
                if sc.status_reasons:
                    reason_text = "; ".join(sc.status_reasons[:2])
                    ax_metrics.text(0.02, 0.12, reason_text,
                                    fontsize=7, color=status_color, va="center",
                                    fontstyle="italic")

            ax_metrics.axis("off")

            # ── Col 3: Sparkline ──
            ax_spark = fig.add_subplot(gs[i, 3])

            if sc.n_trades > 1 and sc.rolling_win_rate_series:
                series = sc.rolling_win_rate_series
                x = np.arange(len(series))
                y = np.array(series)

                ax_spark.plot(x, y, color=NYX_LAVENDER, linewidth=1.5)

                # Gradient fill above/below breakeven (0.5)
                gradient_fill(
                    ax_spark, x, y,
                    baseline=0.5,
                    color_above=WIN_COLOR,
                    color_below=NYX_PINK,
                    alpha=0.25,
                )

                # Breakeven reference line
                ax_spark.axhline(0.5, color=NYX_MUTED, linewidth=0.7,
                                  linestyle=":", alpha=0.6)

                # Y-axis: 0% to 100%
                ax_spark.set_ylim(-0.05, 1.05)
                ax_spark.set_xlim(-0.5, len(series) - 0.5)

                # Current win rate annotation
                current_wr = series[-1]
                wr_color = WIN_COLOR if current_wr >= 0.5 else NYX_PINK
                ax_spark.annotate(
                    f" {current_wr:.0%}",
                    xy=(len(series) - 1, current_wr),
                    fontsize=9, fontweight="bold", color=wr_color,
                    va="center",
                )

                # Minimal axis styling
                ax_spark.set_ylabel("")
                ax_spark.set_xlabel("")
                ax_spark.tick_params(
                    left=False, labelleft=False,
                    bottom=False, labelbottom=False,
                )
                ax_spark.spines["left"].set_visible(False)
                ax_spark.spines["bottom"].set_visible(False)

            elif sc.n_trades == 1:
                ax_spark.text(0.5, 0.5, f"{'Win' if sc.rolling_win_rate and sc.rolling_win_rate > 0.5 else 'Loss'}",
                              ha="center", va="center", fontsize=10,
                              color=WIN_COLOR if sc.rolling_win_rate and sc.rolling_win_rate > 0.5 else NYX_PINK)
                ax_spark.axis("off")
            else:
                ax_spark.text(0.5, 0.5, "—",
                              ha="center", va="center", fontsize=12,
                              color=NYX_MUTED)
                ax_spark.axis("off")

        add_watermark(fig)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✦ Saved: {save_path}")

    return fig


def health_timeline(
    history_df: pd.DataFrame,
    title: str = "Strategy Health Timeline",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Swimlane chart of strategy health states over time.

    One horizontal lane per strategy.  Background color encodes status.
    Transition markers (diamonds) at each state change.
    Uses the axvspan pattern from vol_regime_timeline().

    Args:
        history_df: Output of scorecard_history.load_scorecard_history().
            Must have columns: snapshot_timestamp, strategy_name, status.
        title: Chart title.
        save_path: If provided, save PNG.

    Returns:
        matplotlib Figure.
    """
    STATUS_COLORS = {
        "HEALTHY":  WIN_COLOR,
        "WARNING":  "#FFD180",
        "CRITICAL": NYX_PINK,
    }

    df = history_df.copy()
    df["snapshot_timestamp"] = pd.to_datetime(
        df["snapshot_timestamp"], errors="coerce", utc=True,
    )
    df = df.dropna(subset=["snapshot_timestamp"])

    if df.empty:
        fig = plt.figure()
        return fig

    strategies = sorted(df["strategy_name"].unique())
    n_strats = len(strategies)

    with nyx_style():
        fig_height = max(4, n_strats * 1.2 + 2)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        for i, strat in enumerate(strategies):
            strat_df = df[df["strategy_name"] == strat].sort_values("snapshot_timestamp")

            if strat_df.empty:
                continue

            # Build consecutive status blocks
            timestamps = strat_df["snapshot_timestamp"].values
            statuses = strat_df["status"].values

            for j in range(len(timestamps)):
                status = statuses[j]
                color = STATUS_COLORS.get(status, NEUTRAL)

                # Block start = this timestamp, block end = next timestamp (or last + buffer)
                t_start = pd.Timestamp(timestamps[j])
                if j + 1 < len(timestamps):
                    t_end = pd.Timestamp(timestamps[j + 1])
                else:
                    # Extend last block by average interval or 1 day
                    if len(timestamps) > 1:
                        avg_gap = (pd.Timestamp(timestamps[-1]) - pd.Timestamp(timestamps[0])) / max(1, len(timestamps) - 1)
                        t_end = t_start + avg_gap
                    else:
                        t_end = t_start + pd.Timedelta(days=1)

                # Y-band for this strategy
                y_bottom = n_strats - i - 1
                ax.axvspan(
                    t_start, t_end,
                    ymin=y_bottom / n_strats,
                    ymax=(y_bottom + 1) / n_strats,
                    color=color,
                    alpha=0.35,
                )

                # Transition marker (diamond) where status changed
                if j > 0 and statuses[j] != statuses[j - 1]:
                    ax.plot(
                        t_start, y_bottom + 0.5,
                        marker="D", markersize=7,
                        color=color, markeredgecolor=NYX_TEXT,
                        markeredgewidth=0.8, zorder=5,
                    )

        # Y-axis: strategy names
        ax.set_yticks([n_strats - i - 0.5 for i in range(n_strats)])
        ax.set_yticklabels(strategies, fontsize=10)
        ax.set_ylim(0, n_strats)

        # X-axis
        ax.set_xlabel("Date")

        # Legend
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=c, alpha=0.5, label=label)
            for label, c in STATUS_COLORS.items()
        ]
        ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

        ax.set_title(title, fontsize=14)
        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  ✦ Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Operational
# ═══════════════════════════════════════════════════════════════════════════════

def risk_utilisation_bars(
    executions_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Per-trade risk fraction utilisation vs. the hard cap."""
    df = executions_df.copy()
    df["risk_used_fraction"] = pd.to_numeric(df.get("risk_used_fraction", 0), errors="coerce")
    df["max_risk_fraction"] = pd.to_numeric(df.get("max_risk_fraction", 0.05), errors="coerce")

    with nyx_style():
        fig, ax = plt.subplots(figsize=(14, 5))

        x = np.arange(len(df))
        colors = [NYX_LAVENDER if v <= cap else NYX_PINK
                  for v, cap in zip(df["risk_used_fraction"], df["max_risk_fraction"])]

        ax.bar(x, df["risk_used_fraction"], color=colors, alpha=0.8, width=0.8, label="Risk Used")
        ax.axhline(0.05, color=NYX_PINK, linewidth=1.5, linestyle="--",
                    label="Hard Cap (5%)", alpha=0.8)

        ax.set_title("Risk Utilisation per Trade", fontsize=14)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Fraction of Equity Risked")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.legend(fontsize=9)

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


def signal_funnel(
    n_signals: int,
    n_selections: int,
    n_executions: int,
    n_rejected: int = 0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Pipeline conversion funnel: signals → selections → executions."""
    labels = ["Signals", "Selections", "Executions"]
    values = [n_signals, n_selections, n_executions]
    colors = [NYX_LAVENDER, NYX_ICE, NYX_PURPLE]

    if n_rejected > 0:
        labels.append("Rejected")
        values.append(n_rejected)
        colors.append(NYX_PINK)

    with nyx_style():
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, height=0.6,
                       edgecolor=NYX_GRID)

        # Annotate counts
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=12, fontweight="bold", color=NYX_TEXT)

        # Conversion rates
        for i in range(1, min(3, len(values))):
            if values[i - 1] > 0:
                rate = values[i] / values[i - 1]
                mid_y = (y_pos[i - 1] + y_pos[i]) / 2
                ax.annotate(
                    f"→ {rate:.0%}",
                    xy=(max(values) * 0.5, mid_y),
                    fontsize=10, color=NYX_MUTED, ha="center", va="center",
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=12)
        ax.invert_yaxis()
        ax.set_title("Pipeline Signal Funnel", fontsize=14)
        ax.set_xlabel("Count")

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


def slippage_scatter(
    executions_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Scatter of estimated slippage vs. actual (fill_price - intended_entry_price)."""
    df = executions_df.copy()
    df["intended_entry_price"] = pd.to_numeric(df.get("intended_entry_price", 0), errors="coerce")
    df["fill_price"] = pd.to_numeric(df.get("fill_price", 0), errors="coerce")
    df["slippage_estimate"] = pd.to_numeric(df.get("slippage_estimate", 0), errors="coerce")

    # Only rows with actual fills
    mask = (df["fill_price"] > 0) & (df["intended_entry_price"] > 0)
    df = df[mask].copy()

    if df.empty:
        return plt.figure()

    df["actual_slippage"] = df["fill_price"] - df["intended_entry_price"]

    with nyx_style():
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(df["slippage_estimate"], df["actual_slippage"],
                   color=NYX_LAVENDER, s=40, alpha=0.7, edgecolors=NYX_PURPLE, linewidths=0.5)

        # Perfect-prediction line
        lim_min = min(df["slippage_estimate"].min(), df["actual_slippage"].min()) * 1.2
        lim_max = max(df["slippage_estimate"].max(), df["actual_slippage"].max()) * 1.2
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color=NYX_MUTED, linestyle="--",
                linewidth=0.8, label="Perfect prediction")

        ax.set_title("Slippage: Estimate vs. Actual", fontsize=14)
        ax.set_xlabel("Estimated Slippage ($)")
        ax.set_ylabel("Actual Slippage ($)")
        ax.legend(fontsize=9)

        add_watermark(fig)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  ✦ Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_metric_distribution(output: BootstrapOutput, metric: str) -> np.ndarray:
    """Re-derive a single metric's distribution from stored equity curves + pnls."""
    from analytics.bootstrap_mc import _compute_metrics

    vals = []
    initial_capital = output.observed_equity[0]
    ruin_threshold = -0.50  # default

    for eq, pnls in zip(output.simulated_equities, output.simulated_pnls):
        metrics = _compute_metrics(eq, pnls, initial_capital, ruin_threshold)
        vals.append(metrics.get(metric, 0.0))

    return np.array(vals)
