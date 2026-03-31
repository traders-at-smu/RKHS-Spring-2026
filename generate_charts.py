"""
Generate Midnight Ocean charts from backtest results.
Run after run_backtest.py produces backtest_results.npz.

Usage:
    python3 generate_charts.py                                    # baseline
    python3 generate_charts.py --results backtest_results_optimized.npz \
        --tag _optimized --title-suffix " (MKL Optimized)"        # optimized
"""

import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "reporting"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from reporting.theme import (
    ocean_style, OCEAN_TEAL, OCEAN_CORAL, OCEAN_BLUE, OCEAN_CYAN,
    OCEAN_GOLD, OCEAN_SEA, OCEAN_DARK, OCEAN_SURFACE, OCEAN_GRID,
    OCEAN_TEXT, OCEAN_MUTED, WIN_COLOR, LOSS_COLOR, NEUTRAL,
    SERIES_COLORS, gradient_fill, add_watermark, percentile_band_colors,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="backtest_results.npz",
                        help="Path to results .npz file")
    parser.add_argument("--tag", default="",
                        help="Suffix for output filenames (e.g. '_optimized')")
    parser.add_argument("--title-suffix", default="",
                        help="Extra text appended to chart titles")
    args = parser.parse_args()

    out_dir = os.path.join(_ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)

    tag = args.tag
    tsuf = args.title_suffix

    data = np.load(os.path.join(_ROOT, args.results))
    y_pred = data["y_pred_oos"]
    y_true = data["y_true_oos"]
    s_scores = data["s_scores"]
    positions = data["positions"]
    strat_returns = data["strat_returns"]
    cum_pnl = data["cum_pnl"]

    # MKL weights if present
    alpha_fast = data["alpha_fast"] if "alpha_fast" in data else None
    alpha_slow = data["alpha_slow"] if "alpha_slow" in data else None
    beta_val = float(data["beta"]) if "beta" in data else None

    n = len(cum_pnl)
    x = np.arange(n)

    with ocean_style():

        # ── 1. Equity Curve + Drawdown ────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                        height_ratios=[3, 1], sharex=True)
        fig.suptitle(f"Equity Curve & Underwater Plot{tsuf}", fontsize=16,
                     fontweight="bold", y=0.95)

        ax1.plot(x, cum_pnl, color=OCEAN_TEAL, linewidth=1.5, label="Cum P&L")
        gradient_fill(ax1, x, cum_pnl, baseline=0.0)
        ax1.axhline(0, color=OCEAN_MUTED, linewidth=0.5, linestyle="--")
        ax1.set_ylabel("Cumulative P&L (log returns)")
        ax1.legend(loc="upper left")

        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = cum_pnl - running_max
        ax2.fill_between(x, 0, drawdown, color=OCEAN_CORAL, alpha=0.5)
        ax2.plot(x, drawdown, color=OCEAN_CORAL, linewidth=0.8)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("OOS Bar Index")

        add_watermark(fig, "RKHS")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bt_01_equity_drawdown{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [1/8] Equity + drawdown")

        # ── 2. S-Score Signal Timeline ────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8),
                                        height_ratios=[2, 1], sharex=True)
        fig.suptitle(f"S-Score Signals & Positions{tsuf}", fontsize=16,
                     fontweight="bold", y=0.95)

        ax1.plot(s_scores, color=OCEAN_CYAN, linewidth=0.6, alpha=0.8)
        ax1.axhline(1.25, color=OCEAN_CORAL, linewidth=0.8, linestyle="--",
                     label="Entry +1.25")
        ax1.axhline(-1.25, color=OCEAN_SEA, linewidth=0.8, linestyle="--",
                     label="Entry -1.25")
        ax1.axhline(0.5, color=OCEAN_MUTED, linewidth=0.5, linestyle=":")
        ax1.axhline(-0.5, color=OCEAN_MUTED, linewidth=0.5, linestyle=":")
        ax1.set_ylabel("S-Score")
        ax1.legend(loc="upper right")

        colors = [WIN_COLOR if p > 0 else (LOSS_COLOR if p < 0 else NEUTRAL)
                  for p in positions]
        ax2.bar(range(len(positions)), positions, color=colors, width=1.0,
                alpha=0.6)
        ax2.set_ylabel("Position")
        ax2.set_xlabel("OOS Bar Index")

        add_watermark(fig, "RKHS")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bt_02_sscore_signals{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [2/8] S-score signals")

        # ── 3. OOS Predictions vs Actuals ─────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.scatter(y_true, y_pred, s=3, alpha=0.3, color=OCEAN_CYAN,
                   edgecolors="none")
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, color=OCEAN_CORAL, linewidth=1, linestyle="--",
                label="Perfect")
        ax.set_xlabel("Actual Returns")
        ax.set_ylabel("Predicted Returns")
        ax.set_title(f"OOS Predictions vs Actuals{tsuf}")
        ax.legend()

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        ax.text(0.05, 0.95, f"Corr: {corr:.4f}",
                transform=ax.transAxes, fontsize=12, color=OCEAN_GOLD)

        add_watermark(fig, "RKHS")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bt_03_pred_vs_actual{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [3/8] Predictions vs actuals")

        # ── 4. Return Distribution ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.hist(strat_returns, bins=80, color=OCEAN_TEAL, alpha=0.7,
                edgecolor="none", density=True)
        ax.axvline(0, color=OCEAN_MUTED, linewidth=0.8, linestyle="--")
        ax.axvline(strat_returns.mean(), color=OCEAN_GOLD, linewidth=1.5,
                   label=f"Mean: {strat_returns.mean():.6f}")
        ax.set_xlabel("Strategy Return")
        ax.set_ylabel("Density")
        ax.set_title(f"Strategy Return Distribution{tsuf}")
        ax.legend()

        from scipy.stats import skew, kurtosis
        stats_text = (f"Skew: {skew(strat_returns):.2f}\n"
                      f"Kurt: {kurtosis(strat_returns):.2f}")
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, color=OCEAN_TEXT, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor=OCEAN_SURFACE, alpha=0.9))

        add_watermark(fig, "RKHS")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bt_04_return_dist{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [4/8] Return distribution")

        # ── 5. Rolling Sharpe ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(16, 7))
        window = 252
        if len(strat_returns) > window:
            roll_mean = np.convolve(strat_returns,
                                     np.ones(window) / window, mode="valid")
            roll_std = np.array([
                strat_returns[i:i + window].std()
                for i in range(len(strat_returns) - window + 1)
            ])
            roll_sr = roll_mean / (roll_std + 1e-8) * np.sqrt(252)
            rx = np.arange(len(roll_sr))

            ax.plot(rx, roll_sr, color=OCEAN_TEAL, linewidth=1.2)
            gradient_fill(ax, rx, roll_sr, baseline=0.0)
            ax.axhline(0, color=OCEAN_MUTED, linewidth=0.8, linestyle="--")
            ax.set_title(f"Rolling {window}-bar Sharpe Ratio{tsuf}")
        else:
            ax.text(0.5, 0.5, "Not enough data for rolling window",
                    transform=ax.transAxes, ha="center", color=OCEAN_TEXT)
            ax.set_title(f"Rolling Sharpe Ratio{tsuf}")

        ax.set_xlabel("Bar Index")
        ax.set_ylabel("Sharpe Ratio (annualized)")

        add_watermark(fig, "RKHS")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bt_05_rolling_sharpe{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [5/8] Rolling Sharpe")

        # ── 6. Monte Carlo Fan Chart (bootstrap) ─────────────────────────
        fig, ax = plt.subplots(figsize=(16, 8))
        n_sims = 1000
        rng = np.random.default_rng(42)
        sim_paths = np.zeros((n_sims, n))
        for i in range(n_sims):
            block_size = 20
            bootstrapped = []
            while len(bootstrapped) < n:
                start = rng.integers(0, len(strat_returns))
                end = min(start + block_size, len(strat_returns))
                bootstrapped.extend(strat_returns[start:end].tolist())
            sim_paths[i] = np.cumsum(bootstrapped[:n])

        percentiles = [5, 10, 25, 50, 75, 90, 95]
        pct_vals = np.percentile(sim_paths, percentiles, axis=0)
        band_colors = percentile_band_colors(3)

        ax.fill_between(x, pct_vals[0], pct_vals[6], color=band_colors[0],
                         label="5-95%")
        ax.fill_between(x, pct_vals[1], pct_vals[5], color=band_colors[1],
                         label="10-90%")
        ax.fill_between(x, pct_vals[2], pct_vals[4], color=band_colors[2],
                         label="25-75%")
        ax.plot(x, pct_vals[3], color=OCEAN_GOLD, linewidth=1.5,
                label="Median", linestyle="--")
        ax.plot(x, cum_pnl, color=OCEAN_TEAL, linewidth=2, label="Observed")

        for i in range(min(50, n_sims)):
            ax.plot(x, sim_paths[i], color=OCEAN_CYAN, alpha=0.03, linewidth=0.5)

        ax.axhline(0, color=OCEAN_MUTED, linewidth=0.5, linestyle="--")
        ax.set_title(f"Monte Carlo Fan Chart (1,000 Block Bootstrap){tsuf}")
        ax.set_xlabel("OOS Bar Index")
        ax.set_ylabel("Cumulative P&L")
        ax.legend(loc="upper left")

        add_watermark(fig, "RKHS")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bt_06_mc_fan{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [6/8] MC fan chart")

        # ── 7. Kernel Weight Allocation ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        # Detect kernel configuration from MKL weights
        if alpha_fast is not None and alpha_slow is not None:
            n_fk = len(alpha_fast)
            n_sk = len(alpha_slow)
            weights = list(alpha_fast) + list(alpha_slow)
            weight_label = "MKL Optimized"
            beta_text = f"  |  beta = {beta_val:.4f}" if beta_val else ""

            # v2 architecture: consolidated OrderFlow + VannaCharm fast, 3 slow
            if n_fk == 2 and n_sk == 3:
                kernel_names = ["OrderFlow\n(PCA 15D)", "Vanna\nCharm",
                                "VRP", "Macro\nMotion", "Sentiment"]
            elif n_fk == 2 and n_sk == 2:
                kernel_names = ["OrderFlow\n(PCA 15D)", "Vanna\nCharm",
                                "VRP", "Macro\nMotion"]
            else:
                kernel_names = [f"Fast_{i}" for i in range(n_fk)] + \
                               [f"Slow_{i}" for i in range(n_sk)]
        else:
            kernel_names = ["LOB", "VPIN", "Kyle's\nLambda", "Hawkes", "VRP", "Macro\nMotion"]
            weights = [0.4, 0.25, 0.15, 0.2, 0.5, 0.5]
            weight_label = "Pre-MKL Defaults"
            beta_text = ""
            n_fk = 4

        all_colors = [OCEAN_TEAL, OCEAN_CORAL, SERIES_COLORS[7], SERIES_COLORS[5],
                      OCEAN_SEA, SERIES_COLORS[6], OCEAN_CYAN, OCEAN_GOLD]
        kcolors = all_colors[:len(kernel_names)]

        bars = ax.barh(kernel_names, weights, color=kcolors, height=0.6,
                       edgecolor=OCEAN_GRID)
        ax.set_xlabel("Weight (alpha)")
        ax.set_title(f"Kernel Weight Allocation ({weight_label}){beta_text}")

        for bar, w in zip(bars, weights):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.4f}", va="center", fontsize=11, color=OCEAN_TEXT)

        n_fast_k_chart = n_fk if alpha_fast is not None else 4
        ax.axhline(n_fast_k_chart - 0.5, color=OCEAN_MUTED, linewidth=0.8, linestyle="--")
        ax.text(0.85, n_fast_k_chart + 0.3, "FAST", transform=ax.get_yaxis_transform(),
                fontsize=10, color=OCEAN_TEAL, fontweight="bold")
        ax.text(0.85, 0.8, "SLOW", transform=ax.get_yaxis_transform(),
                fontsize=10, color=OCEAN_SEA, fontweight="bold")

        add_watermark(fig, "RKHS")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"bt_07_kernel_weights{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [7/8] Kernel weights")

        # ── 8. Performance Summary Card ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis("off")

        sr = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)
        max_dd = np.max(running_max - cum_pnl)
        # Sortino: semi-deviation = sqrt(mean(min(r, 0)^2)), same as metrics.py
        downside_dev = np.sqrt(np.mean(np.minimum(strat_returns, 0)**2))
        sortino = np.mean(strat_returns) / (downside_dev + 1e-8) * np.sqrt(252) if downside_dev > 0 else 0

        from scipy.stats import skew, kurtosis

        # Active-day hit rate (not diluted by flat days)
        daily_pos = data["daily_positions"] if "daily_positions" in data else positions
        n_eval = min(len(daily_pos) - 1, len(strat_returns))
        active_mask = daily_pos[:n_eval] != 0
        if active_mask.sum() > 0:
            active_rets = strat_returns[active_mask[:len(strat_returns)]] if len(active_mask) <= len(strat_returns) else strat_returns[strat_returns != 0]
            hit = (active_rets > 0).mean() * 100
            n_wins = int((active_rets > 0).sum())
            n_losses = int((active_rets < 0).sum())
            avg_win = active_rets[active_rets > 0].mean() if n_wins > 0 else 0
            avg_loss = abs(active_rets[active_rets < 0].mean()) if n_losses > 0 else 1e-8
            wl_ratio = avg_win / avg_loss
        else:
            hit = 0
            n_wins = n_losses = 0
            wl_ratio = 0

        stage = "Two-Stage + Kalman" if alpha_fast is not None else "Pre-MKL Baseline"

        metrics = [
            ("Sharpe Ratio",     f"{sr:.4f}"),
            ("Sortino Ratio",    f"{sortino:.4f}"),
            ("Max Drawdown",     f"{max_dd:.4f}"),
            ("Hit Rate (active)",f"{hit:.1f}%"),
            ("Win/Loss Ratio",   f"{wl_ratio:.2f}x"),
            ("Wins / Losses",    f"{n_wins} / {n_losses}"),
            ("Total P&L (net)",  f"{cum_pnl[-1]:.6f}"),
            ("Skewness",         f"{skew(strat_returns):.2f}"),
            ("Kurtosis",         f"{kurtosis(strat_returns):.2f}"),
            ("Trading Days",     f"{len(strat_returns):,}"),
            ("Days Active",      f"{int(active_mask.sum())} ({active_mask.mean():.1%})"),
            ("Kernels Active",   "6 / 10 + 2 gates"),
            ("Stage",            stage),
        ]

        fig.text(0.5, 0.95, f"RKHS Strategy Scorecard",
                 fontsize=18, fontweight="bold", ha="center",
                 color=OCEAN_TEAL)
        fig.text(0.5, 0.91,
                 f"CL Futures  |  Jul 2023 - Dec 2024  |  {stage}",
                 fontsize=11, ha="center", color=OCEAN_MUTED)

        y_pos = 0.82
        for label, value in metrics:
            color = OCEAN_TEXT
            if label in ("Sharpe Ratio", "Sortino Ratio", "Total P&L"):
                try:
                    v = float(value)
                    color = WIN_COLOR if v > 0 else LOSS_COLOR
                except ValueError:
                    pass

            fig.text(0.25, y_pos, label, fontsize=13, color=OCEAN_MUTED,
                     fontweight="bold")
            fig.text(0.75, y_pos, value, fontsize=13, color=color,
                     ha="right", fontfamily="monospace")
            y_pos -= 0.07

        add_watermark(fig, "RKHS")
        fig.savefig(os.path.join(out_dir, f"bt_08_scorecard{tag}.png"), dpi=150)
        plt.close(fig)
        print(f"  [8/8] Scorecard")

    print(f"\n  All 8 charts saved to {out_dir}/ (tag: '{tag}')")


if __name__ == "__main__":
    main()
