"""
Pristine-RKHS Backtest Report Generator
========================================
Loads saved .npz results and generates:
  1. Equity curve + drawdown chart
  2. Monthly returns heatmap
  3. Trade distribution histogram
  4. Rolling Sharpe (60-day)
  5. Position timeline
  6. Bootstrap Monte Carlo confidence intervals
  7. Summary text report

Usage:
    .venv/bin/python generate_report.py [results_file.npz]
    # Defaults to most recent backtest_results_*.npz
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0d1117",
})

C_GREEN = "#3fb950"
C_RED = "#f85149"
C_BLUE = "#58a6ff"
C_PURPLE = "#bc8cff"
C_ORANGE = "#d29922"
C_GRAY = "#484f58"
C_WHITE = "#c9d1d9"

REPORT_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def load_results(path):
    """Load .npz backtest results."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def fig_equity_drawdown(daily_returns, save_path):
    """Dual-panel: cumulative equity curve + underwater drawdown."""
    cum_ret = np.cumsum(daily_returns)
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = cum_ret - running_max

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax = axes[0]
    t = np.arange(len(cum_ret))
    ax.plot(t, cum_ret, lw=1.5, color=C_BLUE, label="Cumulative P&L")
    ax.fill_between(t, 0, cum_ret, where=(cum_ret >= 0), alpha=0.15, color=C_GREEN)
    ax.fill_between(t, 0, cum_ret, where=(cum_ret < 0), alpha=0.15, color=C_RED)
    ax.axhline(0, color=C_GRAY, lw=0.5)

    # Annotate
    sr = np.mean(daily_returns) / (np.std(daily_returns) + 1e-12) * np.sqrt(252)
    ax.text(0.02, 0.92,
            f"Final P&L: {cum_ret[-1]:.4f} ({cum_ret[-1]*100:.2f}%)\n"
            f"Max DD: {drawdown.min():.4f} ({drawdown.min()*100:.2f}%)\n"
            f"Sharpe: {sr:.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                      edgecolor=C_GRAY, alpha=0.95))
    ax.set_ylabel("Cumulative Return")
    ax.set_title("Pristine-RKHS v2 — Equity Curve", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)

    # Drawdown
    ax = axes[1]
    ax.fill_between(t, 0, drawdown, color=C_RED, alpha=0.5)
    ax.plot(t, drawdown, lw=0.7, color=C_RED)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Trading Day")
    ax.set_title("Underwater Chart", fontweight="bold")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(save_path)}")


def fig_monthly_returns(daily_returns, start_date, save_path):
    """Monthly returns heatmap."""
    dates = pd.date_range(start=start_date, periods=len(daily_returns), freq="B")
    df = pd.DataFrame({"return": daily_returns}, index=dates)
    monthly = df.resample("ME").sum()
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month

    pivot = monthly.pivot_table(values="return", index="year", columns="month",
                                 aggfunc="sum")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 3))
    data = pivot.values * 100  # to percentage

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                color = "black" if abs(data[i, j]) < 2 else "white"
                ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    ax.set_title("Monthly Returns (%)", fontweight="bold", fontsize=13)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Return (%)")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(save_path)}")


def fig_trade_distribution(daily_returns, save_path):
    """Histogram of daily P&L when in position."""
    active = daily_returns[daily_returns != 0]
    if len(active) == 0:
        print(f"  ✗ No active trades for distribution")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    wins = active[active > 0]
    losses = active[active < 0]

    bins = np.linspace(active.min(), active.max(), 40)
    ax.hist(wins * 100, bins=bins * 100, color=C_GREEN, alpha=0.7,
            label=f"Wins: {len(wins)} ({len(wins)/len(active)*100:.0f}%)")
    ax.hist(losses * 100, bins=bins * 100, color=C_RED, alpha=0.7,
            label=f"Losses: {len(losses)} ({len(losses)/len(active)*100:.0f}%)")
    ax.axvline(0, color=C_WHITE, lw=1, ls="--")
    ax.axvline(active.mean() * 100, color=C_BLUE, lw=1.5, ls="-",
               label=f"Mean: {active.mean()*100:.3f}%")

    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Daily Return Distribution (Active Days)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(save_path)}")


def fig_rolling_sharpe(daily_returns, window=60, save_path=""):
    """Rolling Sharpe ratio with confidence band."""
    if len(daily_returns) < window:
        print(f"  ✗ Not enough data for rolling Sharpe")
        return

    rolling_mean = pd.Series(daily_returns).rolling(window).mean()
    rolling_std = pd.Series(daily_returns).rolling(window).std()
    rolling_sr = (rolling_mean / (rolling_std + 1e-12)) * np.sqrt(252)

    fig, ax = plt.subplots(figsize=(14, 5))
    t = np.arange(len(rolling_sr))
    sr_vals = rolling_sr.values

    ax.plot(t, sr_vals, lw=1.2, color=C_PURPLE, label=f"{window}-day Rolling Sharpe")
    ax.fill_between(t, sr_vals, 0,
                    where=(sr_vals > 0), alpha=0.15, color=C_GREEN)
    ax.fill_between(t, sr_vals, 0,
                    where=(sr_vals < 0), alpha=0.15, color=C_RED)
    ax.axhline(0, color=C_GRAY, lw=1)
    ax.axhline(1.0, color=C_GREEN, lw=0.8, ls="--", alpha=0.5, label="SR = 1.0")
    ax.axhline(-1.0, color=C_RED, lw=0.8, ls="--", alpha=0.5, label="SR = -1.0")

    # Overall Sharpe
    overall_sr = np.mean(daily_returns) / (np.std(daily_returns) + 1e-12) * np.sqrt(252)
    ax.axhline(overall_sr, color=C_BLUE, lw=1.5, ls="-", alpha=0.8,
               label=f"Overall SR = {overall_sr:.2f}")

    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.set_title(f"{window}-Day Rolling Sharpe Ratio", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)
    ax.set_ylim(-4, 6)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(save_path)}")


def fig_positions_timeline(positions, s_scores, save_path):
    """Position and s-score overlay."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})

    t = np.arange(len(positions))

    # Positions
    ax = axes[0]
    pos_colors = np.where(positions > 0, C_GREEN, np.where(positions < 0, C_RED, C_GRAY))
    ax.bar(t, positions, color=[C_GREEN if p > 0 else C_RED if p < 0 else C_GRAY
                                  for p in positions], width=1.0, alpha=0.7)
    ax.set_ylabel("Position")
    ax.set_title("Position Timeline", fontweight="bold")
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.grid(True)

    # s-scores
    ax = axes[1]
    s_clean = np.nan_to_num(s_scores[:len(t)], nan=0.0)
    ax.plot(t, s_clean, lw=0.8, color=C_PURPLE, alpha=0.8, label="s-score")
    ax.axhline(0.75, color=C_RED, lw=0.8, ls="--", alpha=0.5, label="Entry")
    ax.axhline(-0.75, color=C_GREEN, lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0.25, color=C_ORANGE, lw=0.8, ls=":", alpha=0.5, label="Exit")
    ax.axhline(-0.25, color=C_ORANGE, lw=0.8, ls=":", alpha=0.5)
    ax.set_ylabel("s-score")
    ax.set_xlabel("Bar Index")
    ax.set_title("s-Score Signal", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(save_path)}")


def run_bootstrap(daily_returns, n_sims=10000, seed=42):
    """Simple bootstrap for Sharpe CI."""
    rng = np.random.RandomState(seed)
    active = daily_returns[daily_returns != 0]
    if len(active) < 10:
        return None

    n = len(active)
    boot_sharpes = np.zeros(n_sims)
    for i in range(n_sims):
        sample = rng.choice(active, size=n, replace=True)
        boot_sharpes[i] = np.mean(sample) / (np.std(sample) + 1e-12) * np.sqrt(252)

    return {
        "observed_sr": np.mean(active) / (np.std(active) + 1e-12) * np.sqrt(252),
        "mean_sr": np.mean(boot_sharpes),
        "median_sr": np.median(boot_sharpes),
        "ci_5": np.percentile(boot_sharpes, 5),
        "ci_95": np.percentile(boot_sharpes, 95),
        "prob_negative": np.mean(boot_sharpes < 0),
        "boot_sharpes": boot_sharpes,
    }


def fig_bootstrap(boot_result, save_path):
    """Bootstrap Sharpe distribution with CI."""
    if boot_result is None:
        print(f"  ✗ Not enough trades for bootstrap")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(boot_result["boot_sharpes"], bins=80, color=C_PURPLE, alpha=0.6,
            edgecolor=C_GRAY, linewidth=0.3)

    # CI lines
    ax.axvline(boot_result["ci_5"], color=C_RED, lw=1.5, ls="--",
               label=f"5th pct: {boot_result['ci_5']:.2f}")
    ax.axvline(boot_result["ci_95"], color=C_GREEN, lw=1.5, ls="--",
               label=f"95th pct: {boot_result['ci_95']:.2f}")
    ax.axvline(boot_result["observed_sr"], color=C_BLUE, lw=2,
               label=f"Observed: {boot_result['observed_sr']:.2f}")
    ax.axvline(0, color=C_RED, lw=1, ls=":", alpha=0.5)

    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bootstrap Sharpe Distribution (n=10,000)\n"
                 f"P(SR < 0) = {boot_result['prob_negative']:.1%}",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ {os.path.basename(save_path)}")


def generate_text_report(results, boot_result, run_label, save_path):
    """Generate summary text report."""
    daily_ret = results["daily_strat_returns"]
    positions = results["positions"]
    s_scores = results["s_scores"]

    active_days = daily_ret[daily_ret != 0]
    sr = np.mean(daily_ret) / (np.std(daily_ret) + 1e-12) * np.sqrt(252)

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-12
    sortino = np.mean(daily_ret) / (downside_std + 1e-12) * np.sqrt(252)

    # Max drawdown
    cum = np.cumsum(daily_ret)
    max_dd = (cum - np.maximum.accumulate(cum)).min()

    # Win/loss
    wins = active_days[active_days > 0]
    losses = active_days[active_days < 0]
    hit_rate = len(wins) / len(active_days) if len(active_days) > 0 else 0
    wl_ratio = (np.mean(wins) / abs(np.mean(losses))) if len(losses) > 0 and len(wins) > 0 else 0
    n_trades = int(np.sum(np.diff((results.get("daily_positions", positions) != 0).astype(int)) > 0))

    report_lines = [
        "=" * 70,
        "  PRISTINE-RKHS BACKTEST REPORT",
        f"  Run: {run_label}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "── PERFORMANCE SUMMARY ─────────────────────────────────────────────",
        f"  Sharpe Ratio:       {sr:.4f}",
        f"  Sortino Ratio:      {sortino:.4f}",
        f"  Max Drawdown:       {max_dd:.4f} ({max_dd*100:.2f}%)",
        f"  Total P&L:          {cum[-1]:.6f} ({cum[-1]*100:.4f}%)",
        f"  Annualized Return:  {np.mean(daily_ret)*252*100:.2f}%",
        f"  Annualized Vol:     {np.std(daily_ret)*np.sqrt(252)*100:.2f}%",
        "",
        "── TRADE STATISTICS ────────────────────────────────────────────────",
        f"  Trading Days:       {len(daily_ret)}",
        f"  Days in Position:   {len(active_days)} ({len(active_days)/len(daily_ret)*100:.1f}%)",
        f"  Estimated Trades:   {n_trades}",
        f"  Hit Rate:           {hit_rate:.1%}",
        f"  Win/Loss Ratio:     {wl_ratio:.2f}x",
        f"  Avg Win:            {np.mean(wins)*100:.4f}%" if len(wins) > 0 else "  Avg Win:            N/A",
        f"  Avg Loss:           {np.mean(losses)*100:.4f}%" if len(losses) > 0 else "  Avg Loss:           N/A",
        f"  Best Day:           {daily_ret.max()*100:.4f}%",
        f"  Worst Day:          {daily_ret.min()*100:.4f}%",
        "",
        "── SIGNAL QUALITY ──────────────────────────────────────────────────",
        f"  s-score range:      [{np.nanmin(s_scores):.3f}, {np.nanmax(s_scores):.3f}]",
        f"  s-score NaN%:       {np.isnan(s_scores).mean():.1%}",
        f"  s-score mean:       {np.nanmean(s_scores):.4f}",
        f"  s-score std:        {np.nanstd(s_scores):.4f}",
    ]

    if boot_result is not None:
        report_lines.extend([
            "",
            "── BOOTSTRAP MONTE CARLO (10,000 sims) ────────────────────────────",
            f"  Observed Sharpe:    {boot_result['observed_sr']:.4f}",
            f"  Bootstrap Mean:     {boot_result['mean_sr']:.4f}",
            f"  Bootstrap Median:   {boot_result['median_sr']:.4f}",
            f"  90% CI:             [{boot_result['ci_5']:.4f}, {boot_result['ci_95']:.4f}]",
            f"  P(Sharpe < 0):      {boot_result['prob_negative']:.1%}",
        ])

    report_lines.extend([
        "",
        "── CONFIGURATION ───────────────────────────────────────────────────",
        f"  Architecture:       v2 (OrderFlow consolidated)",
        f"  Signal Mode:        Kalman (aggressive Q)",
        f"  Warmup Days:        60",
        f"  Vol Sizing:         Enabled (15% target)",
        f"  Fast Kernels:       OrderFlow (PCA 15D), VannaCharm (6D)",
        f"  Slow Kernels:       VRP, Macro, Sentiment (PCA 10D)",
        f"  Combiner:           Walk-forward ElasticNet",
        f"  Position Sizing:    VPIN/Hawkes gates + inverse vol scaling",
        "",
        "── RISK ASSESSMENT ─────────────────────────────────────────────────",
    ])

    # Risk flags
    if sr > 1.0:
        report_lines.append(f"  ✓ Sharpe > 1.0 — acceptable risk-adjusted return")
    else:
        report_lines.append(f"  ✗ Sharpe < 1.0 — marginal risk-adjusted return")

    if abs(max_dd) < 0.05:
        report_lines.append(f"  ✓ Max DD < 5% — tight risk control")
    elif abs(max_dd) < 0.10:
        report_lines.append(f"  ~ Max DD 5-10% — moderate drawdown")
    else:
        report_lines.append(f"  ✗ Max DD > 10% — excessive drawdown")

    if boot_result and boot_result["prob_negative"] < 0.05:
        report_lines.append(f"  ✓ P(SR<0) < 5% — statistically significant")
    elif boot_result and boot_result["prob_negative"] < 0.10:
        report_lines.append(f"  ~ P(SR<0) 5-10% — borderline significance")
    elif boot_result:
        report_lines.append(f"  ✗ P(SR<0) > 10% — not statistically significant")

    if n_trades >= 30:
        report_lines.append(f"  ✓ {n_trades} trades — sufficient sample size")
    else:
        report_lines.append(f"  ⚠ {n_trades} trades — low sample size, interpret with caution")

    report_lines.extend([
        "",
        "=" * 70,
        "  END OF REPORT",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    with open(save_path, "w") as f:
        f.write(report_text)
    print(f"  ✓ {os.path.basename(save_path)}")

    return report_text


def main():
    # Find results file
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Find most recent .npz
        pattern = os.path.join(_ROOT, "backtest_results_*.npz")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not files:
            print("ERROR: No backtest results found. Run run_backtest.py first.")
            sys.exit(1)
        results_file = files[0]

    run_label = os.path.basename(results_file).replace("backtest_results_", "").replace(".npz", "")
    print(f"\n{'=' * 60}")
    print(f"  Generating Report: {run_label}")
    print(f"  Source: {os.path.basename(results_file)}")
    print(f"{'=' * 60}\n")

    results = load_results(results_file)

    # Use daily returns for all metrics
    if "daily_strat_returns" in results:
        daily_ret = results["daily_strat_returns"]
    else:
        daily_ret = results["strat_returns"]

    print("  Generating charts...")

    # 1. Equity curve + drawdown
    fig_equity_drawdown(
        daily_ret,
        os.path.join(REPORT_DIR, f"01_equity_drawdown_{run_label}.png"),
    )

    # 2. Monthly returns heatmap
    fig_monthly_returns(
        daily_ret,
        start_date="2023-07-03",  # CL backtest start
        save_path=os.path.join(REPORT_DIR, f"02_monthly_returns_{run_label}.png"),
    )

    # 3. Trade distribution
    fig_trade_distribution(
        daily_ret,
        os.path.join(REPORT_DIR, f"03_trade_distribution_{run_label}.png"),
    )

    # 4. Rolling Sharpe
    fig_rolling_sharpe(
        daily_ret,
        window=60,
        save_path=os.path.join(REPORT_DIR, f"04_rolling_sharpe_{run_label}.png"),
    )

    # 5. Position timeline
    if "positions" in results and "s_scores" in results:
        fig_positions_timeline(
            results["positions"],
            results["s_scores"],
            os.path.join(REPORT_DIR, f"05_positions_signal_{run_label}.png"),
        )

    # 6. Bootstrap MC
    print("\n  Running bootstrap Monte Carlo (10,000 sims)...")
    boot_result = run_bootstrap(daily_ret, n_sims=10000)
    fig_bootstrap(
        boot_result,
        os.path.join(REPORT_DIR, f"06_bootstrap_sharpe_{run_label}.png"),
    )

    # 7. Text report
    print("\n  Generating text report...")
    report_text = generate_text_report(
        results, boot_result, run_label,
        os.path.join(REPORT_DIR, f"report_{run_label}.txt"),
    )

    print(f"\n{'=' * 60}")
    print(f"  All reports saved to: {REPORT_DIR}/")
    print(f"{'=' * 60}")
    print(f"\n{report_text}")


if __name__ == "__main__":
    main()
