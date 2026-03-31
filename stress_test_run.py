"""
Stress Test Runner — RKHS 4fast/2slow Two-Stage Kalman Backtest
===============================================================
Loads backtest results, applies 6 historical crisis scenarios to the
strategy returns, computes resilience metrics, and generates a
Midnight Ocean themed stress test chart.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from analytics.stress_test import (
    Scenario, run_stress_test, HISTORICAL_SCENARIOS,
)
from reporting.theme import (
    ocean_style, SERIES_COLORS, OCEAN_DARK, OCEAN_SURFACE, OCEAN_GRID,
    OCEAN_TEXT, OCEAN_MUTED, OCEAN_TEAL, OCEAN_CORAL, OCEAN_GOLD,
    OCEAN_CYAN, OCEAN_SEA, OCEAN_BLUE, WIN_COLOR, LOSS_COLOR,
    gradient_fill, add_watermark, format_pct,
)


# ── Configuration ────────────────────────────────────────────────────────────

NPZ_PATH = sys.argv[1] if len(sys.argv) > 1 else "backtest_results_2fast_3slow_twostage_v2_kalman_wu60.npz"
_tag = os.path.basename(NPZ_PATH).replace("backtest_results_", "").replace(".npz", "")
OUTPUT_PATH = f"figures/bt_10_stress_test_{_tag}.png"
INITIAL_CAPITAL = 100_000


# ── Define the 6 requested scenarios ─────────────────────────────────────────

SCENARIOS = {
    "covid_crash_2020": Scenario(
        name="COVID Crash (Mar 2020)",
        description="Sudden 35% drawdown in 3 weeks, VIX to 82",
        vol_multiplier=3.5,
        win_rate_haircut=0.20,
        drawdown_shock=-0.08,
        duration_trades=15,   # ~3 weeks of daily trading
        regime="high",
    ),
    "negative_oil_2020": Scenario(
        name="Negative Oil (Apr 2020)",
        description="CL went negative, extreme vol across commodities",
        vol_multiplier=5.0,
        win_rate_haircut=0.25,
        drawdown_shock=-0.10,
        duration_trades=10,
        regime="high",
    ),
    "volmageddon_2018": Scenario(
        name="Volmageddon (Feb 2018)",
        description="VIX spike from 10 to 50, XIV collapse",
        vol_multiplier=4.0,
        win_rate_haircut=0.25,
        drawdown_shock=-0.12,
        duration_trades=5,
        regime="high",
    ),
    "gfc_2008": Scenario(
        name="GFC (Sep-Oct 2008)",
        description="Sustained multi-week drawdown, VIX to 80",
        vol_multiplier=4.0,
        win_rate_haircut=0.30,
        drawdown_shock=-0.15,
        duration_trades=30,
        regime="high",
    ),
    "flash_crash_2010": Scenario(
        name="Flash Crash (May 2010)",
        description="SPX -9% intraday, rapid recovery",
        vol_multiplier=3.0,
        win_rate_haircut=0.15,
        drawdown_shock=-0.06,
        duration_trades=3,
        regime="high",
    ),
    "rate_shock_2022": Scenario(
        name="2022 Rate Shock",
        description="Sustained trend reversal, Fed tightening, SPX -27%",
        vol_multiplier=2.0,
        win_rate_haircut=0.12,
        drawdown_shock=-0.04,
        duration_trades=25,
        regime="high",
    ),
}


# ── Load data ────────────────────────────────────────────────────────────────

print("=" * 72)
print("  RKHS STRESS TEST — 4fast/2slow Two-Stage Kalman")
print("=" * 72)

data = np.load(NPZ_PATH, allow_pickle=True)
strat_returns = data["strat_returns"]

print(f"\nLoaded {len(strat_returns)} daily strategy returns")
print(f"  Mean daily return: {strat_returns.mean():.4%}")
print(f"  Std daily return:  {strat_returns.std():.4%}")
print(f"  Min daily return:  {strat_returns.min():.4%}")
print(f"  Max daily return:  {strat_returns.max():.4%}")
ann_sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252) if strat_returns.std() > 0 else 0
print(f"  Annualized Sharpe: {ann_sharpe:.2f}")


# ── Run stress test ──────────────────────────────────────────────────────────

print("\n" + "-" * 72)
print("  Running stress scenarios...")
print("-" * 72)

output = run_stress_test(
    pnl_pcts=strat_returns,
    scenarios=SCENARIOS,
    initial_capital=INITIAL_CAPITAL,
    risk_per_trade_pct=1.0,
    ruin_threshold=-0.50,
    seed=42,
)


# ── Compute per-scenario metrics matching the request ────────────────────────
# The run_stress_test engine uses trade-level simulation.  We also compute
# the requested metrics: max DD, time to recovery, worst single-day loss,
# Sharpe during stress period.

print("\n" + "=" * 72)
print("  STRESS TEST RESULTS")
print("=" * 72)

header = (
    f"{'Scenario':<30s}  {'Max DD':>8s}  {'Recovery':>10s}  "
    f"{'Worst Day':>10s}  {'Stress Sharpe':>14s}  {'Survived':>8s}"
)
print(header)
print("-" * len(header))

scenario_metrics = {}
for key, result in output.results.items():
    eq = output.scenario_equities[key]

    # Max portfolio drawdown (from initial capital through stress)
    full_eq = np.concatenate([[INITIAL_CAPITAL], eq])
    running_max = np.maximum.accumulate(full_eq)
    dd_series = (full_eq - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_dd = dd_series.min()

    # Time to recovery (trades after trough to regain pre-crisis equity)
    trough_idx = np.argmin(dd_series)
    recovery = "Never"
    recovery_n = -1
    for j in range(trough_idx, len(full_eq)):
        if full_eq[j] >= INITIAL_CAPITAL:
            recovery_n = j - trough_idx
            recovery = f"{recovery_n} trades"
            break

    # Worst single-day (trade) loss
    daily_rets = np.diff(full_eq) / full_eq[:-1]
    worst_day = daily_rets.min() if len(daily_rets) > 0 else 0.0

    # Sharpe during stress period
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        stress_sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        stress_sharpe = 0.0

    scenario_metrics[key] = {
        "name": result.scenario_name,
        "max_dd": max_dd,
        "recovery": recovery,
        "recovery_n": recovery_n,
        "worst_day": worst_day,
        "stress_sharpe": stress_sharpe,
        "survived": result.survival,
        "equity": eq,
    }

    survived_str = "YES" if result.survival else "NO"
    print(
        f"{result.scenario_name:<30s}  {max_dd:>8.2%}  {recovery:>10s}  "
        f"{worst_day:>10.2%}  {stress_sharpe:>14.2f}  {survived_str:>8s}"
    )

print("-" * len(header))
print(f"\nInitial capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Ruin threshold:  -50%")
all_survived = all(m["survived"] for m in scenario_metrics.values())
print(f"All scenarios survived: {'YES' if all_survived else 'NO'}")


# ── Generate Midnight Ocean stress test chart ────────────────────────────────

print("\n" + "-" * 72)
print("  Generating stress test chart...")
print("-" * 72)

# Colors for each scenario
scenario_colors = [
    OCEAN_CORAL,    # COVID — red/coral for severity
    "#E040FB",      # Negative Oil — magenta for extreme
    OCEAN_GOLD,     # Volmageddon — gold
    "#EF5350",      # GFC — deep red
    OCEAN_CYAN,     # Flash Crash — cyan
    OCEAN_BLUE,     # Rate Shock — blue
]

with ocean_style():
    fig = plt.figure(figsize=(18, 12))

    # Layout: top row = equity curves (large), bottom row = bar charts
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.6, 1], hspace=0.32, wspace=0.28,
        left=0.07, right=0.95, top=0.91, bottom=0.07,
    )

    # ── Panel 1: Equity curves under each scenario (top-left, wide) ──
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_title("Portfolio Equity Under Stress Scenarios", fontsize=14, fontweight="bold")

    for i, (key, metrics) in enumerate(scenario_metrics.items()):
        eq = metrics["equity"]
        full_eq = np.concatenate([[INITIAL_CAPITAL], eq])
        norm_eq = full_eq / INITIAL_CAPITAL  # normalize to 1.0
        trades = np.arange(len(full_eq))
        color = scenario_colors[i % len(scenario_colors)]
        ax_eq.plot(trades, norm_eq, color=color, linewidth=2.0,
                   label=metrics["name"], alpha=0.9)

    # Baseline
    ax_eq.axhline(1.0, color=OCEAN_MUTED, linewidth=0.8, linestyle="--", alpha=0.5, label="Initial Capital")
    # Ruin threshold
    ax_eq.axhline(0.5, color=LOSS_COLOR, linewidth=1.0, linestyle=":", alpha=0.6, label="Ruin Threshold (-50%)")

    ax_eq.set_xlabel("Trades into Scenario", fontsize=10)
    ax_eq.set_ylabel("Equity (normalized to 1.0)", fontsize=10)
    ax_eq.legend(loc="lower left", fontsize=8, ncol=2)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    # ── Panel 2: Max Drawdown bars (bottom-left) ──
    ax_dd = fig.add_subplot(gs[1, 0])
    ax_dd.set_title("Maximum Drawdown by Scenario", fontsize=12, fontweight="bold")

    names = [m["name"] for m in scenario_metrics.values()]
    max_dds = [m["max_dd"] for m in scenario_metrics.values()]
    y_pos = np.arange(len(names))

    bars = ax_dd.barh(y_pos, [abs(d) * 100 for d in max_dds],
                      color=scenario_colors[:len(names)], alpha=0.85, edgecolor="none")

    ax_dd.set_yticks(y_pos)
    ax_dd.set_yticklabels(names, fontsize=8)
    ax_dd.set_xlabel("Max Drawdown (%)", fontsize=10)
    ax_dd.invert_yaxis()

    # Add value labels
    for bar, dd in zip(bars, max_dds):
        ax_dd.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                   f"{abs(dd):.1%}", va="center", fontsize=8, color=OCEAN_TEXT)

    # ── Panel 3: Stress Sharpe + Worst Day (bottom-right) ──
    ax_sharpe = fig.add_subplot(gs[1, 1])
    ax_sharpe.set_title("Stress Period Sharpe Ratio & Worst Day Loss", fontsize=12, fontweight="bold")

    sharpes = [m["stress_sharpe"] for m in scenario_metrics.values()]
    worst_days = [abs(m["worst_day"]) * 100 for m in scenario_metrics.values()]

    x_pos = np.arange(len(names))
    width = 0.35

    bars1 = ax_sharpe.bar(x_pos - width / 2, sharpes, width,
                          color=OCEAN_TEAL, alpha=0.85, label="Stress Sharpe (ann.)")
    bars2 = ax_sharpe.bar(x_pos + width / 2, worst_days, width,
                          color=OCEAN_CORAL, alpha=0.85, label="Worst Day Loss (%)")

    ax_sharpe.set_xticks(x_pos)
    ax_sharpe.set_xticklabels([n.split("(")[0].strip() for n in names],
                               fontsize=7, rotation=25, ha="right")
    ax_sharpe.legend(fontsize=8, loc="upper right")
    ax_sharpe.axhline(0, color=OCEAN_MUTED, linewidth=0.5, alpha=0.5)

    # ── Suptitle ──
    fig.suptitle(
        "RKHS Stress Test — 4 Fast / 2 Slow / Two-Stage Kalman",
        fontsize=16, fontweight="bold", color=OCEAN_TEAL, y=0.97,
    )

    add_watermark(fig, text="RKHS", alpha=0.03)

    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

print(f"\nChart saved to: {OUTPUT_PATH}")
print("\nDone.")
