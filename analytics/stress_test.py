"""
Stress testing & scenario analysis for portfolio resilience.

Defines canonical historical shock periods and replays the strategy's
trade-level P&L through each scenario, applying the existing risk
framework (Kelly sizing, regime gating, hard cap).

The core question: "If today's portfolio had existed during [crisis X],
what would the drawdown, P&L, and survival probability look like?"

Two modes:
  1. **Historical replay** — uses the strategy's own trade distribution
     but applies scenario-specific vol multipliers and win-rate haircuts.
  2. **Parametric shock** — user-defined single-trade or multi-trade
     shock with configurable magnitude.

All results feed into the ``stress_test_chart()`` visualisation.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("nyx.layer5.stress_test")


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario Definitions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Scenario:
    """A historical or parametric stress scenario."""
    name: str
    description: str
    vol_multiplier: float      # scales trade P&L dispersion
    win_rate_haircut: float    # subtracted from base win rate (0–1)
    drawdown_shock: float      # one-time equity shock (fraction, e.g. -0.15)
    duration_trades: int       # how many trades the crisis spans
    regime: str                # expected vol regime during scenario ("high")


# Canonical historical scenarios — calibrated from real market data
HISTORICAL_SCENARIOS: Dict[str, Scenario] = {
    "covid_crash_2020": Scenario(
        name="COVID Crash (Mar 2020)",
        description="VIX spike to 82, SPX -34% in 23 trading days",
        vol_multiplier=3.5,
        win_rate_haircut=0.20,
        drawdown_shock=-0.08,
        duration_trades=8,
        regime="high",
    ),
    "rate_hike_2022": Scenario(
        name="Rate Hike Regime (2022)",
        description="Fed tightening, SPX -27%, sustained high vol for months",
        vol_multiplier=2.0,
        win_rate_haircut=0.12,
        drawdown_shock=-0.04,
        duration_trades=25,
        regime="high",
    ),
    "volmageddon_2018": Scenario(
        name="Volmageddon (Feb 2018)",
        description="XIV collapse, VIX +115% in 1 day, vol sellers wiped out",
        vol_multiplier=4.0,
        win_rate_haircut=0.25,
        drawdown_shock=-0.12,
        duration_trades=5,
        regime="high",
    ),
    "flash_crash_2010": Scenario(
        name="Flash Crash (May 2010)",
        description="SPX -9% intraday, options spreads blown through",
        vol_multiplier=3.0,
        win_rate_haircut=0.15,
        drawdown_shock=-0.06,
        duration_trades=3,
        regime="high",
    ),
    "gfc_2008": Scenario(
        name="GFC (Sep-Nov 2008)",
        description="Lehman collapse, VIX to 80, sustained extreme vol",
        vol_multiplier=4.0,
        win_rate_haircut=0.30,
        drawdown_shock=-0.15,
        duration_trades=30,
        regime="high",
    ),
    "mild_correction": Scenario(
        name="Mild Correction (-10%)",
        description="Typical correction, moderate vol increase",
        vol_multiplier=1.5,
        win_rate_haircut=0.05,
        drawdown_shock=-0.02,
        duration_trades=10,
        regime="normal",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Stress Test Results
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioResult:
    """Results of running one scenario against the portfolio."""
    scenario_name: str
    description: str
    peak_drawdown: float          # worst drawdown during scenario
    total_pnl_pct: float          # total P&L across scenario trades
    survival: bool                # equity stayed above ruin threshold?
    ending_equity_pct: float      # equity at end as % of starting
    n_trades: int                 # trades simulated in scenario
    worst_single_trade: float     # worst individual trade P&L
    recovery_trades: int          # trades needed to recover (0 = never hit DD)
    regime: str                   # vol regime applied


@dataclass
class StressTestOutput:
    """Complete stress test results across all scenarios."""
    results: Dict[str, ScenarioResult]
    scenario_equities: Dict[str, np.ndarray]  # equity curves per scenario
    base_equity: float
    ruin_threshold: float
    risk_per_trade_pct: float

    def summary_df(self) -> pd.DataFrame:
        """Convert results to a summary DataFrame for display."""
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Scenario": r.scenario_name,
                "Peak DD": r.peak_drawdown,
                "Total P&L": r.total_pnl_pct,
                "Survived": "✓" if r.survival else "✗",
                "End Equity %": r.ending_equity_pct,
                "Trades": r.n_trades,
                "Worst Trade": r.worst_single_trade,
                "Recovery": r.recovery_trades if r.recovery_trades > 0 else "N/A",
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Core Engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_stress_test(
    pnl_pcts: np.ndarray,
    scenarios: Optional[Dict[str, Scenario]] = None,
    initial_capital: float = 100_000,
    risk_per_trade_pct: float = 1.0,
    ruin_threshold: float = -0.50,
    regime_scale: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> StressTestOutput:
    """Run all stress scenarios against the portfolio's trade distribution.

    For each scenario:
      1. Apply the initial drawdown shock to starting equity.
      2. Resample ``duration_trades`` trades from the observed P&L
         distribution, with:
         - P&L scaled by ``vol_multiplier``
         - Win rate reduced by ``win_rate_haircut``
         - Position sizes scaled by ``regime_gate`` for the scenario's regime
      3. Build an equity curve and compute metrics.

    Args:
        pnl_pcts: Array of observed per-trade returns (pnl / risk).
        scenarios: Dict of scenario_key → Scenario.  Defaults to HISTORICAL_SCENARIOS.
        initial_capital: Starting equity.
        risk_per_trade_pct: % of equity risked per trade (pre-regime-scaling).
        ruin_threshold: Drawdown level considered catastrophic.
        regime_scale: Dict mapping regime → sizing multiplier.
            Default: {"low": 1.0, "normal": 0.75, "high": 0.50}.
        seed: For reproducibility.

    Returns:
        StressTestOutput with results for each scenario.
    """
    if scenarios is None:
        scenarios = HISTORICAL_SCENARIOS

    if regime_scale is None:
        regime_scale = {"low": 1.0, "normal": 0.75, "high": 0.50}

    rng = np.random.default_rng(seed)

    if len(pnl_pcts) == 0:
        log.warning("Stress test: no trades to analyse")
        return StressTestOutput(
            results={},
            scenario_equities={},
            base_equity=initial_capital,
            ruin_threshold=ruin_threshold,
            risk_per_trade_pct=risk_per_trade_pct,
        )

    results: Dict[str, ScenarioResult] = {}
    equities: Dict[str, np.ndarray] = {}

    for key, scenario in scenarios.items():
        result, eq_curve = _run_single_scenario(
            pnl_pcts=pnl_pcts,
            scenario=scenario,
            initial_capital=initial_capital,
            risk_per_trade_pct=risk_per_trade_pct,
            ruin_threshold=ruin_threshold,
            regime_scale=regime_scale,
            rng=rng,
        )
        results[key] = result
        equities[key] = eq_curve

        log.info(
            f"Stress [{scenario.name}]: DD={result.peak_drawdown:.1%}, "
            f"P&L={result.total_pnl_pct:.1%}, survived={result.survival}"
        )

    return StressTestOutput(
        results=results,
        scenario_equities=equities,
        base_equity=initial_capital,
        ruin_threshold=ruin_threshold,
        risk_per_trade_pct=risk_per_trade_pct,
    )


def _run_single_scenario(
    pnl_pcts: np.ndarray,
    scenario: Scenario,
    initial_capital: float,
    risk_per_trade_pct: float,
    ruin_threshold: float,
    regime_scale: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[ScenarioResult, np.ndarray]:
    """Simulate a single stress scenario.

    Steps:
      1. Apply initial drawdown shock to equity.
      2. Resample trades with modified P&L distribution.
      3. Apply regime-scaled position sizing.
      4. Build equity curve and compute scenario metrics.
    """
    n_trades = scenario.duration_trades
    n_observed = len(pnl_pcts)

    # ── Step 1: Apply initial shock ──────────────────────────────────────
    shocked_equity = initial_capital * (1.0 + scenario.drawdown_shock)

    # ── Step 2: Resample trades with scenario modifications ──────────────
    # Separate winners and losers
    winners = pnl_pcts[pnl_pcts > 0]
    losers = pnl_pcts[pnl_pcts <= 0]

    base_win_rate = float((pnl_pcts > 0).mean()) if len(pnl_pcts) > 0 else 0.5
    scenario_win_rate = max(0.05, base_win_rate - scenario.win_rate_haircut)

    # Generate scenario trades
    scenario_pnls = np.empty(n_trades)
    for i in range(n_trades):
        is_winner = rng.random() < scenario_win_rate

        if is_winner and len(winners) > 0:
            base_pnl = rng.choice(winners)
            # Winners are dampened in crisis (harder to profit)
            scenario_pnls[i] = base_pnl / scenario.vol_multiplier
        elif len(losers) > 0:
            base_pnl = rng.choice(losers)
            # Losses are amplified by vol multiplier
            scenario_pnls[i] = base_pnl * scenario.vol_multiplier
        else:
            # Fallback: generate a loss proportional to vol
            scenario_pnls[i] = -0.30 * scenario.vol_multiplier

    # Clip to reasonable bounds (max loss is -1.0 = total loss of risk capital)
    scenario_pnls = np.clip(scenario_pnls, -1.0, 5.0)

    # ── Step 3: Build equity curve with regime-scaled sizing ─────────────
    regime_mult = regime_scale.get(scenario.regime, 0.75)
    effective_risk_pct = risk_per_trade_pct * regime_mult

    equity = np.empty(n_trades + 1)
    equity[0] = shocked_equity

    for i, pnl in enumerate(scenario_pnls):
        risk_dollars = equity[i] * (effective_risk_pct / 100.0)
        equity[i + 1] = equity[i] + risk_dollars * pnl

    # ── Step 4: Compute metrics ──────────────────────────────────────────
    peak = np.maximum.accumulate(
        np.concatenate([[initial_capital], equity])
    )
    dd_from_original = (equity - peak[1:]) / np.where(peak[1:] > 0, peak[1:], 1.0)

    # Include the initial shock in the drawdown calculation
    all_equity = np.concatenate([[initial_capital], equity])
    all_peak = np.maximum.accumulate(all_equity)
    all_dd = (all_equity - all_peak) / np.where(all_peak > 0, all_peak, 1.0)
    peak_drawdown = float(np.min(all_dd))

    total_pnl_pct = (equity[-1] / initial_capital) - 1.0
    ending_equity_pct = equity[-1] / initial_capital
    survival = peak_drawdown > ruin_threshold
    worst_trade = float(np.min(scenario_pnls))

    # Recovery: how many trades after max DD to get back to pre-crisis equity
    recovery_trades = 0
    dd_idx = int(np.argmin(all_dd))
    if dd_idx < len(all_equity):
        pre_crisis = initial_capital
        for j in range(dd_idx, len(all_equity)):
            if all_equity[j] >= pre_crisis:
                recovery_trades = j - dd_idx
                break
        else:
            recovery_trades = -1  # never recovered within scenario

    return ScenarioResult(
        scenario_name=scenario.name,
        description=scenario.description,
        peak_drawdown=peak_drawdown,
        total_pnl_pct=total_pnl_pct,
        survival=survival,
        ending_equity_pct=ending_equity_pct,
        n_trades=n_trades,
        worst_single_trade=worst_trade,
        recovery_trades=recovery_trades,
        regime=scenario.regime,
    ), equity


def create_custom_scenario(
    name: str,
    description: str,
    vol_multiplier: float = 2.0,
    win_rate_haircut: float = 0.10,
    drawdown_shock: float = -0.05,
    duration_trades: int = 10,
    regime: str = "high",
) -> Scenario:
    """Create a custom stress scenario for ad-hoc testing."""
    return Scenario(
        name=name,
        description=description,
        vol_multiplier=vol_multiplier,
        win_rate_haircut=win_rate_haircut,
        drawdown_shock=drawdown_shock,
        duration_trades=duration_trades,
        regime=regime,
    )
