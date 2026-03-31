"""
Bootstrap Monte Carlo for EOD options strategies.

Resamples at the TRADE level (not day level) because:
  - Trade durations vary (7–35 days)
  - Trades overlap in calendar time
  - Options P&L is path-dependent within a trade

Supports:
  - IID trade resample (block_size=1)
  - Stationary bootstrap (block_size>1, Politis & Romano 1994)
    — geometrically-distributed random block lengths preserve
      stationarity while capturing streak autocorrelation.
      block_size is the *mean* block length.
  - Fixed-fraction position sizing for equity curve construction
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class BootstrapResult:
    metric_name: str
    observed: float
    mean: float
    median: float
    ci_5: float          # 5th percentile
    ci_95: float         # 95th percentile
    prob_negative: float # P(metric < 0)


@dataclass
class BootstrapOutput:
    """Full bootstrap output including metric results AND simulated paths.

    Attributes:
        results: Dict of metric name → BootstrapResult.
        observed_equity: Equity curve for the observed trade series.
        simulated_equities: List of simulated equity curves (np arrays).
            Each has length = n_trades_per_sim + 1.
        simulated_pnls: List of resampled P&L arrays per simulation.
    """
    results: Dict[str, BootstrapResult]
    observed_equity: np.ndarray
    simulated_equities: List[np.ndarray]
    simulated_pnls: List[np.ndarray]


def bootstrap_trade_series(
    pnl_pcts: np.ndarray,
    n_simulations: int = 10_000,
    n_trades_per_sim: Optional[int] = None,
    initial_capital: float = 100_000,
    risk_per_trade_pct: float = 1.0,
    block_size: int = 1,
    ruin_threshold: float = -0.50,
    seed: int = 42,
    store_paths: bool = False,
) -> BootstrapOutput:
    """Run trade-level bootstrap Monte Carlo.

    Uses the stationary bootstrap (Politis & Romano 1994) when
    ``block_size > 1``: block lengths are drawn from a geometric
    distribution with mean ``block_size``, and the series wraps
    circularly so no blocks are truncated.

    Args:
        pnl_pcts: Array of per-trade returns (pnl / risk).
        n_simulations: Number of Monte Carlo paths.
        n_trades_per_sim: Trades per path (default: same as observed).
        initial_capital: Starting equity for each simulation.
        risk_per_trade_pct: % of current equity risked per trade.
        block_size: Mean block length (1 = iid, >1 = stationary bootstrap).
        ruin_threshold: Max drawdown level considered "ruin".
        seed: For reproducibility.
        store_paths: If True, retain all simulated equity curves & pnl arrays.
            Required for MC fan charts.  Can be memory-heavy for large n_sims.

    Returns:
        BootstrapOutput containing metric results and (optionally) simulated paths.
    """
    rng = np.random.default_rng(seed)
    n_observed = len(pnl_pcts)
    n_trades = n_trades_per_sim or n_observed

    if n_observed == 0:
        empty = BootstrapResult("empty", 0, 0, 0, 0, 0, 0)
        return BootstrapOutput(
            results={"total_return": empty},
            observed_equity=np.array([initial_capital]),
            simulated_equities=[],
            simulated_pnls=[],
        )

    # Observed metrics
    obs_eq = _build_equity_curve(pnl_pcts, initial_capital, risk_per_trade_pct)
    obs_metrics = _compute_metrics(obs_eq, pnl_pcts, initial_capital, ruin_threshold)

    # Simulations
    sim_metrics: Dict[str, List[float]] = {k: [] for k in obs_metrics}
    sim_equities: List[np.ndarray] = []
    sim_pnl_arrays: List[np.ndarray] = []

    for _ in range(n_simulations):
        if block_size <= 1:
            idx = rng.integers(0, n_observed, size=n_trades)
            sim_pnls = pnl_pcts[idx]
        else:
            sim_pnls = _stationary_resample(rng, pnl_pcts, n_trades, block_size)

        eq = _build_equity_curve(sim_pnls, initial_capital, risk_per_trade_pct)
        metrics = _compute_metrics(eq, sim_pnls, initial_capital, ruin_threshold)
        for k, v in metrics.items():
            sim_metrics[k].append(v)

        if store_paths:
            sim_equities.append(eq)
            sim_pnl_arrays.append(sim_pnls)

    # Build results
    results: Dict[str, BootstrapResult] = {}
    for k in obs_metrics:
        arr = np.array(sim_metrics[k])
        results[k] = BootstrapResult(
            metric_name=k,
            observed=obs_metrics[k],
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            ci_5=float(np.percentile(arr, 5)),
            ci_95=float(np.percentile(arr, 95)),
            prob_negative=float((arr < 0).mean()),
        )

    return BootstrapOutput(
        results=results,
        observed_equity=obs_eq,
        simulated_equities=sim_equities,
        simulated_pnls=sim_pnl_arrays,
    )


def _stationary_resample(
    rng: np.random.Generator,
    pnl_pcts: np.ndarray,
    n_trades: int,
    mean_block_size: int,
) -> np.ndarray:
    """Stationary bootstrap resampling (Politis & Romano 1994).

    At each step, with probability ``p = 1/mean_block_size`` we jump to
    a new random start position; otherwise we advance to the next element.
    The series wraps circularly so blocks are never truncated.

    Returns an array of length ``n_trades``.
    """
    n = len(pnl_pcts)
    p = 1.0 / mean_block_size
    result = np.empty(n_trades)

    # Start at a random position
    pos = rng.integers(0, n)

    for i in range(n_trades):
        result[i] = pnl_pcts[pos % n]
        # With probability p, jump to a new random start
        if rng.random() < p:
            pos = rng.integers(0, n)
        else:
            pos += 1

    return result


def _build_equity_curve(
    pnl_pcts: np.ndarray,
    initial_capital: float,
    risk_pct: float,
) -> np.ndarray:
    """Build equity curve with fixed-fraction position sizing."""
    equity = np.empty(len(pnl_pcts) + 1)
    equity[0] = initial_capital
    for i, ret in enumerate(pnl_pcts):
        risk_dollars = equity[i] * (risk_pct / 100.0)
        equity[i + 1] = equity[i] + risk_dollars * ret
    return equity


def _compute_metrics(
    equity: np.ndarray,
    pnl_pcts: np.ndarray,
    initial_capital: float,
    ruin_threshold: float,
) -> Dict[str, float]:
    """Standard performance metrics from equity curve + trade returns."""
    total_return = (equity[-1] / initial_capital) - 1.0
    n_trades = len(pnl_pcts)

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(np.min(drawdowns))

    # Win rate
    win_rate = float((pnl_pcts > 0).mean()) if n_trades > 0 else 0.0

    # Profit factor
    gross_profit = float(pnl_pcts[pnl_pcts > 0].sum()) if (pnl_pcts > 0).any() else 0.0
    gross_loss = float(np.abs(pnl_pcts[pnl_pcts < 0].sum())) if (pnl_pcts < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    # Expectancy
    expectancy = float(pnl_pcts.mean()) if n_trades > 0 else 0.0

    # Trade Sharpe (mean / std of per-trade returns)
    trade_sharpe = float(pnl_pcts.mean() / pnl_pcts.std()) if n_trades > 1 and pnl_pcts.std() > 0 else 0.0

    # Ruin flag
    ruin_flag = float(max_dd <= ruin_threshold)

    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "trade_sharpe": trade_sharpe,
        "ruin_flag": ruin_flag,
        "n_trades": float(n_trades),
    }
