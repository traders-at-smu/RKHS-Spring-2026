"""
Alpha, beta, and Information Ratio — strategy vs. benchmark.

Handles the same sparse-return pitfalls as correlation.py:
  - P&L attributed to exit date
  - Resampled to weekly frequency
  - Missing weeks filled with 0 (no trade = no return)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analytics.trade_simulator import TradeResult


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class AlphaStats:
    """Results container for alpha analysis."""

    annualised_alpha: float       # Jensen's alpha, annualised (×52)
    beta: float                   # Cov(strat, bench) / Var(bench)
    information_ratio: float      # mean(excess) / std(excess) × √52
    n_periods: int                # overlapping weekly periods used
    strategy_total_return: float  # cumulative strategy return
    benchmark_total_return: float # cumulative benchmark return


# ── Return alignment ─────────────────────────────────────────────────────────


def align_strategy_to_benchmark(
    trades: List[TradeResult],
    benchmark_closes: pd.DataFrame,
    initial_capital: float = 100_000,
    freq: str = "W",
) -> pd.DataFrame:
    """Align strategy weekly returns with benchmark weekly returns.

    Strategy returns are dollar P&L (attributed to exit date) divided by
    initial capital.  Benchmark returns are close-to-close percentage
    changes at the same frequency.

    Args:
        trades: TradeResult list from backtest.
        benchmark_closes: DataFrame with DatetimeIndex and ``close`` column
            (output of ``fetch_daily_bars``).
        initial_capital: Starting equity for return-fraction conversion.
        freq: Resample frequency (``"W"`` recommended).

    Returns:
        DataFrame with columns ``['strategy', 'benchmark']`` and a
        DatetimeIndex covering the full trade period.
    """
    # ── Strategy returns: attribute P&L to exit date ──
    daily_pnl: Dict[pd.Timestamp, float] = {}
    for t in trades:
        exit_dt = pd.Timestamp(t.exit_date)
        daily_pnl[exit_dt] = daily_pnl.get(exit_dt, 0.0) + t.pnl_dollars

    if not daily_pnl:
        return pd.DataFrame(columns=["strategy", "benchmark"])

    strat_s = pd.Series(daily_pnl).sort_index()
    strat_s = strat_s.resample(freq).sum()
    strat_s = strat_s / initial_capital  # convert to return fraction

    # ── Benchmark returns: weekly close-to-close ──
    bench_weekly = benchmark_closes["close"].resample(freq).last()
    bench_ret = bench_weekly.pct_change().dropna()

    # ── Align on common date range ──
    combined = pd.DataFrame({
        "strategy": strat_s,
        "benchmark": bench_ret,
    })
    combined = combined.fillna(0.0)

    # Trim to trade activity window
    first_trade = strat_s.index.min()
    last_trade = strat_s.index.max()
    combined = combined.loc[first_trade:last_trade]

    return combined


# ── Alpha / beta / IR computation ────────────────────────────────────────────


def compute_alpha_stats(
    returns_df: pd.DataFrame,
    min_periods: int = 13,
    annual_factor: int = 52,
) -> Optional[AlphaStats]:
    """Compute Jensen's alpha, beta, and Information Ratio.

    Args:
        returns_df: Output of :func:`align_strategy_to_benchmark` with
            ``strategy`` and ``benchmark`` columns.
        min_periods: Minimum weeks required for reliable statistics.
        annual_factor: Annualisation multiplier (52 for weekly data).

    Returns:
        :class:`AlphaStats`, or ``None`` if insufficient data.
    """
    if len(returns_df) < min_periods:
        return None

    strat = returns_df["strategy"].values
    bench = returns_df["benchmark"].values

    # Beta = Cov(strat, bench) / Var(bench)
    cov_matrix = np.cov(strat, bench)
    var_bench = cov_matrix[1, 1]
    beta = float(cov_matrix[0, 1] / var_bench) if var_bench > 0 else 0.0

    # Alpha (weekly) = mean(strat) - beta × mean(bench), then annualise
    alpha_weekly = np.mean(strat) - beta * np.mean(bench)
    alpha_annual = float(alpha_weekly * annual_factor)

    # Information Ratio = mean(excess) / std(excess) × √(annual_factor)
    excess = strat - bench
    excess_std = float(np.std(excess, ddof=1))
    ir = (float(np.mean(excess)) / excess_std * np.sqrt(annual_factor)
          if excess_std > 0 else 0.0)

    # Cumulative returns for display
    strat_cum = float((1 + pd.Series(strat)).cumprod().iloc[-1] - 1)
    bench_cum = float((1 + pd.Series(bench)).cumprod().iloc[-1] - 1)

    return AlphaStats(
        annualised_alpha=alpha_annual,
        beta=beta,
        information_ratio=ir,
        n_periods=len(returns_df),
        strategy_total_return=strat_cum,
        benchmark_total_return=bench_cum,
    )


# ── Cumulative curve helpers ─────────────────────────────────────────────────


def build_cumulative_curves(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative return series for plotting.

    Returns DataFrame with columns:
      - ``strategy_cum`` — cumulative strategy return (starts at 0)
      - ``benchmark_cum`` — cumulative benchmark return (starts at 0)
      - ``alpha_spread`` — strategy_cum − benchmark_cum
    """
    df = returns_df.copy()
    df["strategy_cum"] = (1 + df["strategy"]).cumprod() - 1
    df["benchmark_cum"] = (1 + df["benchmark"]).cumprod() - 1
    df["alpha_spread"] = df["strategy_cum"] - df["benchmark_cum"]
    return df


def rolling_alpha(
    returns_df: pd.DataFrame,
    window: int = 13,
    annual_factor: int = 52,
) -> pd.Series:
    """Rolling annualised alpha (trailing *window* weeks).

    Uses excess return (strategy − benchmark) as the alpha proxy.
    Full rolling-beta recomputation would be noisy at small windows.
    """
    excess = returns_df["strategy"] - returns_df["benchmark"]
    return excess.rolling(window).mean() * annual_factor


def rolling_sharpe(
    returns_df: pd.DataFrame,
    window: int = 13,
    annual_factor: int = 52,
    column: str = "strategy",
) -> pd.Series:
    """Rolling annualised Sharpe ratio (trailing *window* weeks).

    Sharpe = mean(r) / std(r) * sqrt(annual_factor).  When the rolling
    standard deviation is zero (constant returns), the result is NaN.

    Args:
        returns_df: DataFrame with at least a *column* column of period returns.
        window: Rolling window in periods.
        annual_factor: 52 for weekly data.
        column: Which column to compute Sharpe for.

    Returns:
        Series of rolling Sharpe values, same index as *returns_df*.
        First (*window* - 1) values will be NaN.
    """
    r = returns_df[column]
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=1)
    # Replace zero std with NaN to avoid inf
    roll_std = roll_std.replace(0.0, np.nan)
    return (roll_mean / roll_std) * np.sqrt(annual_factor)


def rolling_sortino(
    returns_df: pd.DataFrame,
    window: int = 13,
    annual_factor: int = 52,
    column: str = "strategy",
    target: float = 0.0,
) -> pd.Series:
    """Rolling annualised Sortino ratio (trailing *window* weeks).

    Like Sharpe, but only penalises *downside* volatility.  For options
    strategies with asymmetric payoffs (capped loss, uncapped gain),
    Sortino gives a truer risk-adjusted picture because upside swings
    don't inflate the denominator.

    Sortino = mean(r - target) / downside_deviation × √annual_factor

    Args:
        returns_df: DataFrame with at least a *column* column of period returns.
        window: Rolling window in periods.
        annual_factor: 52 for weekly data.
        column: Which column to compute Sortino for.
        target: Minimum acceptable return (default 0).

    Returns:
        Series of rolling Sortino values, same index as *returns_df*.
        First (*window* - 1) values will be NaN.
    """
    r = returns_df[column]
    excess = r - target
    roll_mean = excess.rolling(window).mean()

    # Downside deviation: std of negative excess returns only
    downside = excess.clip(upper=0.0)
    roll_downside_std = downside.rolling(window).std(ddof=1)
    roll_downside_std = roll_downside_std.replace(0.0, np.nan)

    return (roll_mean / roll_downside_std) * np.sqrt(annual_factor)


def rolling_win_rate(
    returns_df: pd.DataFrame,
    window: int = 13,
    column: str = "strategy",
) -> pd.Series:
    """Rolling win rate (fraction of positive-return periods in trailing window).

    Args:
        returns_df: DataFrame with at least a *column* column of period returns.
        window: Rolling window in periods.
        column: Which column to evaluate.

    Returns:
        Series of rolling win rates (0.0–1.0), same index as *returns_df*.
    """
    return (returns_df[column] > 0).astype(float).rolling(window).mean()
