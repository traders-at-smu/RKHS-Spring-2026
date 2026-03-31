"""
Per-strategy health scorecards — automated decay detection.

Answers: "Is each strategy still performing as expected, or is something
breaking down?"

For each strategy, computes rolling health metrics across three tiers:
  1. **Core health** (from ~1 trade): win rate, profit factor, loss streak
  2. **Decay detection** (from ~10–20 trades): Sharpe, win-rate trend,
     baseline deviation, Anderson-Darling drift test
  3. **Regime-conditional** (when features available): performance by
     vol regime, best/worst regime

Each strategy is flagged HEALTHY / WARNING / CRITICAL based on
configurable thresholds.  Designed to run from trade #1 — all metrics
degrade gracefully with fewer observations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("nyx.layer6.scorecard")


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyScorecard:
    """Health scorecard for a single strategy."""
    strategy_name: str
    n_trades: int

    # Tier 1 — Core Health
    rolling_win_rate: Optional[float]
    rolling_profit_factor: Optional[float]
    consecutive_loss_count: int
    avg_pnl_per_trade: float

    # Tier 2 — Decay Detection
    rolling_sharpe: Optional[float]
    rolling_sortino: Optional[float]         # downside-only risk-adjusted return
    win_rate_trend: Optional[float]          # slope; negative = declining
    baseline_win_rate: Optional[float]       # from Strategies.json
    live_vs_baseline_delta: Optional[float]  # live - baseline
    anderson_darling_pvalue: Optional[float]

    # Tier 3 — Regime-Conditional
    win_rate_by_regime: Dict[str, float]
    mean_pnl_by_regime: Dict[str, float]
    best_regime: Optional[str]
    worst_regime: Optional[str]

    # Sparkline data
    rolling_win_rate_series: List[float]

    # Health status
    status: str                              # "HEALTHY" | "WARNING" | "CRITICAL"
    status_reasons: List[str]


@dataclass
class PortfolioScorecard:
    """Aggregated health across all strategies."""
    strategies: Dict[str, StrategyScorecard]
    n_total_trades: int
    n_strategies: int
    n_healthy: int
    n_warning: int
    n_critical: int
    overall_status: str

    def summary_df(self) -> pd.DataFrame:
        """Console-friendly summary table."""
        rows = []
        for name, sc in self.strategies.items():
            rows.append({
                "Strategy": name,
                "Trades": sc.n_trades,
                "Status": sc.status,
                "Win Rate": f"{sc.rolling_win_rate:.0%}" if sc.rolling_win_rate is not None else "n/a",
                "Profit Factor": f"{sc.rolling_profit_factor:.2f}" if sc.rolling_profit_factor is not None else "n/a",
                "Avg P&L": f"{sc.avg_pnl_per_trade:+.1%}" if sc.n_trades > 0 else "n/a",
                "Loss Streak": sc.consecutive_loss_count,
                "Sharpe": f"{sc.rolling_sharpe:.2f}" if sc.rolling_sharpe is not None else "n/a",
                "Sortino": f"{sc.rolling_sortino:.2f}" if sc.rolling_sortino is not None and sc.rolling_sortino != float("inf") else ("inf" if sc.rolling_sortino == float("inf") else "n/a"),
                "vs Baseline": f"{sc.live_vs_baseline_delta:+.0%}" if sc.live_vs_baseline_delta is not None else "n/a",
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Core computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_strategy_scorecard(
    trades: pd.DataFrame,
    strategy_name: str,
    features_df: Optional[pd.DataFrame] = None,
    baseline_config: Optional[Dict] = None,
    trailing_window: int = 20,
) -> StrategyScorecard:
    """Compute health scorecard for a single strategy.

    Args:
        trades: DataFrame filtered to this strategy's trades, sorted by
            exit_date.  Required columns: pnl_pct.  Optional: signal_id.
        strategy_name: Display name for the strategy.
        features_df: Features with vol_regime column, joinable by signal_id.
        baseline_config: Strategy entry from Strategies.json.
        trailing_window: Number of trades for rolling metrics.

    Returns:
        StrategyScorecard with all computable metrics filled, others None.
    """
    n = len(trades)

    if n == 0:
        sc = _empty_scorecard(strategy_name, baseline_config)
        sc.status, sc.status_reasons = "HEALTHY", []
        return sc

    pnls = pd.to_numeric(trades["pnl_pct"], errors="coerce").fillna(0.0).values

    # ── Tier 1: Core Health ───────────────────────────────────────────────
    window = min(trailing_window, n)
    trailing = pnls[-window:]

    rolling_wr = float((trailing > 0).mean())
    avg_pnl = float(pnls.mean())

    # Profit factor
    wins = trailing[trailing > 0]
    losses = trailing[trailing <= 0]
    if len(losses) == 0:
        pf = float("inf") if len(wins) > 0 else 0.0
    elif len(wins) == 0:
        pf = 0.0
    else:
        pf = float(wins.sum() / abs(losses.sum()))

    # Consecutive loss streak (from most recent trade backwards)
    consec_losses = 0
    for p in reversed(pnls):
        if p <= 0:
            consec_losses += 1
        else:
            break

    # Rolling win rate series (at each trade, compute trailing window WR)
    wr_series = []
    for i in range(n):
        w = min(trailing_window, i + 1)
        chunk = pnls[max(0, i + 1 - w):i + 1]
        wr_series.append(float((chunk > 0).mean()))

    # ── Tier 2: Decay Detection ───────────────────────────────────────────
    rolling_sharpe = None
    rolling_sortino = None
    win_rate_trend = None
    baseline_wr = None
    live_vs_baseline = None
    ad_pvalue = None

    if n >= 10:
        trail_mean = float(trailing.mean())
        trail_std = float(trailing.std(ddof=1))
        trades_per_year = _estimate_trades_per_year(trades)
        ann = np.sqrt(max(trades_per_year, 1))
        if trail_std > 0:
            rolling_sharpe = trail_mean / trail_std * ann
        # Sortino: only penalise downside volatility
        downside = trailing.copy()
        downside = np.minimum(downside, 0.0)
        downside_std = float(np.std(downside, ddof=1))
        if downside_std > 0:
            rolling_sortino = trail_mean / downside_std * ann
        elif trail_mean > 0:
            rolling_sortino = float("inf")  # no downside risk, positive mean

    if n >= 20 and len(wr_series) >= 20:
        # Win rate trend: slope over 4 evenly spaced checkpoints
        n_points = min(4, len(wr_series))
        indices = np.linspace(0, len(wr_series) - 1, n_points, dtype=int)
        wr_checkpoints = [wr_series[i] for i in indices]
        if len(wr_checkpoints) >= 2:
            x = np.arange(len(wr_checkpoints), dtype=float)
            coeffs = np.polyfit(x, wr_checkpoints, 1)
            win_rate_trend = float(coeffs[0])

    if baseline_config is not None:
        baseline_wr = baseline_config.get("kelly_win_rate")
        if baseline_wr is not None and rolling_wr is not None and n >= 5:
            live_vs_baseline = rolling_wr - baseline_wr

    if n >= 20 and baseline_config is not None:
        ad_pvalue = _anderson_darling_vs_baseline(trailing, baseline_config)

    # ── Tier 3: Regime-Conditional ────────────────────────────────────────
    wr_by_regime: Dict[str, float] = {}
    pnl_by_regime: Dict[str, float] = {}
    best_regime = None
    worst_regime = None

    if features_df is not None and not features_df.empty:
        regime_trades = _join_regime(trades, features_df)
        if "vol_regime" in regime_trades.columns:
            for regime, group in regime_trades.groupby("vol_regime"):
                regime_str = str(regime)
                if regime_str == "unknown":
                    continue
                rpnls = pd.to_numeric(group["pnl_pct"], errors="coerce").fillna(0.0).values
                if len(rpnls) > 0:
                    wr_by_regime[regime_str] = float((rpnls > 0).mean())
                    pnl_by_regime[regime_str] = float(rpnls.mean())

            if pnl_by_regime:
                best_regime = max(pnl_by_regime, key=pnl_by_regime.get)
                worst_regime = min(pnl_by_regime, key=pnl_by_regime.get)

    # ── Build scorecard ───────────────────────────────────────────────────
    sc = StrategyScorecard(
        strategy_name=strategy_name,
        n_trades=n,
        rolling_win_rate=rolling_wr,
        rolling_profit_factor=pf,
        consecutive_loss_count=consec_losses,
        avg_pnl_per_trade=avg_pnl,
        rolling_sharpe=rolling_sharpe,
        rolling_sortino=rolling_sortino,
        win_rate_trend=win_rate_trend,
        baseline_win_rate=baseline_wr,
        live_vs_baseline_delta=live_vs_baseline,
        anderson_darling_pvalue=ad_pvalue,
        win_rate_by_regime=wr_by_regime,
        mean_pnl_by_regime=pnl_by_regime,
        best_regime=best_regime,
        worst_regime=worst_regime,
        rolling_win_rate_series=wr_series,
        status="HEALTHY",
        status_reasons=[],
    )

    sc.status, sc.status_reasons = _compute_health_status(sc, baseline_config)
    return sc


def compute_portfolio_scorecard(
    trades_df: pd.DataFrame,
    strategies_config: Dict[str, Dict],
    features_df: Optional[pd.DataFrame] = None,
    trailing_window: int = 20,
) -> PortfolioScorecard:
    """Compute scorecards for all strategies found in trades + config.

    New strategies in Strategies.json with no trades yet appear with
    n_trades=0 and status=HEALTHY.

    Args:
        trades_df: All trades with strategy_name, pnl_pct columns.
        strategies_config: Output of schemas.load_strategies().
        features_df: Optional features for regime-conditional analysis.
        trailing_window: Window size for rolling metrics.

    Returns:
        PortfolioScorecard aggregating all strategies.
    """
    # Ensure pnl_pct is numeric
    trades = trades_df.copy()
    if "pnl_pct" in trades.columns:
        trades["pnl_pct"] = pd.to_numeric(trades["pnl_pct"], errors="coerce").fillna(0.0)

    # Collect all strategy names from trades + config
    trade_strats = set()
    if "strategy_name" in trades.columns:
        trade_strats = set(trades["strategy_name"].unique())
    config_strats = set(strategies_config.keys())
    all_strats = trade_strats | config_strats

    scorecards: Dict[str, StrategyScorecard] = {}

    for strat in sorted(all_strats):
        # Filter trades for this strategy
        if "strategy_name" in trades.columns:
            strat_trades = trades[trades["strategy_name"] == strat].copy()
        else:
            strat_trades = pd.DataFrame()

        # Sort by exit date if available
        if not strat_trades.empty and "exit_date" in strat_trades.columns:
            strat_trades = strat_trades.sort_values("exit_date")

        # Find matching config (try exact, then check if trade name is in config key)
        baseline = strategies_config.get(strat)
        if baseline is None:
            for config_key, config_val in strategies_config.items():
                if strat in config_key or config_key in strat:
                    baseline = config_val
                    break

        scorecards[strat] = compute_strategy_scorecard(
            trades=strat_trades,
            strategy_name=strat,
            features_df=features_df,
            baseline_config=baseline,
            trailing_window=trailing_window,
        )

    n_total = sum(sc.n_trades for sc in scorecards.values())
    n_healthy = sum(1 for sc in scorecards.values() if sc.status == "HEALTHY")
    n_warning = sum(1 for sc in scorecards.values() if sc.status == "WARNING")
    n_critical = sum(1 for sc in scorecards.values() if sc.status == "CRITICAL")

    if n_critical > 0:
        overall = "CRITICAL"
    elif n_warning > 0:
        overall = "WARNING"
    else:
        overall = "HEALTHY"

    return PortfolioScorecard(
        strategies=scorecards,
        n_total_trades=n_total,
        n_strategies=len(scorecards),
        n_healthy=n_healthy,
        n_warning=n_warning,
        n_critical=n_critical,
        overall_status=overall,
    )


def load_trades_for_scorecard(
    backtest_csv: Optional[Path] = None,
    positions_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Load trades from backtest and/or live positions.

    Merges both sources into a unified DataFrame with columns:
    strategy_name, pnl_pct, exit_date, signal_id, exit_reason.

    Positions are filtered to CLOSED status only.
    """
    frames = []

    if backtest_csv is not None and backtest_csv.exists():
        bt = pd.read_csv(backtest_csv, dtype=str).fillna("")
        if not bt.empty:
            frames.append(bt[["strategy_name", "pnl_pct", "exit_date",
                              "signal_id", "exit_reason"]].copy())

    if positions_csv is not None and positions_csv.exists():
        pos = pd.read_csv(positions_csv, dtype=str).fillna("")
        if not pos.empty:
            closed = pos[pos.get("position_status", pos.get("status", "")) == "CLOSED"]
            if not closed.empty:
                cols = []
                for c in ["strategy_name", "pnl_pct", "exit_date", "signal_id", "exit_reason"]:
                    if c in closed.columns:
                        cols.append(c)
                    elif c == "pnl_pct" and "mtm_pnl_pct" in closed.columns:
                        closed = closed.copy()
                        closed["pnl_pct"] = closed["mtm_pnl_pct"]
                        cols.append("pnl_pct")
                if "strategy_name" in cols and "pnl_pct" in cols:
                    frames.append(closed[cols].copy())

    if not frames:
        return pd.DataFrame(columns=["strategy_name", "pnl_pct", "exit_date",
                                      "signal_id", "exit_reason"])

    merged = pd.concat(frames, ignore_index=True)
    if "exit_date" in merged.columns:
        merged = merged.sort_values("exit_date")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# Health status logic
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_health_status(
    sc: StrategyScorecard,
    baseline_config: Optional[Dict] = None,
    consecutive_loss_threshold: int = 5,
    wr_below_breakeven_checks: int = 4,
    sharpe_warning: float = 0.5,
    baseline_deviation_warning: float = 0.15,
) -> Tuple[str, List[str]]:
    """Determine HEALTHY / WARNING / CRITICAL status."""
    reasons: List[str] = []

    # Compute true breakeven from baseline if available
    breakeven = 0.5
    if baseline_config is not None:
        avg_win = baseline_config.get("kelly_avg_win", 0)
        avg_loss = baseline_config.get("kelly_avg_loss", 0)
        if avg_win + avg_loss > 0:
            breakeven = avg_loss / (avg_win + avg_loss)

    # ── CRITICAL checks ──────────────────────────────────────────────────
    critical_reasons: List[str] = []

    if sc.consecutive_loss_count >= consecutive_loss_threshold:
        critical_reasons.append(
            f"Loss streak: {sc.consecutive_loss_count} consecutive losses"
        )

    if (sc.rolling_win_rate_series
            and len(sc.rolling_win_rate_series) >= wr_below_breakeven_checks):
        recent = sc.rolling_win_rate_series[-wr_below_breakeven_checks:]
        if all(wr < breakeven for wr in recent):
            critical_reasons.append(
                f"Win rate below breakeven ({breakeven:.0%}) for "
                f"{wr_below_breakeven_checks} consecutive windows"
            )

    if critical_reasons:
        return "CRITICAL", critical_reasons

    # ── WARNING checks ───────────────────────────────────────────────────
    if sc.rolling_sharpe is not None and sc.rolling_sharpe < sharpe_warning:
        reasons.append(f"Rolling Sharpe {sc.rolling_sharpe:.2f} < {sharpe_warning}")

    if sc.win_rate_trend is not None and sc.win_rate_trend < 0:
        reasons.append(f"Win rate trending down (slope: {sc.win_rate_trend:.4f})")

    if (sc.live_vs_baseline_delta is not None
            and abs(sc.live_vs_baseline_delta) > baseline_deviation_warning):
        reasons.append(
            f"Win rate deviates {sc.live_vs_baseline_delta:+.0%} from baseline"
        )

    if reasons:
        return "WARNING", reasons

    return "HEALTHY", []


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_scorecard(
    strategy_name: str,
    baseline_config: Optional[Dict] = None,
) -> StrategyScorecard:
    """Return a scorecard for a strategy with no trades."""
    baseline_wr = None
    if baseline_config is not None:
        baseline_wr = baseline_config.get("kelly_win_rate")

    return StrategyScorecard(
        strategy_name=strategy_name,
        n_trades=0,
        rolling_win_rate=None,
        rolling_profit_factor=None,
        consecutive_loss_count=0,
        avg_pnl_per_trade=0.0,
        rolling_sharpe=None,
        rolling_sortino=None,
        win_rate_trend=None,
        baseline_win_rate=baseline_wr,
        live_vs_baseline_delta=None,
        anderson_darling_pvalue=None,
        win_rate_by_regime={},
        mean_pnl_by_regime={},
        best_regime=None,
        worst_regime=None,
        rolling_win_rate_series=[],
        status="HEALTHY",
        status_reasons=[],
    )


def _estimate_trades_per_year(trades: pd.DataFrame) -> float:
    """Estimate annualised trade frequency from date range."""
    if "exit_date" not in trades.columns or len(trades) < 2:
        return 48.0  # default: ~4 trades/month

    dates = pd.to_datetime(trades["exit_date"], errors="coerce").dropna()
    if len(dates) < 2:
        return 48.0

    span_days = (dates.max() - dates.min()).days
    if span_days <= 0:
        return 48.0

    return len(dates) / (span_days / 365.25)


def _anderson_darling_vs_baseline(
    trailing_pnls: np.ndarray,
    baseline_config: Dict,
    n_samples: int = 1000,
    seed: int = 42,
) -> Optional[float]:
    """Run 2-sample Anderson-Darling test: trailing trades vs baseline.

    Returns the significance level (p-value approximation) or None if
    the test cannot be run.
    """
    baseline = _build_baseline_pnl_distribution(baseline_config, n_samples, seed)
    if baseline is None or len(trailing_pnls) < 5:
        return None

    try:
        from scipy.stats import anderson_ksamp
        result = anderson_ksamp([trailing_pnls, baseline])
        return float(result.significance_level)
    except Exception as e:
        log.debug(f"Anderson-Darling test failed: {e}")
        return None


def _build_baseline_pnl_distribution(
    config: Dict,
    n_samples: int = 1000,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Synthesize a P&L distribution from Strategies.json baseline params.

    Uses kelly_win_rate, kelly_avg_win, kelly_avg_loss to create a
    realistic two-component distribution for comparison.
    """
    wr = config.get("kelly_win_rate")
    avg_win = config.get("kelly_avg_win")
    avg_loss = config.get("kelly_avg_loss")

    if wr is None or avg_win is None or avg_loss is None:
        return None

    rng = np.random.default_rng(seed)
    n_wins = int(n_samples * wr)
    n_losses = n_samples - n_wins

    # Triangular-ish distributions centred on the expected values
    wins = rng.uniform(0.0, 2 * avg_win, size=n_wins)
    losses = rng.uniform(-2 * avg_loss, 0.0, size=n_losses)

    return np.concatenate([wins, losses])


def _join_regime(
    trades: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach vol_regime to each trade (reuses pnl_attribution pattern)."""
    trades = trades.copy()

    if "vol_regime" not in features_df.columns:
        trades["vol_regime"] = "unknown"
        return trades

    # Try signal_id join first
    if "signal_id" in trades.columns and "signal_id" in features_df.columns:
        regime_map = features_df.set_index("signal_id")["vol_regime"].to_dict()
        trades["vol_regime"] = trades["signal_id"].map(regime_map).fillna("unknown")

        matched = (trades["vol_regime"] != "unknown").sum()
        if matched > len(trades) * 0.5:
            return trades

    # Fallback: date proximity
    if "entry_date" in trades.columns and "as_of_date" in features_df.columns:
        feat = features_df.copy()
        feat["_dt"] = pd.to_datetime(feat["as_of_date"], errors="coerce")
        feat = feat.dropna(subset=["_dt"])

        def _closest(entry_date):
            try:
                entry = pd.Timestamp(entry_date)
                deltas = (feat["_dt"] - entry).abs()
                return str(feat.loc[deltas.idxmin(), "vol_regime"])
            except Exception:
                return "unknown"

        trades["vol_regime"] = trades["entry_date"].apply(_closest)

    if "vol_regime" not in trades.columns:
        trades["vol_regime"] = "unknown"

    return trades
