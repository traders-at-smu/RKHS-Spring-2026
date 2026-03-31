"""
Live P&L attribution — track predicted vs actual regime performance.

Answers the key question: "Are we making money in the regime the model
says we're in?"

Attribution breakdowns:
  1. **By regime**: P&L attributed to the vol regime active at trade entry.
     If HMM says "high vol", did we actually reduce size and survive?
  2. **By strategy**: P&L per strategy, split by the regime at entry.
  3. **Regime accuracy**: Did the regime assignment match reality?
     (e.g., was RV actually elevated during "high" regime labels?)

This is a closed-loop validation: model predictions → actions → outcomes.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("nyx.layer6.pnl_attribution")


@dataclass
class RegimePerformance:
    """P&L summary for trades entered during a specific regime."""
    regime: str
    n_trades: int
    total_pnl_pct: float
    mean_pnl_pct: float
    win_rate: float
    avg_risk_used: float           # average risk fraction per trade
    worst_trade: float
    best_trade: float


@dataclass
class StrategyRegimeBreakdown:
    """P&L by strategy × regime combination."""
    strategy: str
    regime: str
    n_trades: int
    total_pnl_pct: float
    mean_pnl_pct: float
    win_rate: float


@dataclass
class AttributionReport:
    """Complete P&L attribution report."""
    by_regime: Dict[str, RegimePerformance]
    by_strategy_regime: List[StrategyRegimeBreakdown]
    regime_accuracy: Dict[str, float]  # regime → RV calibration score
    total_trades: int
    total_pnl_pct: float
    regime_method_distribution: Dict[str, int]  # count of "hmm" vs "tercile"

    def summary_df(self) -> pd.DataFrame:
        """Regime-level summary as DataFrame."""
        rows = []
        for regime, perf in self.by_regime.items():
            rows.append({
                "Regime": regime.title(),
                "Trades": perf.n_trades,
                "Total P&L": f"{perf.total_pnl_pct:+.1%}",
                "Mean P&L": f"{perf.mean_pnl_pct:+.1%}",
                "Win Rate": f"{perf.win_rate:.0%}",
                "Avg Risk": f"{perf.avg_risk_used:.2%}",
                "Worst": f"{perf.worst_trade:+.1%}",
                "Best": f"{perf.best_trade:+.1%}",
            })
        return pd.DataFrame(rows)

    def strategy_regime_df(self) -> pd.DataFrame:
        """Strategy × regime breakdown as DataFrame."""
        rows = []
        for sr in self.by_strategy_regime:
            rows.append({
                "Strategy": sr.strategy,
                "Regime": sr.regime.title(),
                "Trades": sr.n_trades,
                "Total P&L": f"{sr.total_pnl_pct:+.1%}",
                "Mean P&L": f"{sr.mean_pnl_pct:+.1%}",
                "Win Rate": f"{sr.win_rate:.0%}",
            })
        return pd.DataFrame(rows)


def compute_pnl_attribution(
    trades_df: pd.DataFrame,
    features_df: pd.DataFrame,
    executions_df: Optional[pd.DataFrame] = None,
) -> AttributionReport:
    """Compute P&L attribution by regime and strategy.

    Joins trades to features to get the regime active at each trade's
    entry, then computes performance breakdowns.

    Args:
        trades_df: Backtest or live trades with columns:
            signal_id, strategy_name, pnl_pct, entry_date, exit_date.
        features_df: Feature rows with columns:
            signal_id (or as_of_date), vol_regime, regime_method.
        executions_df: Optional execution data with risk_used_fraction.

    Returns:
        AttributionReport with full breakdown.
    """
    if trades_df.empty:
        return _empty_report()

    # Ensure numeric P&L
    trades = trades_df.copy()
    trades["pnl_pct"] = pd.to_numeric(trades["pnl_pct"], errors="coerce").fillna(0.0)

    # Join regime to each trade
    trades = _join_regime(trades, features_df)

    # Join risk used from executions (if available)
    if executions_df is not None and not executions_df.empty:
        trades = _join_risk_used(trades, executions_df)
    else:
        trades["risk_used_fraction"] = 0.01  # default

    # ── By regime ──
    by_regime = {}
    for regime in ["low", "normal", "high", "unknown"]:
        mask = trades["vol_regime"] == regime
        subset = trades[mask]
        if subset.empty:
            continue

        pnls = subset["pnl_pct"].values
        risk_fracs = pd.to_numeric(subset["risk_used_fraction"], errors="coerce").fillna(0.01).values

        by_regime[regime] = RegimePerformance(
            regime=regime,
            n_trades=len(subset),
            total_pnl_pct=float(pnls.sum()),
            mean_pnl_pct=float(pnls.mean()),
            win_rate=float((pnls > 0).mean()),
            avg_risk_used=float(risk_fracs.mean()),
            worst_trade=float(pnls.min()),
            best_trade=float(pnls.max()),
        )

    # ── By strategy × regime ──
    by_strat_regime = []
    if "strategy_name" in trades.columns:
        for (strat, regime), group in trades.groupby(["strategy_name", "vol_regime"]):
            pnls = group["pnl_pct"].values
            by_strat_regime.append(StrategyRegimeBreakdown(
                strategy=str(strat),
                regime=str(regime),
                n_trades=len(group),
                total_pnl_pct=float(pnls.sum()),
                mean_pnl_pct=float(pnls.mean()),
                win_rate=float((pnls > 0).mean()),
            ))

    # Sort by strategy then regime
    by_strat_regime.sort(key=lambda x: (x.strategy, x.regime))

    # ── Regime accuracy ──
    regime_accuracy = _compute_regime_accuracy(trades, features_df)

    # ── Regime method distribution ──
    method_dist = {}
    if "regime_method" in features_df.columns:
        for method, count in features_df["regime_method"].value_counts().items():
            method_dist[str(method)] = int(count)

    total_pnl = float(trades["pnl_pct"].sum())

    return AttributionReport(
        by_regime=by_regime,
        by_strategy_regime=by_strat_regime,
        regime_accuracy=regime_accuracy,
        total_trades=len(trades),
        total_pnl_pct=total_pnl,
        regime_method_distribution=method_dist,
    )


def _join_regime(
    trades: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach vol_regime to each trade, matching by signal_id or date."""
    trades = trades.copy()

    if "vol_regime" not in features_df.columns:
        trades["vol_regime"] = "unknown"
        return trades

    # Try joining on signal_id first
    if "signal_id" in trades.columns and "signal_id" in features_df.columns:
        regime_map = features_df.set_index("signal_id")["vol_regime"].to_dict()
        trades["vol_regime"] = trades["signal_id"].map(regime_map).fillna("unknown")

        # If most got matched, we're done
        matched = (trades["vol_regime"] != "unknown").sum()
        if matched > len(trades) * 0.5:
            return trades

    # Fallback: match by date proximity
    if "entry_date" in trades.columns and "as_of_date" in features_df.columns:
        trades["vol_regime"] = trades["entry_date"].apply(
            lambda d: _closest_regime(d, features_df)
        )

    if "vol_regime" not in trades.columns:
        trades["vol_regime"] = "unknown"

    return trades


def _closest_regime(entry_date: str, features_df: pd.DataFrame) -> str:
    """Find the vol_regime closest to the entry date."""
    try:
        entry = pd.Timestamp(entry_date)
        feat = features_df.copy()
        feat["_dt"] = pd.to_datetime(feat["as_of_date"], errors="coerce")
        feat = feat.dropna(subset=["_dt"])
        feat["_delta"] = (feat["_dt"] - entry).abs()
        closest = feat.loc[feat["_delta"].idxmin()]
        return str(closest.get("vol_regime", "unknown"))
    except Exception:
        return "unknown"


def _join_risk_used(
    trades: pd.DataFrame,
    executions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join risk_used_fraction from executions to trades."""
    if "risk_used_fraction" in executions_df.columns:
        # Create a mapping — use index alignment or simple merge
        if "signal_id" in trades.columns and "signal_id" in executions_df.columns:
            risk_map = executions_df.groupby("signal_id")["risk_used_fraction"].first().to_dict()
            trades["risk_used_fraction"] = trades["signal_id"].map(risk_map).fillna(0.01)
        else:
            # Positional alignment (demo mode)
            risk_vals = pd.to_numeric(
                executions_df["risk_used_fraction"], errors="coerce"
            ).fillna(0.01).values
            n = min(len(trades), len(risk_vals))
            trades.loc[trades.index[:n], "risk_used_fraction"] = risk_vals[:n]

    if "risk_used_fraction" not in trades.columns:
        trades["risk_used_fraction"] = 0.01

    return trades


def _compute_regime_accuracy(
    trades: pd.DataFrame,
    features_df: pd.DataFrame,
) -> Dict[str, float]:
    """Score regime calibration: was RV actually elevated during 'high' labels?

    For each regime label, compute the mean RV. If "high" has the highest
    mean RV and "low" has the lowest, the labelling is well-calibrated.

    Returns a score per regime: closer to 1.0 = well-calibrated.
    """
    accuracy = {}

    if "rv20" not in features_df.columns or "vol_regime" not in features_df.columns:
        return accuracy

    feat = features_df.copy()
    feat["rv20"] = pd.to_numeric(feat["rv20"], errors="coerce")
    feat = feat.dropna(subset=["rv20"])

    if feat.empty:
        return accuracy

    global_mean = float(feat["rv20"].mean())
    global_std = float(feat["rv20"].std()) if feat["rv20"].std() > 0 else 1.0

    for regime in ["low", "normal", "high"]:
        mask = feat["vol_regime"] == regime
        if not mask.any():
            continue

        regime_mean = float(feat.loc[mask, "rv20"].mean())

        # Score: how well does the mean match expectations?
        # low should be below global mean, high should be above
        if regime == "low":
            score = 1.0 if regime_mean < global_mean else max(0.0, 1.0 - (regime_mean - global_mean) / global_std)
        elif regime == "high":
            score = 1.0 if regime_mean > global_mean else max(0.0, 1.0 - (global_mean - regime_mean) / global_std)
        else:  # normal
            distance = abs(regime_mean - global_mean) / global_std
            score = max(0.0, 1.0 - distance)

        accuracy[regime] = round(score, 4)

    return accuracy


def _empty_report() -> AttributionReport:
    """Return an empty attribution report."""
    return AttributionReport(
        by_regime={},
        by_strategy_regime=[],
        regime_accuracy={},
        total_trades=0,
        total_pnl_pct=0.0,
        regime_method_distribution={},
    )
