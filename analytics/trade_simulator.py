"""
Trade simulator: replay signals through exit rules to produce per-trade P&L.

Applies deterministic exit rules from Strategies.json:
  - take_profit_pct
  - stop_loss_pct
  - time_stop_days
  - expiry handling

Supports two price modes:
  - "option" (default): daily_closes are option premiums.
    P&L = direction × (exit_premium - entry_premium) × multiplier × quantity.
  - "stock_proxy": daily_closes are underlying stock prices.  Converts stock
    move into an approximate option-equivalent return using delta:
      option_return ≈ delta × stock_return.
    This is less accurate but allows backtesting when historical option
    chain data is unavailable.  The approximation degrades for large moves
    and long holding periods (gamma + theta decay are not modelled).

Input: historical signals with entry prices and subsequent daily closes.
Output: list of completed TradeResult objects.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

log = logging.getLogger("nyx.layer5.simulator")


@dataclass
class TradeResult:
    trade_id: str
    signal_id: str
    strategy_name: str
    symbol: str
    entry_date: str
    exit_date: str
    holding_days: int
    entry_price: float
    exit_price: float
    pnl_dollars: float
    pnl_pct: float           # return on risk (pnl / entry_price)
    risk_dollars: float
    exit_reason: str          # "tp" | "sl" | "time_stop" | "expiry" | "end_of_data"
    pipeline_version: str
    price_type: str = "option"   # "option" or "stock_proxy" — audit trail

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def simulate_trade(
    signal_id: str,
    strategy_name: str,
    symbol: str,
    entry_date: str,
    entry_price: float,
    daily_closes: pd.Series,
    take_profit_pct: float = 0.50,
    stop_loss_pct: float = 0.50,
    time_stop_days: int = 14,
    pipeline_version: str = "l5_backtest_v1",
    action: str = "BUY",
    multiplier: int = 100,
    quantity: int = 1,
    price_type: str = "option",
    delta: float = 0.50,
) -> TradeResult:
    """Simulate a single trade through exit rules.

    Args:
        daily_closes: Series of daily close prices AFTER entry, indexed by date.
            If *price_type* is ``"option"``, these are option premiums.
            If *price_type* is ``"stock_proxy"``, these are underlying stock prices
            and P&L is approximated via delta × stock_return.
        take_profit_pct: exit when unrealised gain >= this fraction of entry.
        stop_loss_pct: exit when unrealised loss >= this fraction of entry.
        time_stop_days: exit after this many calendar days.
        action: "BUY" for long, "SELL" for short.
        price_type: ``"option"`` (premium changes) or ``"stock_proxy"``
            (delta-approximated from stock prices).
        delta: Option delta for stock_proxy mode (0.0-1.0).
            Only used when *price_type* is ``"stock_proxy"``.
            Typical ATM delta ≈ 0.50; deep ITM ≈ 0.80; deep OTM ≈ 0.15.
    """
    if entry_price <= 0 or daily_closes.empty:
        return TradeResult(
            trade_id=f"bt_{signal_id}",
            signal_id=signal_id,
            strategy_name=strategy_name,
            symbol=symbol,
            entry_date=entry_date,
            exit_date=entry_date,
            holding_days=0,
            entry_price=entry_price,
            exit_price=entry_price,
            pnl_dollars=0.0,
            pnl_pct=0.0,
            risk_dollars=entry_price * multiplier * quantity,
            exit_reason="no_data",
            pipeline_version=pipeline_version,
            price_type=price_type,
        )

    direction = 1.0 if action.upper() == "BUY" else -1.0
    risk_dollars = entry_price * multiplier * quantity

    # For stock_proxy mode: remember the stock price at entry
    stock_entry = float(daily_closes.iloc[0]) if price_type == "stock_proxy" else None

    for i, (dt, close) in enumerate(daily_closes.items()):
        days_held = i + 1

        if price_type == "stock_proxy" and stock_entry is not None and stock_entry > 0:
            # Delta approximation: option_return ≈ delta × stock_return
            stock_return = (close - stock_entry) / stock_entry
            unrealised_pct = direction * delta * stock_return
        else:
            # Option premium mode — direct comparison
            unrealised_pct = direction * (close - entry_price) / entry_price

        exit_reason = None

        # Take profit
        if unrealised_pct >= take_profit_pct:
            exit_reason = "tp"
        # Stop loss
        elif unrealised_pct <= -stop_loss_pct:
            exit_reason = "sl"
        # Time stop
        elif days_held >= time_stop_days:
            exit_reason = "time_stop"

        if exit_reason:
            pnl_pct = unrealised_pct
            if price_type == "stock_proxy" and stock_entry is not None:
                # P&L in dollars = delta × stock_move × multiplier × quantity
                pnl_dollars = direction * delta * (close - stock_entry) * multiplier * quantity
            else:
                pnl_dollars = direction * (close - entry_price) * multiplier * quantity
            exit_date = str(dt.date()) if hasattr(dt, "date") else str(dt)
            return TradeResult(
                trade_id=f"bt_{signal_id}",
                signal_id=signal_id,
                strategy_name=strategy_name,
                symbol=symbol,
                entry_date=entry_date,
                exit_date=exit_date,
                holding_days=days_held,
                entry_price=entry_price,
                exit_price=float(close),
                pnl_dollars=round(pnl_dollars, 2),
                pnl_pct=round(pnl_pct, 6),
                risk_dollars=risk_dollars,
                exit_reason=exit_reason,
                pipeline_version=pipeline_version,
                price_type=price_type,
            )

    # End of data — close position
    last_close = float(daily_closes.iloc[-1])
    last_date = daily_closes.index[-1]
    exit_date = str(last_date.date()) if hasattr(last_date, "date") else str(last_date)

    if price_type == "stock_proxy" and stock_entry is not None and stock_entry > 0:
        stock_return = (last_close - stock_entry) / stock_entry
        pnl_pct = direction * delta * stock_return
        pnl_dollars = direction * delta * (last_close - stock_entry) * multiplier * quantity
    else:
        pnl_pct = direction * (last_close - entry_price) / entry_price
        pnl_dollars = direction * (last_close - entry_price) * multiplier * quantity

    return TradeResult(
        trade_id=f"bt_{signal_id}",
        signal_id=signal_id,
        strategy_name=strategy_name,
        symbol=symbol,
        entry_date=entry_date,
        exit_date=exit_date,
        holding_days=len(daily_closes),
        entry_price=entry_price,
        exit_price=last_close,
        pnl_dollars=round(pnl_dollars, 2),
        pnl_pct=round(pnl_pct, 6),
        risk_dollars=risk_dollars,
        exit_reason="end_of_data",
        pipeline_version=pipeline_version,
        price_type=price_type,
    )
