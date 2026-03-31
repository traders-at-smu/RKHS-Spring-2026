"""
Portfolio allocation: risk parity, volatility targeting, fractional Kelly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def equal_weight(n_strategies: int) -> np.ndarray:
    """Simple 1/N allocation."""
    return np.ones(n_strategies) / n_strategies


def inverse_vol_weights(returns_df: pd.DataFrame) -> np.ndarray:
    """Risk-parity-lite: allocate inversely proportional to volatility."""
    vols = returns_df.std().values
    vols = np.where(vols > 0, vols, 1e-9)
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


def vol_target_scalar(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    target_annual_vol: float = 0.15,
    annualisation_factor: float = np.sqrt(52),
) -> float:
    """Scalar to apply to portfolio to hit target annualised volatility.

    Returns a multiplier: if current portfolio vol is 0.10 and target is
    0.15, returns 1.5.
    """
    cov = returns_df.cov().values
    port_var = float(weights @ cov @ weights)
    port_vol_annual = np.sqrt(port_var) * annualisation_factor
    if port_vol_annual <= 0:
        return 1.0
    return target_annual_vol / port_vol_annual


def fractional_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly criterion for position sizing.

    Full Kelly is often too aggressive for options; fraction=0.25 (quarter
    Kelly) is a common conservative choice.

    Returns the fraction of capital to risk per trade.
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / abs(avg_loss)
    kelly_f = (b * win_rate - (1 - win_rate)) / b
    return max(0.0, kelly_f * fraction)


def regime_gate(
    vol_regime: str,
    base_allocation: float,
    low_vol_scale: float = 1.0,
    normal_vol_scale: float = 0.75,
    high_vol_scale: float = 0.50,
) -> float:
    """Scale allocation based on current vol regime.

    Reduces size in high-vol regimes when strategies cluster risk.
    """
    scales = {
        "low": low_vol_scale,
        "normal": normal_vol_scale,
        "high": high_vol_scale,
    }
    return base_allocation * scales.get(vol_regime, normal_vol_scale)
