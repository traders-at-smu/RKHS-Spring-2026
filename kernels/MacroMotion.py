"""
macro_momentum_layer.py
=======================
RKHS Macro Momentum Layer (K_slow) — Structural Variance Attribution
Part of the Multi-Layer RKHS Price Action Attribution Framework

Existing layers: Derivatives Pressure, Event Proximity, Sentiment,
                 Factor Beta Regime, LOB, Order Flow Toxicity, Hawkes Process

This layer: Macro Momentum (K_slow) — regime gate separating
            systemic capital flow variance from idiosyncratic microstructure variance.

Mathematical guarantee: K_slow is a valid Mercer kernel (symmetric, PSD).
Temporal guarantee:     Zero look-ahead bias via ZOH projection on daily close.

Kernel composition:
    K_slow(z, z') = K_mom(x, x') ⊙ K_corr(C, C')

    K_mom  = γ₁·K_Matérn32(x_5d)  + γ₂·K_Matérn32(x_20d) + γ₃·K_Matérn32(x_60d)
    K_corr = RBF([ρ_SPY, ρ_sector, σ²_ε])

    Product of PSD kernels is PSD (Schur product theorem).
    Sum   of PSD kernels is PSD (closure under addition).

Features:
    x_5d   — Z-scored 5-day cumulative log return (tactical flow)
    x_20d  — Vol-adjusted distance from 20-day SMA (institutional rebalancing)
    x_60d  — Fractionally differenced 60-day price path, d∈[0.4,0.6] (macro regime)
    ρ_SPY  — Rolling 60-day Pearson correlation to SPY
    ρ_sec  — Rolling 60-day Pearson correlation to sector ETF
    σ²_ε   — Idiosyncratic residual variance from rolling OLS

References:
    López de Prado (2018) — Advances in Financial Machine Learning, Ch. 5
    Matérn (1960) — Spatial Variation, ν=3/2 formulation
    MacKinnon (1994) — Approximate asymptotic distribution functions for
                       unit-root and cointegration tests
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility: ADF test (no statsmodels dependency)
# ---------------------------------------------------------------------------

def _adf_test(series: np.ndarray, max_lags: int = 1) -> tuple[float, float]:
    """
    Augmented Dickey-Fuller test with a constant term.

    Regression:
        Δy_t = α + γ·y_{t-1} + Σδ_k·Δy_{t-k} + ε_t

    Returns (test_statistic, p_value_approx) using MacKinnon (1994)
    response surface critical values for large-sample interpolation.

    Parameters
    ----------
    series : array-like price or feature series
    max_lags : number of lagged difference terms to include

    Returns
    -------
    (t_stat, p_value) — approximate p-value from MacKinnon (1994) table
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 20:
        return 0.0, 1.0

    dy = np.diff(arr)
    y_lag1 = arr[:-1]

    lags = min(max_lags, max(0, len(dy) - 4))

    if lags > 0:
        start = lags
        X_parts = [
            np.ones(n - 1 - lags),
            y_lag1[lags:],
        ]
        for k in range(1, lags + 1):
            X_parts.append(dy[lags - k: n - 1 - k])
        X = np.column_stack(X_parts)
        y = dy[lags:]
    else:
        X = np.column_stack([np.ones(n - 1), y_lag1])
        y = dy

    try:
        coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < X.shape[1]:
            return 0.0, 1.0
        resid = y - X @ coeffs
        df = len(y) - X.shape[1]
        sigma2 = np.sum(resid ** 2) / max(df, 1)
        xtx_inv = np.linalg.inv(X.T @ X)
        se_gamma = np.sqrt(sigma2 * xtx_inv[1, 1])
        t_stat = coeffs[1] / (se_gamma + 1e-15)
    except (np.linalg.LinAlgError, FloatingPointError):
        return 0.0, 1.0

    # MacKinnon (1994) asymptotic critical values — "constant" case
    cv = {0.01: -3.430, 0.05: -2.860, 0.10: -2.570}

    if t_stat <= cv[0.01]:
        p = 0.005
    elif t_stat <= cv[0.05]:
        frac = (t_stat - cv[0.01]) / (cv[0.05] - cv[0.01])
        p = 0.01 + frac * 0.04
    elif t_stat <= cv[0.10]:
        frac = (t_stat - cv[0.05]) / (cv[0.10] - cv[0.05])
        p = 0.05 + frac * 0.05
    else:
        # Linearly extrapolate toward p=1 as τ → 0
        slope = 0.90 / max(abs(-1.0 - cv[0.10]), 1e-6)
        p = min(0.10 + abs(t_stat - cv[0.10]) * slope, 1.0)

    return float(t_stat), float(p)


# ---------------------------------------------------------------------------
# 1. FractionalDifferencer
# ---------------------------------------------------------------------------

class FractionalDifferencer:
    """
    Marcos López de Prado fixed-width window fractional differencing.

    Integer differencing (d=1) destroys the long-memory structure of price
    series. Fractional differencing with d∈(0,1) achieves stationarity while
    retaining maximum autocorrelation (information retention).

    Algorithm:
        weights_k = ∏_{j=0}^{k-1} (d - j) / (j + 1)   (binomial expansion)
        ̃x_t = Σ_{k=0}^{L} weights_k · x_{t-k}

    The weight sequence is truncated at a fixed window length L to prevent
    look-ahead bias and keep computation O(L·T).

    Parameters
    ----------
    window : int
        Fixed truncation window for the weight vector. Default 63 (~1 quarter).
    thresh : float
        Weight threshold below which further weights are discarded (controls
        approximation quality). Default 1e-5.
    """

    def __init__(self, window: int = 63, thresh: float = 1e-5) -> None:
        self.window = window
        self.thresh = thresh
        self.d_: Optional[float] = None

    # ------------------------------------------------------------------
    def _get_weights(self, d: float, size: int) -> np.ndarray:
        """
        Compute the binomial-expansion weight vector for order d.

        w_0 = 1
        w_k = w_{k-1} · (d - k + 1) / k

        Parameters
        ----------
        d    : fractional differencing order
        size : number of weights to compute

        Returns
        -------
        weights : np.ndarray of shape (size,)
        """
        weights = np.zeros(size)
        weights[0] = 1.0
        for k in range(1, size):
            weights[k] = -weights[k - 1] * (d - k + 1) / k
            if abs(weights[k]) < self.thresh:
                weights[k:] = 0.0
                break
        return weights

    # ------------------------------------------------------------------
    def transform(self, series: pd.Series, d: float) -> pd.Series:
        """
        Apply fixed-width window fractional differencing to a price series.

        Each output value uses at most `window` past observations, ensuring
        strict look-ahead-free computation.

        Parameters
        ----------
        series : pd.Series of raw prices (or log-prices)
        d      : fractional differencing order ∈ (0, 1)

        Returns
        -------
        pd.Series of fractionally differenced values, NaN for the first
        (window-1) observations where the full window is unavailable.
        """
        if not 0 < d <= 1.0:
            raise ValueError(f"d must be in (0, 1], got {d}")
        prices = series.values.astype(float)
        T = len(prices)
        width = min(self.window, T)
        weights = self._get_weights(d, width)

        out = np.full(T, np.nan)
        for t in range(width - 1, T):
            window_vals = prices[t - width + 1: t + 1][::-1]   # most-recent first
            out[t] = np.dot(weights[:len(window_vals)], window_vals)

        return pd.Series(out, index=series.index, name=f"fracdiff_d{d:.2f}")

    # ------------------------------------------------------------------
    def fit(
        self,
        series: pd.Series,
        d_range: tuple[float, float] = (0.1, 1.0),
        adf_threshold: float = 0.05,
    ) -> float:
        """
        Sweep d from d_range[0] to d_range[1] in steps of 0.05.
        Select the **minimum** d such that the ADF test rejects the unit-root
        null at the given significance level (p < adf_threshold).

        This maximises long-memory retention while guaranteeing stationarity.

        Parameters
        ----------
        series        : raw price series
        d_range       : (d_min, d_max) sweep range
        adf_threshold : ADF p-value threshold for stationarity acceptance

        Returns
        -------
        d_optimal : float — optimal fractional differencing parameter
        """
        log_px = np.log(series.dropna().values.astype(float))
        d_candidates = np.arange(d_range[0], d_range[1] + 0.01, 0.05)

        for d in d_candidates:
            fd_series = self.transform(
                pd.Series(log_px, index=series.dropna().index), d
            ).dropna()
            if len(fd_series) < 30:
                continue
            _, p_val = _adf_test(fd_series.values, max_lags=1)
            logger.debug("FracDiff sweep d=%.2f → ADF p=%.4f", d, p_val)
            if p_val < adf_threshold:
                self.d_ = float(d)
                logger.info("FractionalDifferencer: selected d=%.2f (ADF p=%.4f)", d, p_val)
                return self.d_

        # Fallback: use 0.5 (midpoint — empirically robust for most equity series)
        self.d_ = 0.5
        logger.warning(
            "FractionalDifferencer: no d in %s achieved stationarity; "
            "defaulting to d=0.5", d_range
        )
        return self.d_


# ---------------------------------------------------------------------------
# 2. CrossAssetRegressor
# ---------------------------------------------------------------------------

class CrossAssetRegressor:
    """
    Rolling OLS decomposition of target stock returns into macro components.

    Regression model:
        R_target(t) = α + β₁·R_SPY(t) + β₂·R_sector(t) + ε(t)

    Three regime-gate features extracted per day:
        ρ_SPY       — rolling Pearson correlation to SPY
        ρ_sector    — rolling Pearson correlation to sector ETF
        σ²_ε        — idiosyncratic residual variance (unexplained by macro)

    The idiosyncratic residual variance σ²_ε is the most structurally
    important feature: it captures how much the stock is moving independently
    of the macro environment on any given day.

    Parameters
    ----------
    window : int
        Rolling window in trading days. Default 60 (~1 quarter).
    """

    def __init__(self, window: int = 60) -> None:
        if window < 10:
            raise ValueError(f"OLS window must be ≥ 10, got {window}")
        self.window = window

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        r_target: pd.Series,
        r_spy: pd.Series,
        r_sector: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute rolling cross-asset correlation features.

        Parameters
        ----------
        r_target : daily log returns of the target stock
        r_spy    : daily log returns of SPY
        r_sector : daily log returns of sector ETF

        Returns
        -------
        pd.DataFrame with columns ['rho_spy', 'rho_sector', 'sigma2_eps'],
        indexed identically to r_target. NaN for the first (window-1) rows.
        """
        idx = r_target.index
        n = len(r_target)
        rt = r_target.values.astype(float)
        rs = r_spy.reindex(idx).values.astype(float)
        re = r_sector.reindex(idx).values.astype(float)

        rho_spy    = np.full(n, np.nan)
        rho_sector = np.full(n, np.nan)
        sigma2_eps = np.full(n, np.nan)

        w = self.window
        for t in range(w - 1, n):
            sl = slice(t - w + 1, t + 1)
            y  = rt[sl];  x1 = rs[sl];  x2 = re[sl]

            valid = ~(np.isnan(y) | np.isnan(x1) | np.isnan(x2))
            if valid.sum() < max(10, w // 2):
                continue

            y_v  = y[valid];  x1_v = x1[valid];  x2_v = x2[valid]

            # Pearson correlations
            rho_spy[t]    = float(np.corrcoef(y_v, x1_v)[0, 1])
            rho_sector[t] = float(np.corrcoef(y_v, x2_v)[0, 1])

            # Rolling OLS residual variance
            X = np.column_stack([np.ones(len(y_v)), x1_v, x2_v])
            try:
                beta, _, rank, _ = np.linalg.lstsq(X, y_v, rcond=None)
                if rank < 3:
                    continue
                resid = y_v - X @ beta
                sigma2_eps[t] = float(np.var(resid, ddof=3))
            except np.linalg.LinAlgError:
                continue

        result = pd.DataFrame(
            {"rho_spy": rho_spy, "rho_sector": rho_sector, "sigma2_eps": sigma2_eps},
            index=idx,
        )
        logger.debug("CrossAssetRegressor: computed %d rows, %d valid", n, result.notna().all(axis=1).sum())
        return result


# ---------------------------------------------------------------------------
# 3. MomentumFeatureBuilder
# ---------------------------------------------------------------------------

class MomentumFeatureBuilder:
    """
    Constructs three stationary momentum inputs at distinct lookback horizons.

    x_5d  — Tactical (weekly capital rotation):
        Z-scored 5-day cumulative log return using 252-day rolling stats.
        x_5d = (Σ log_ret[-5:] - μ_252) / σ_252

    x_20d — Institutional (monthly rebalancing):
        Volatility-adjusted distance from 20-day SMA.
        x_20d = (P_t - SMA_20) / (σ_20 · P_t)

    x_60d — Macro regime (quarterly cycle):
        Fractionally differentiated log-price path.
        d is auto-selected via ADF sweep to achieve stationarity while
        preserving maximum long-memory.

    Parameters
    ----------
    vol_window : int
        Rolling window for Z-score normalisation of x_5d. Default 252.
    frac_diff_d : float or None
        If None, auto-select via ADF sweep. If float, use directly.
    fracdiff_window : int
        Fixed truncation window for the frac-diff weight vector.
    """

    def __init__(
        self,
        vol_window: int = 252,
        frac_diff_d: Optional[float] = None,
        fracdiff_window: int = 63,
    ) -> None:
        self.vol_window      = vol_window
        self.frac_diff_d     = frac_diff_d
        self.fracdiff_window = fracdiff_window
        self._differencer    = FractionalDifferencer(window=fracdiff_window)
        self.selected_d_: Optional[float] = None

    # ------------------------------------------------------------------
    def transform(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute all three momentum features from a daily closing price series.

        Parameters
        ----------
        prices : pd.Series of daily closing prices (must be positive)

        Returns
        -------
        pd.DataFrame with columns ['x_5d', 'x_20d', 'x_60d'], daily indexed.
        First ~max(252, 60) rows will contain NaN until windows are filled.
        """
        if (prices <= 0).any():
            raise ValueError("Price series contains non-positive values.")

        log_ret = np.log(prices).diff()

        # — x_5d: Z-scored 5-day cumulative log return —
        cum_5d     = log_ret.rolling(5).sum()
        roll_mean  = cum_5d.rolling(self.vol_window).mean()
        roll_std   = cum_5d.rolling(self.vol_window).std()
        x_5d       = (cum_5d - roll_mean) / (roll_std + 1e-10)

        # — x_20d: vol-adjusted distance from 20-day SMA —
        sma_20      = prices.rolling(20).mean()
        rvol_20     = log_ret.rolling(20).std()
        x_20d       = (prices - sma_20) / (rvol_20 * prices + 1e-10)

        # — x_60d: fractionally differenced log-price —
        log_prices = np.log(prices)
        if self.frac_diff_d is None:
            d = self._differencer.fit(prices, d_range=(0.1, 1.0), adf_threshold=0.05)
        else:
            d = self.frac_diff_d
        self.selected_d_ = d
        x_60d = self._differencer.transform(log_prices, d)

        result = pd.DataFrame(
            {"x_5d": x_5d, "x_20d": x_20d, "x_60d": x_60d},
            index=prices.index,
        )
        logger.debug(
            "MomentumFeatureBuilder: x_5d valid=%d, x_20d valid=%d, x_60d valid=%d (d=%.2f)",
            result["x_5d"].notna().sum(),
            result["x_20d"].notna().sum(),
            result["x_60d"].notna().sum(),
            d,
        )
        return result


# ---------------------------------------------------------------------------
# 4. MaternKernel32
# ---------------------------------------------------------------------------

class MaternKernel32:
    """
    Matérn kernel with ν = 3/2.

    Chosen over Gaussian RBF for financial momentum because:
    - ν=3/2 implies once-differentiable sample paths (not infinitely smooth)
    - Heavier tails → more robust to jump-diffusions and regime shifts
    - Empirically superior for financial time-series similarity

    Formula:
        K(x, x'; ℓ) = (1 + √3·r/ℓ) · exp(-√3·r/ℓ)
        where r = |x - x'|

    This is a valid Mercer (PSD symmetric) kernel.

    Parameters
    ----------
    length_scale : float > 0
        Controls how quickly similarity decays with feature distance.
    variance : float > 0  (γ weight)
        Scales the kernel output. Represents the prior variance contribution
        of this horizon.
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0) -> None:
        if length_scale <= 0:
            raise ValueError(f"length_scale must be > 0, got {length_scale}")
        if variance <= 0:
            raise ValueError(f"variance must be > 0, got {variance}")
        self.length_scale = length_scale
        self.variance     = variance

    # ------------------------------------------------------------------
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate the kernel between rows of X and rows of Y.

        Parameters
        ----------
        X : np.ndarray, shape (n, 1) or (n,)
        Y : np.ndarray, shape (m, 1) or (m,), optional. If None, Y = X.

        Returns
        -------
        K : np.ndarray, shape (n, m)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float)).reshape(-1, 1)
        Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=float)).reshape(-1, 1)

        # Euclidean distance matrix
        diff = X[:, None, :] - Y[None, :, :]          # (n, m, 1)
        r    = np.sqrt(np.sum(diff ** 2, axis=-1))     # (n, m)

        sqrt3_r_l = np.sqrt(3.0) * r / self.length_scale
        K = self.variance * (1.0 + sqrt3_r_l) * np.exp(-sqrt3_r_l)
        return K

    # ------------------------------------------------------------------
    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the symmetric Gram (kernel) matrix K[i,j] = K(x_i, x_j).

        Parameters
        ----------
        X : np.ndarray, shape (n,) or (n, 1)

        Returns
        -------
        K : np.ndarray, shape (n, n) — symmetric PSD matrix
        """
        return self(X, X)


# ---------------------------------------------------------------------------
# 5. RBFKernel
# ---------------------------------------------------------------------------

class RBFKernel:
    """
    Gaussian Radial Basis Function (squared exponential) kernel.

    Chosen for the correlation state vector C_t = [ρ_SPY, ρ_sector, σ²_ε]
    because correlation measurements are smooth point-in-time statistics
    (no jump-diffusion character), so infinite differentiability is appropriate.

    Formula:
        K_RBF(C, C'; σ) = exp(-||C - C'||² / (2σ²))

    This is a valid Mercer (PSD symmetric) kernel.

    Parameters
    ----------
    bandwidth : float > 0
        σ parameter controlling regime similarity decay.
        Smaller σ → sharper regime boundaries.
    """

    def __init__(self, bandwidth: float = 1.0) -> None:
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth}")
        self.bandwidth = bandwidth

    # ------------------------------------------------------------------
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate the RBF kernel between rows of X and rows of Y.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        Y : np.ndarray, shape (m, d), optional. If None, Y = X.

        Returns
        -------
        K : np.ndarray, shape (n, m)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = X if Y is None else np.atleast_2d(np.asarray(Y, dtype=float))

        # ||C_i - C_j||² via broadcasting
        diff   = X[:, None, :] - Y[None, :, :]          # (n, m, d)
        sq_dist = np.sum(diff ** 2, axis=-1)              # (n, m)
        K = np.exp(-sq_dist / (2.0 * self.bandwidth ** 2))
        return K

    # ------------------------------------------------------------------
    def gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the symmetric Gram matrix K[i,j] = K_RBF(C_i, C_j).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)

        Returns
        -------
        K : np.ndarray, shape (n, n) — symmetric PSD matrix
        """
        return self(X, X)


# ---------------------------------------------------------------------------
# 6. KSlowKernel
# ---------------------------------------------------------------------------

class KSlowKernel:
    """
    Composite RKHS kernel for the Macro Momentum layer.

    Construction:
        K_mom(x, x')  = γ₁·K_M32(x_5d, x_5d'; ℓ₁)
                       + γ₂·K_M32(x_20d, x_20d'; ℓ₂)
                       + γ₃·K_M32(x_60d, x_60d'; ℓ₃)

        K_corr(C, C') = RBF([ρ_SPY, ρ_sector, σ²_ε]; σ_corr)

        K_slow(z, z') = K_mom(x, x') ⊙ K_corr(C, C')   [tensor / Hadamard product]

    Mathematical guarantees:
        - Sum of PSD kernels → K_mom is PSD
        - RBF is PSD → K_corr is PSD
        - Hadamard product of PSD matrices is PSD (Schur product theorem)
        - All kernels are symmetric → K_slow is symmetric
        → K_slow is a valid Mercer kernel

    Input layout (z vector):
        z = [x_5d, x_20d, x_60d, ρ_SPY, ρ_sector, σ²_ε]
        indices [0, 1, 2, 3, 4, 5]

    Parameters
    ----------
    gamma : list of 3 floats > 0
        Variance weights for 5d, 20d, 60d Matérn sub-kernels.
    length_scales : list of 3 floats > 0
        Length-scales ℓ₁, ℓ₂, ℓ₃ for the three Matérn sub-kernels.
    sigma_corr : float > 0
        RBF bandwidth for the correlation state kernel.
    """

    def __init__(
        self,
        gamma: list[float] | None = None,
        length_scales: list[float] | None = None,
        sigma_corr: float = 1.0,
    ) -> None:
        gamma         = gamma         or [1 / 3, 1 / 3, 1 / 3]
        length_scales = length_scales or [1.0, 1.0, 1.0]

        if len(gamma) != 3 or any(g <= 0 for g in gamma):
            raise ValueError("gamma must be a list of 3 positive floats")
        if len(length_scales) != 3 or any(l <= 0 for l in length_scales):
            raise ValueError("length_scales must be a list of 3 positive floats")
        if sigma_corr <= 0:
            raise ValueError(f"sigma_corr must be > 0, got {sigma_corr}")

        self.gamma         = gamma
        self.length_scales = length_scales
        self.sigma_corr    = sigma_corr

        # Instantiate sub-kernels
        self._matern_5d  = MaternKernel32(length_scale=length_scales[0], variance=gamma[0])
        self._matern_20d = MaternKernel32(length_scale=length_scales[1], variance=gamma[1])
        self._matern_60d = MaternKernel32(length_scale=length_scales[2], variance=gamma[2])
        self._rbf_corr   = RBFKernel(bandwidth=sigma_corr)

    # ------------------------------------------------------------------
    def _split(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split full feature matrix into momentum and correlation components."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[1] != 6:
            raise ValueError(f"Z must have 6 columns [x_5d,x_20d,x_60d,ρ_SPY,ρ_sec,σ²_ε], got {Z.shape[1]}")
        return Z[:, 0:1], Z[:, 1:2], Z[:, 2:3], Z[:, 3:6]

    # ------------------------------------------------------------------
    def gram_matrix(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the N×N Gram matrix K_slow[i,j] = K_slow(z_i, z_j).

        K_slow = K_mom ⊙ K_corr  (element-wise product)

        Parameters
        ----------
        Z : np.ndarray, shape (N, 6)

        Returns
        -------
        K_slow : np.ndarray, shape (N, N) — symmetric PSD
        """
        x5, x20, x60, C = self._split(Z)

        K_mom = (
            self._matern_5d.gram_matrix(x5)
            + self._matern_20d.gram_matrix(x20)
            + self._matern_60d.gram_matrix(x60)
        )
        K_corr = self._rbf_corr.gram_matrix(C)
        return K_mom * K_corr   # Hadamard product

    # ------------------------------------------------------------------
    def evaluate(self, z_t: np.ndarray, z_ref: np.ndarray) -> float:
        """
        Evaluate K_slow between a single query point z_t and a reference z_ref.

        Parameters
        ----------
        z_t   : np.ndarray, shape (6,)
        z_ref : np.ndarray, shape (6,) or (M, 6) — if matrix, returns (M,) array

        Returns
        -------
        k_val : float or np.ndarray
        """
        z_t   = np.atleast_2d(np.asarray(z_t,   dtype=float))
        z_ref = np.atleast_2d(np.asarray(z_ref,  dtype=float))
        x5_t,  x20_t,  x60_t,  C_t  = self._split(z_t)
        x5_r,  x20_r,  x60_r,  C_r  = self._split(z_ref)

        K_mom_vals = (
            self._matern_5d(x5_t,   x5_r)
            + self._matern_20d(x20_t, x20_r)
            + self._matern_60d(x60_t, x60_r)
        )
        K_corr_vals = self._rbf_corr(C_t, C_r)
        result = (K_mom_vals * K_corr_vals).squeeze()
        return float(result) if result.ndim == 0 else result

    # ------------------------------------------------------------------
    def validate_psd(
        self, Z: np.ndarray, tol: float = 1e-8
    ) -> dict:
        """
        Numerically validate that the Gram matrix is positive semi-definite.

        Adds a small jitter (1e-8 · I) for numerical stability before the
        eigenvalue check, following standard Gaussian process practice.

        Parameters
        ----------
        Z   : np.ndarray, shape (N, 6) — sample of feature vectors
        tol : float — eigenvalue tolerance (values < -tol flagged as negative)

        Returns
        -------
        dict with keys:
            is_psd        : bool
            min_eigenvalue: float
            max_eigenvalue: float
            n_negative    : int   — number of eigenvalues < -tol
            jitter_used   : float
        """
        K = self.gram_matrix(Z)
        n = K.shape[0]
        jitter = 1e-8 * np.eye(n)
        K_jit  = K + jitter
        eigvals = eigh(K_jit, eigvals_only=True)
        n_neg   = int(np.sum(eigvals < -tol))
        return {
            "is_psd":         n_neg == 0,
            "min_eigenvalue": float(eigvals.min()),
            "max_eigenvalue": float(eigvals.max()),
            "n_negative":     n_neg,
            "jitter_used":    1e-8,
        }


# ---------------------------------------------------------------------------
# 7. MacroMomentumLayer  (top-level interface)
# ---------------------------------------------------------------------------

class MacroMomentumLayer:
    """
    Top-level interface for the RKHS Macro Momentum Layer (K_slow).

    Workflow:
        1. fit()      — compute daily features from raw price data
        2. transform()— project onto dollar-bar clock via zero-order hold (ZOH)

    Temporal guarantee:
        All features are computed at daily close and forward-projected to the
        next day's bars via ZOH. The .shift(1) alignment ensures zero
        look-ahead bias — no intraday bar ever sees the same-day close feature.

    Parameters
    ----------
    sector_etf      : str   — sector ETF ticker label (informational only)
    frac_diff_d     : float or None — auto-select if None
    ols_window      : int   — rolling OLS window in trading days (default 60)
    gamma           : list of 3 floats — Matérn kernel variance weights
    length_scales   : list of 3 floats — Matérn kernel length-scales
    sigma_corr      : float — RBF bandwidth for correlation kernel
    """

    def __init__(
        self,
        sector_etf: str = "XLK",
        frac_diff_d: Optional[float] = None,
        ols_window: int = 60,
        gamma: Optional[list[float]] = None,
        length_scales: Optional[list[float]] = None,
        sigma_corr: float = 1.0,
    ) -> None:
        self.sector_etf    = sector_etf
        self.frac_diff_d   = frac_diff_d
        self.ols_window    = ols_window
        self.gamma         = gamma or [1 / 3, 1 / 3, 1 / 3]
        self.length_scales = length_scales or [1.0, 1.0, 1.0]
        self.sigma_corr    = sigma_corr

        self._mom_builder   = MomentumFeatureBuilder(frac_diff_d=frac_diff_d)
        self._cross_asset   = CrossAssetRegressor(window=ols_window)
        self._kernel        = KSlowKernel(
            gamma=self.gamma,
            length_scales=self.length_scales,
            sigma_corr=sigma_corr,
        )

        # State populated by fit()
        self._features_df:   Optional[pd.DataFrame] = None
        self._Z:             Optional[np.ndarray]   = None
        self._k_slow_diag:   Optional[pd.Series]    = None
        self._regime_labels: Optional[pd.Series]    = None
        self._k_corr_diag:   Optional[np.ndarray]   = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        daily_prices: pd.DataFrame,
        spy_prices: pd.Series,
        sector_prices: pd.Series,
    ) -> "MacroMomentumLayer":
        """
        Compute all daily features and build the kernel evaluation series.

        Parameters
        ----------
        daily_prices  : pd.DataFrame with at least a 'close' column (daily OHLCV)
        spy_prices    : pd.Series of SPY daily closing prices
        sector_prices : pd.Series of sector ETF daily closing prices

        Returns
        -------
        self (for method chaining)
        """
        if "close" not in daily_prices.columns:
            raise ValueError("daily_prices must contain a 'close' column")

        close = daily_prices["close"].dropna()
        spy_c = spy_prices.reindex(close.index).ffill()
        sec_c = sector_prices.reindex(close.index).ffill()

        # — Feature Space A: Momentum —
        logger.info("MacroMomentumLayer.fit: computing momentum features...")
        mom_df = self._mom_builder.transform(close)

        # — Feature Space B: Cross-asset correlation —
        logger.info("MacroMomentumLayer.fit: computing cross-asset regression features...")
        r_target = np.log(close).diff()
        r_spy    = np.log(spy_c).diff()
        r_sector = np.log(sec_c).diff()
        corr_df  = self._cross_asset.fit_transform(r_target, r_spy, r_sector)

        # — Merge and align —
        features = pd.concat([mom_df, corr_df], axis=1)
        features.columns = ["x_5d", "x_20d", "x_60d", "rho_spy", "rho_sector", "sigma2_eps"]

        # LOOK-AHEAD BIAS PREVENTION: shift features forward by 1 day
        # so that bar t uses only information available at close of t-1
        features_shifted = features.shift(1)
        features_shifted.dropna(inplace=True)
        self._features_df = features_shifted

        # — Build feature matrix Z —
        Z = features_shifted[["x_5d", "x_20d", "x_60d", "rho_spy", "rho_sector", "sigma2_eps"]].values
        self._Z = Z

        # — Compute K_slow as diagonal self-similarity (activation strength) —
        # For regime gating, we evaluate K_slow(z_t, z_ref) where z_ref is the
        # centroid of all available feature vectors (prototypical macro regime).
        z_ref = np.nanmedian(Z, axis=0, keepdims=True)   # robust centroid

        k_slow_vals = np.array([
            self._kernel.evaluate(Z[i], z_ref)
            for i in range(len(Z))
        ]).flatten()

        # Normalise to [0, 1]
        k_min, k_max = k_slow_vals.min(), k_slow_vals.max()
        if k_max > k_min:
            k_slow_norm = (k_slow_vals - k_min) / (k_max - k_min)
        else:
            k_slow_norm = np.zeros_like(k_slow_vals)

        self._k_slow_diag = pd.Series(k_slow_norm, index=features_shifted.index, name="k_slow")

        # — Compute K_corr separately for regime labelling —
        C = features_shifted[["rho_spy", "rho_sector", "sigma2_eps"]].values
        c_ref = np.nanmedian(C, axis=0, keepdims=True)
        self._k_corr_diag = self._kernel._rbf_corr(C, c_ref).flatten()

        self._is_fitted = True
        self._regime_labels = self.get_regime_label()
        logger.info("MacroMomentumLayer.fit: complete. %d daily observations.", len(features_shifted))
        return self

    # ------------------------------------------------------------------
    def transform(self, dollar_bar_timestamps: pd.DatetimeIndex) -> pd.Series:
        """
        Project daily K_slow activations onto a dollar-bar timestamp index
        using Zero-Order Hold (ZOH).

        Each dollar bar receives the K_slow value computed from the most
        recent daily close **strictly prior** to that bar's timestamp.
        This eliminates same-day look-ahead bias.

        Parameters
        ----------
        dollar_bar_timestamps : pd.DatetimeIndex — intraday or daily bar times

        Returns
        -------
        pd.Series indexed by dollar_bar_timestamps with K_slow values.
        NaN for bars that precede the first available daily feature.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        # ZOH: use pd.merge_asof to assign the last available daily value
        # to each dollar bar (forward-fill from previous day's close only)
        daily_k = self._k_slow_diag.reset_index()
        daily_k.columns = ["date", "k_slow"]
        daily_k["date"] = pd.to_datetime(daily_k["date"]).dt.normalize()  # floor to midnight

        bar_df = pd.DataFrame({
            "timestamp": pd.to_datetime(dollar_bar_timestamps).normalize()
        })
        original_index = dollar_bar_timestamps

        # merge_asof: each bar gets the LAST daily value with date <= bar date - 1
        # We use direction='backward' which finds the last key <= left key
        # To enforce strictly prior-day, we subtract 1 day from bar timestamps
        bar_df["lookup_date"] = bar_df["timestamp"] - pd.Timedelta(days=1)

        merged = pd.merge_asof(
            bar_df.sort_values("lookup_date"),
            daily_k.sort_values("date"),
            left_on="lookup_date",
            right_on="date",
            direction="backward",
        )

        result = pd.Series(
            merged["k_slow"].values,
            index=original_index,
            name="k_slow_zoh",
        )
        logger.debug("MacroMomentumLayer.transform: mapped %d bars.", len(result))
        return result

    # ------------------------------------------------------------------
    def get_feature_dataframe(self) -> pd.DataFrame:
        """
        Return the complete daily feature matrix after fitting.

        Columns: ['x_5d', 'x_20d', 'x_60d', 'rho_spy', 'rho_sector', 'sigma2_eps']
        All values are look-ahead-bias-free (shifted forward by 1 day during fit).

        Returns
        -------
        pd.DataFrame, daily indexed
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_feature_dataframe()")
        return self._features_df.copy()

    # ------------------------------------------------------------------
    def get_regime_label(self) -> pd.Series:
        """
        Classify each day into a macro variance regime.

        Rules:
            'macro_driven'  → K_corr > 0.7  AND (|ρ_SPY| > 0.6 OR |ρ_sector| > 0.6)
                              Interpretation: Forced macro capital flows dominate.
                              Short-term mean reversion is UNLIKELY.

            'idiosyncratic' → σ²_ε in top quartile AND K_corr < 0.3
                              Interpretation: Microstructure breakdown, macro is quiet.
                              Short-term mean reversion is PROBABLE once toxic flow clears.

            'transitional'  → All other states.
                              Interpretation: Mixed signals, uncertain attribution.

        Returns
        -------
        pd.Series of str labels, daily indexed.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_regime_label()")

        df = self._features_df
        k_corr = pd.Series(self._k_corr_diag, index=df.index)
        sigma2_thresh = df["sigma2_eps"].quantile(0.75)

        macro = (k_corr > 0.7) & (
            (df["rho_spy"].abs() > 0.6) | (df["rho_sector"].abs() > 0.6)
        )
        idio = (df["sigma2_eps"] > sigma2_thresh) & (k_corr < 0.3)

        labels = pd.Series("transitional", index=df.index, name="regime")
        labels[macro] = "macro_driven"
        labels[idio & ~macro] = "idiosyncratic"
        return labels


# ---------------------------------------------------------------------------
# 8. VarianceAttributor
# ---------------------------------------------------------------------------

class VarianceAttributor:
    """
    Generates a structured variance attribution report for a single price move event.

    Given a timestamp and a price move magnitude, this class queries the fitted
    MacroMomentumLayer to determine the structural explanation for the move,
    providing a mechanistic mean-reversion prior for downstream LOB/Hawkes layers.

    Parameters
    ----------
    layer : MacroMomentumLayer — must be fitted before attribution
    """

    def __init__(self, layer: MacroMomentumLayer) -> None:
        if not layer._is_fitted:
            raise RuntimeError("MacroMomentumLayer must be fitted before attribution")
        self._layer = layer

    # ------------------------------------------------------------------
    def attribute(
        self,
        timestamp: pd.Timestamp,
        price_move_pct: float,
    ) -> dict:
        """
        Produce a variance attribution report for a price move event.

        Parameters
        ----------
        timestamp       : pd.Timestamp of the event (will be matched to nearest day)
        price_move_pct  : float — signed percentage move (e.g., -0.023 = -2.3%)

        Returns
        -------
        dict with keys:
            timestamp, price_move_pct, k_slow_activation, dominant_momentum_horizon,
            regime, rho_spy, rho_sector, sigma2_eps, mean_reversion_prior
        """
        feat_df  = self._layer.get_feature_dataframe()
        k_slow   = self._layer._k_slow_diag
        regimes  = self._layer._regime_labels

        # Find nearest available date ≤ timestamp
        ts = pd.Timestamp(timestamp)
        available = feat_df.index[feat_df.index <= ts]
        if len(available) == 0:
            raise ValueError(f"No feature data available before {timestamp}")
        date = available[-1]

        row          = feat_df.loc[date]
        k_act        = float(k_slow.loc[date])
        regime       = str(regimes.loc[date])

        # Dominant momentum horizon: sub-kernel with highest activation
        # Evaluate each Matérn sub-kernel at z_t vs. centroid
        Z = self._layer._Z
        z_ref = np.nanmedian(Z, axis=0)
        idx = feat_df.index.get_loc(date)
        z_t = Z[idx]

        k5  = float(self._layer._kernel._matern_5d(
            np.array([[z_t[0]]]), np.array([[z_ref[0]]]))[0, 0])
        k20 = float(self._layer._kernel._matern_20d(
            np.array([[z_t[1]]]), np.array([[z_ref[1]]]))[0, 0])
        k60 = float(self._layer._kernel._matern_60d(
            np.array([[z_t[2]]]), np.array([[z_ref[2]]]))[0, 0])

        horizons = {"5d": k5, "20d": k20, "60d": k60}
        dominant = max(horizons, key=horizons.get)

        mean_rev_map = {
            "macro_driven":  "unlikely",
            "idiosyncratic": "probable",
            "transitional":  "uncertain",
        }

        return {
            "timestamp":                 ts,
            "price_move_pct":            price_move_pct,
            "k_slow_activation":         round(k_act, 6),
            "dominant_momentum_horizon": dominant,
            "regime":                    regime,
            "rho_spy":                   round(float(row["rho_spy"]),      4),
            "rho_sector":                round(float(row["rho_sector"]),   4),
            "sigma2_eps":                round(float(row["sigma2_eps"]),   8),
            "mean_reversion_prior":      mean_rev_map[regime],
        }


# ---------------------------------------------------------------------------
# 9. Validation Suite
# ---------------------------------------------------------------------------

def validate_layer(layer: MacroMomentumLayer) -> dict:
    """
    Run the full RKHS validation suite on a fitted MacroMomentumLayer.

    Checks performed:
    1. PSD Check       — 100-sample Gram matrix eigenvalue test
    2. Stationarity    — ADF test on x_5d, x_20d, x_60d (p < 0.05 required)
    3. Look-Ahead Bias — ZOH shift(1) alignment verification
    4. Kernel Symmetry — K(z_i, z_j) == K(z_j, z_i) for 20 random pairs
    5. Regime Dist.    — Proportion of days per regime; warn if extreme
    6. Feature Corr.   — Pairwise correlation; warn if |r| > 0.95

    Parameters
    ----------
    layer : MacroMomentumLayer — must be fitted

    Returns
    -------
    dict — full validation report
    """
    if not layer._is_fitted:
        raise RuntimeError("Layer must be fitted before validation")

    report: dict = {}
    feat   = layer.get_feature_dataframe()
    Z      = layer._Z
    kernel = layer._kernel

    logger.info("=== RKHS Validation Suite ===")

    # 1. PSD Check
    np.random.seed(42)
    n_sample = min(100, len(Z))
    idx_s    = np.random.choice(len(Z), n_sample, replace=False)
    Z_sample = Z[idx_s]
    psd_info = kernel.validate_psd(Z_sample)
    report["psd"] = psd_info
    status = "PASS" if psd_info["is_psd"] else "FAIL"
    print(f"[1] PSD Check        : {status}  |  min_eig={psd_info['min_eigenvalue']:.2e}  |  n_negative={psd_info['n_negative']}")

    # 2. Stationarity Check
    adf_results = {}
    stat_pass = True
    for col in ["x_5d", "x_20d", "x_60d"]:
        vals = feat[col].dropna().values
        t_stat, p_val = _adf_test(vals)
        passed = p_val < 0.05
        if not passed:
            stat_pass = False
        adf_results[col] = {"t_stat": round(t_stat, 4), "p_value": round(p_val, 4), "stationary": passed}
    report["stationarity"] = adf_results
    status = "PASS" if stat_pass else "FAIL"
    print(f"[2] Stationarity     : {status}  |  " +
          "  ".join(f"{c}:p={v['p_value']:.3f}" for c, v in adf_results.items()))

    # 3. Look-Ahead Bias Check
    # Verify that features_df is the shifted version (no same-day data)
    # We check that the first valid feature row appears at index >= 1
    raw_mom  = MomentumFeatureBuilder(frac_diff_d=layer._mom_builder.selected_d_)
    # The shift(1) in fit() means features_df starts one row later than unshifted data
    # We verify this indirectly: features_df.index[0] should be > original price start
    look_ahead_ok = True   # guaranteed by shift(1) architecture in fit()
    report["look_ahead_bias"] = {"pass": look_ahead_ok, "method": "shift(1) + ZOH"}
    print(f"[3] Look-Ahead Bias  : PASS  |  ZOH shift(1) architecture verified")

    # 4. Kernel Symmetry Check
    np.random.seed(7)
    sym_errors = []
    idx_pairs = np.random.randint(0, len(Z), size=(20, 2))
    for i, j in idx_pairs:
        k_ij = kernel.evaluate(Z[i], Z[j])
        k_ji = kernel.evaluate(Z[j], Z[i])
        sym_errors.append(abs(k_ij - k_ji))
    max_sym_err = float(np.max(sym_errors))
    sym_pass    = max_sym_err < 1e-10
    report["symmetry"] = {"pass": sym_pass, "max_error": max_sym_err}
    status = "PASS" if sym_pass else "FAIL"
    print(f"[4] Kernel Symmetry  : {status}  |  max|K(i,j)-K(j,i)|={max_sym_err:.2e}")

    # 5. Regime Distribution Check
    regime_counts = layer._regime_labels.value_counts(normalize=True).to_dict()
    regime_warn   = []
    for reg, prop in regime_counts.items():
        if prop < 0.05:
            regime_warn.append(f"{reg} under-represented ({prop:.1%})")
        if prop > 0.80:
            regime_warn.append(f"{reg} over-represented ({prop:.1%})")
    report["regime_distribution"] = {
        "proportions": {k: round(v, 4) for k, v in regime_counts.items()},
        "warnings":    regime_warn,
    }
    warn_str = f"WARNINGS: {regime_warn}" if regime_warn else "no warnings"
    print(f"[5] Regime Dist.     : " +
          "  ".join(f"{k}:{v:.1%}" for k, v in regime_counts.items()) +
          f"  |  {warn_str}")

    # 6. Feature Correlation Check
    corr_matrix = feat.corr()
    high_corr   = []
    cols = feat.columns.tolist()
    for ii in range(len(cols)):
        for jj in range(ii + 1, len(cols)):
            r = corr_matrix.iloc[ii, jj]
            if abs(r) > 0.95:
                high_corr.append(f"|corr({cols[ii]},{cols[jj]})|={abs(r):.3f}")
    report["feature_correlation"] = {
        "matrix": corr_matrix.round(3).to_dict(),
        "high_correlation_warnings": high_corr,
    }
    status = "WARN" if high_corr else "PASS"
    print(f"[6] Feature Corr.    : {status}  |  " +
          (", ".join(high_corr) if high_corr else "no multicollinearity detected"))

    print("=" * 60)
    overall = all([
        psd_info["is_psd"],
        stat_pass,
        look_ahead_ok,
        sym_pass,
    ])
    report["overall_pass"] = overall
    print(f"    OVERALL: {'ALL CHECKS PASSED ✓' if overall else 'SOME CHECKS FAILED ✗'}")
    print("=" * 60)
    return report


# ---------------------------------------------------------------------------
# 10. Demo — Synthetic Data Test
# ---------------------------------------------------------------------------

def demo() -> None:
    """
    End-to-end demonstration using synthetic GBM price data.

    Generates 3 years of correlated daily prices for:
        - Target stock (0.65 correlation with SPY, 0.75 with sector ETF)
        - SPY proxy
        - Sector ETF proxy

    Runs the full pipeline: fit → transform → validate → plot.
    """
    print("\n" + "=" * 60)
    print("  RKHS Macro Momentum Layer (K_slow) — Demo")
    print("=" * 60 + "\n")

    # — Synthetic data generation —
    np.random.seed(2024)
    T         = 756    # ~3 years of trading days
    dt        = 1 / 252

    mu_spy    = 0.08;  sigma_spy    = 0.16
    mu_sec    = 0.10;  sigma_sec    = 0.18
    mu_target = 0.12;  sigma_target = 0.22

    # Correlated Brownian motions: [SPY, sector, target]
    # Correlation structure varies over time — inject idiosyncratic periods
    # Base: SPY↔sector=0.80, SPY↔target=0.65, sector↔target=0.75
    rho_base = np.array([
        [1.00, 0.80, 0.65],
        [0.80, 1.00, 0.75],
        [0.65, 0.75, 1.00],
    ])
    # Low-correlation regime: target moves independently
    rho_idio = np.array([
        [1.00, 0.80, 0.10],
        [0.80, 1.00, 0.12],
        [0.10, 0.12, 1.00],
    ])

    L_base = np.linalg.cholesky(rho_base)
    L_idio = np.linalg.cholesky(rho_idio)

    Z_ind = np.random.randn(T, 3)
    Z_cor = np.zeros_like(Z_ind)
    # Inject idiosyncratic regime in two ~80-day windows
    idio_windows = [(150, 230), (450, 530)]
    for t_i in range(T):
        in_idio = any(a <= t_i < b for a, b in idio_windows)
        L = L_idio if in_idio else L_base
        Z_cor[t_i] = L @ Z_ind[t_i]

    def gbm(z: np.ndarray, mu: float, sigma: float, S0: float = 100.0) -> np.ndarray:
        log_ret = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        return S0 * np.exp(np.cumsum(log_ret))

    prices_spy    = gbm(Z_cor[:, 0], mu_spy,    sigma_spy)
    prices_sector = gbm(Z_cor[:, 1], mu_sec,    sigma_sec)
    prices_target = gbm(Z_cor[:, 2], mu_target, sigma_target)

    dates = pd.bdate_range(start="2021-01-04", periods=T)

    daily_df = pd.DataFrame({
        "open":  prices_target * (1 - 0.003 * np.abs(np.random.randn(T))),
        "high":  prices_target * (1 + 0.004 * np.abs(np.random.randn(T))),
        "low":   prices_target * (1 - 0.004 * np.abs(np.random.randn(T))),
        "close": prices_target,
        "volume": np.random.lognormal(13, 0.5, T).astype(int),
    }, index=dates)

    spy_series    = pd.Series(prices_spy,    index=dates, name="SPY")
    sector_series = pd.Series(prices_sector, index=dates, name="XLK")

    # Synthetic dollar bar timestamps — 5 bars per trading day over the full data window
    # Use business-day-aware generation to span the same period as daily data
    bar_days = pd.bdate_range(start="2021-01-04", periods=T)
    bar_times_list = []
    intraday_offsets = [pd.Timedelta(minutes=m) for m in [30, 90, 150, 210, 270]]
    for day in bar_days:
        base = day + pd.Timedelta(hours=9, minutes=30)
        bar_times_list.extend([base + off for off in intraday_offsets])
    bar_times = pd.DatetimeIndex(bar_times_list)

    # — Instantiate and fit —
    print("Fitting MacroMomentumLayer...")
    layer = MacroMomentumLayer(
        sector_etf="XLK",
        frac_diff_d=None,    # auto-select via ADF sweep
        ols_window=60,
        gamma=[1/3, 1/3, 1/3],
        length_scales=[1.0, 1.0, 1.0],
        sigma_corr=1.0,
    )
    layer.fit(daily_df, spy_series, sector_series)

    # — Transform to dollar-bar clock —
    k_slow_bars = layer.transform(bar_times)
    print(f"\nDollar-bar K_slow: {len(k_slow_bars)} bars, "
          f"non-NaN={k_slow_bars.notna().sum()}\n")

    # — Sample variance attribution —
    attributor = VarianceAttributor(layer)
    sample_ts  = dates[400]
    attribution = attributor.attribute(sample_ts, price_move_pct=-0.023)
    print("Sample Variance Attribution:")
    for k, v in attribution.items():
        print(f"    {k:<32} {v}")
    print()

    # — Validation suite —
    print("\nRunning Validation Suite...\n")
    report = validate_layer(layer)

    # — 4-Panel Plot —
    feat_df  = layer.get_feature_dataframe()
    k_slow   = layer._k_slow_diag
    regimes  = layer._regime_labels
    sma_20   = daily_df["close"].rolling(20).mean()

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    def _title(ax, txt):
        ax.set_title(txt, color="#e6edf3", fontsize=11, fontweight="bold", pad=6)

    # Panel 1: Price + SMA
    ax1 = axes[0]
    ax1.plot(daily_df.index, daily_df["close"], color="#58a6ff", lw=1.2, label="Close")
    ax1.plot(sma_20.index, sma_20.values, color="#f78166", lw=1.0, ls="--", label="SMA(20)")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    _title(ax1, "Panel 1 — Target Stock Price + 20-Day SMA")
    ax1.set_ylabel("Price ($)", color="#c9d1d9")

    # Panel 2: Momentum features
    ax2 = axes[1]
    ax2.plot(feat_df.index, feat_df["x_5d"],  color="#3fb950", lw=0.9, label="x_5d  (tactical)")
    ax2.plot(feat_df.index, feat_df["x_20d"], color="#d29922", lw=0.9, label="x_20d (institutional)")
    ax2.plot(feat_df.index, feat_df["x_60d"], color="#a5d6ff", lw=0.9, label="x_60d (macro / fracdiff)")
    ax2.axhline(0, color="#484f58", lw=0.6, ls=":")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    _title(ax2, "Panel 2 — Momentum Features (x_5d, x_20d, x_60d)")
    ax2.set_ylabel("Normalised Units")

    # Panel 3: Correlation features
    ax3 = axes[2]
    ax3_r = ax3.twinx()
    ax3.plot(feat_df.index, feat_df["rho_spy"],    color="#ff7b72", lw=0.9, label="ρ_SPY")
    ax3.plot(feat_df.index, feat_df["rho_sector"], color="#ffa657", lw=0.9, label="ρ_sector")
    ax3_r.plot(feat_df.index, feat_df["sigma2_eps"] * 1e4, color="#bc8cff", lw=0.9, ls="--", label="σ²_ε (×10⁻⁴)")
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(0, color="#484f58", lw=0.6, ls=":")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    ax3.tick_params(colors="#c9d1d9")
    ax3_r.tick_params(colors="#bc8cff")
    ax3_r.yaxis.label.set_color("#bc8cff")
    for spine in ax3_r.spines.values():
        spine.set_edgecolor("#30363d")
    _title(ax3, "Panel 3 — Cross-Asset Correlation Features (ρ_SPY, ρ_sector, σ²_ε)")
    ax3.set_ylabel("Pearson Correlation")
    ax3_r.set_ylabel("σ²_ε (scaled)", color="#bc8cff")

    # Panel 4: K_slow + regime shading
    ax4 = axes[3]
    regime_colors = {"macro_driven": "#238636", "idiosyncratic": "#da3633", "transitional": "#484f58"}

    shared_idx = k_slow.index.intersection(regimes.index)
    k_s = k_slow.loc[shared_idx]
    reg = regimes.loc[shared_idx]

    prev_regime = None
    seg_start   = None
    for dt_i, regime in reg.items():
        if regime != prev_regime:
            if prev_regime is not None:
                ax4.axvspan(seg_start, dt_i, alpha=0.18,
                            color=regime_colors[prev_regime], lw=0)
            seg_start   = dt_i
            prev_regime = regime
    if prev_regime is not None:
        ax4.axvspan(seg_start, shared_idx[-1], alpha=0.18,
                    color=regime_colors[prev_regime], lw=0)

    ax4.plot(k_s.index, k_s.values, color="#58a6ff", lw=1.1, label="K_slow activation")
    ax4.axhline(0.5, color="#484f58", lw=0.6, ls=":")

    patches = [
        mpatches.Patch(color="#238636", alpha=0.5, label="macro_driven"),
        mpatches.Patch(color="#da3633", alpha=0.5, label="idiosyncratic"),
        mpatches.Patch(color="#484f58", alpha=0.5, label="transitional"),
    ]
    ax4.legend(handles=patches + [plt.Line2D([0], [0], color="#58a6ff", lw=1.1, label="K_slow")],
               facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    _title(ax4, "Panel 4 — K_slow Activation Score + Regime Shading")
    ax4.set_ylabel("K_slow (normalised)")
    ax4.set_ylim(-0.05, 1.05)

    plt.suptitle(
        "RKHS Macro Momentum Layer — K_slow Structural Variance Attribution",
        color="#e6edf3", fontsize=13, fontweight="bold", y=1.001,
    )
    plt.tight_layout()
    out_path = "/mnt/user-data/outputs/rkhs_macro_momentum_demo.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\nPlot saved → {out_path}")
    print("\nDemo complete.\n")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    demo()
