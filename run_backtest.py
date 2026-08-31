"""
Run Backtest — Pristine-RKHS Pipeline
=======================================
Traders@SMU — Quantitative Strategies Group

Runs the full walk-forward backtest using real CL futures data.

v1 Architecture (default):
    Fast layer: LOB + VPIN + Kyle's Lambda + Hawkes [+ VannaCharm]
    Slow layer: VRP + MacroMotion [+ Sentiment]
    Gates:      MomentumGate (Arjan) + EventProximity + CKA redundancy

v2 Architecture (ARCH_VERSION=v2):
    Fast layer: OrderFlow (PCA-consolidated) [+ VannaCharm]
    Slow layer: VRP + MacroMotion + Sentiment
    Gates:      MomentumGate + EventProximity + VPIN/Hawkes position-sizing

Architecture: K_total = K_fast + β · K_fast · K_slow_gated

Usage:
    python3 run_backtest.py                         # v1 default
    ARCH_VERSION=v2 python3 run_backtest.py          # v2 restructured
"""

import sys
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Every RNG in this file is seeded through _seed() so a whole run can be
# repeated under a different draw with SEED=<n>. Random Fourier features,
# PCA and the CKA subsample all depend on it, so a result that only holds
# for SEED=0 is a result about one draw, not about the strategy.
_SEED_OFFSET = int(os.environ.get("SEED", "0"))


def _seed(base: int) -> int:
    return base + _SEED_OFFSET
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kernels"))
sys.path.insert(0, os.path.join(_ROOT, "backtesting"))
sys.path.insert(0, os.path.join(_ROOT, "analytics"))
sys.path.insert(0, os.path.join(_ROOT, "reporting"))

import pandas as pd
from data_loader import DataLoader, LOBSnapshot
from feature_alignment import (
    to_epoch_seconds, build_master_bar_schedule, align_fast_features,
    first_valid_bar, check_target_not_degenerate, reindex_daily,
    expanding_zscore, expanding_mean,
)

# Kernels
from kernels.LOB import (
    VolumeProfileKernel, BookShapeKernel, DepthImbalanceKernel,
    RKHSLayer, MaternRFF,
)
from kernels.momentum_gate import MomentumGate

# Backtesting
from backtesting.walk_forward import (
    PurgedWalkForward, MultiResolutionAligner,
    TwoLevelKernelCombiner, exponential_decay_weights,
    CKARedundancyGate,
)
from backtesting.signal_definition import (
    OUSignalGenerator, estimate_ou_params, generate_positions,
    multi_horizon_s_scores, KalmanSignalGenerator,
)
from backtesting.metrics import (
    deflated_sharpe_ratio, sortino_ratio,
    centered_kernel_alignment, effective_dimensionality,
)

# ============================================================================
# Adapter: wraps a raw feature matrix as a KernelLike with RFF feature_map
# ============================================================================

class FeatureMatrixKernel:
    """
    Wraps a pre-computed (n, d) feature matrix into a KernelLike object
    that produces RFF feature maps. This lets us plug VPIN, Kyle's Lambda,
    VRP, and MacroMotion features into the TwoLevelKernelCombiner.
    """

    def __init__(self, features: np.ndarray, nu=1.5, length_scale=1.0,
                 n_rff=500, name="features", seed=_seed(42)):
        self.features = features
        self.name = name
        self.nu = nu
        self.length_scale = length_scale
        self.n_rff = n_rff
        self._d_input = features.shape[1]
        self._seed = seed
        self._rff = MaternRFF(
            nu=nu, length_scale=length_scale,
            n_features=n_rff, d_input=features.shape[1], seed=seed,
        )
        # Aliases for inner_cv_grid_search compatibility
        self.rff = self._rff
        self._w = None
        self._fitted = False

    def _init_rff(self):
        """Re-initialize RFF (called by inner_cv_grid_search)."""
        self._rff = MaternRFF(
            nu=self.nu, length_scale=self.length_scale,
            n_features=self.n_rff, d_input=self._d_input, seed=self._seed,
        )
        self.rff = self._rff

    def feature_map(self, data) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return self._rff.transform(data)
        # If data is indices, look up from stored features
        idx = np.asarray(data, dtype=int)
        return self._rff.transform(self.features[idx])

    def fit(self, data, y, reg_lambda=1e-3, **kwargs):
        Phi = self.feature_map(data)
        D = Phi.shape[1]
        self._w = np.linalg.solve(
            Phi.T @ Phi + reg_lambda * np.eye(D), Phi.T @ y
        )
        self._fitted = True

    def predict(self, data) -> np.ndarray:
        return self.feature_map(data) @ self._w


class CombinedKernel:
    """
    Combines multiple FeatureMatrixKernels by slicing the correct
    columns from a concatenated feature matrix and applying each
    sub-kernel's RFF independently, then concatenating the outputs.
    """

    def __init__(self, kernels, col_slices, weights=None):
        """
        Parameters
        ----------
        kernels : list of FeatureMatrixKernel
        col_slices : list of slice
            Which columns of the input data belong to each kernel.
        weights : list of float
        """
        self.kernels = kernels
        self.col_slices = col_slices
        n = len(kernels)
        self.weights = np.array(weights if weights else [1.0 / n] * n)
        self._w = None
        self._fitted = False

    def feature_map(self, data) -> np.ndarray:
        phis = []
        for k, sl, w in zip(self.kernels, self.col_slices, self.weights):
            sub_data = data[:, sl] if isinstance(data, np.ndarray) else data
            phis.append(np.sqrt(w) * k.feature_map(sub_data))
        return np.concatenate(phis, axis=1)

    def fit(self, data, y, reg_lambda=1e-3, **kwargs):
        Phi = self.feature_map(data)
        D = Phi.shape[1]
        self._w = np.linalg.solve(
            Phi.T @ Phi + reg_lambda * np.eye(D), Phi.T @ y
        )
        self._fitted = True

    def predict(self, data) -> np.ndarray:
        return self.feature_map(data) @ self._w


# ============================================================================
# VPIN feature builder (from trades)
# ============================================================================

def build_vpin_features(trades_df, dollar_threshold=1_000_000, vpin_window=50,
                        return_timestamps=False):
    """Build VPIN feature matrix: [VPIN, delta_VPIN, bar_vol, rel_volume].

    With return_timestamps=True also returns the epoch-second close time of
    each bar, so features can be aligned by wall clock instead of row number.
    """
    prices = trades_df["price"].values
    sizes = trades_df["size"].values.astype(float)
    trade_ts = to_epoch_seconds(trades_df["ts_event"])

    # Dollar bars
    bars = []
    bar_close_ts = []
    cum_dol = 0.0
    bar_prices, bar_sizes = [], []
    for i in range(len(prices)):
        p, s = prices[i], sizes[i]
        cum_dol += p * s
        bar_prices.append(p)
        bar_sizes.append(s)
        if cum_dol >= dollar_threshold:
            bp = np.array(bar_prices)
            bs = np.array(bar_sizes)
            bars.append({
                "close": bp[-1],
                "volume": bs.sum(),
                "dollar_volume": cum_dol,
                "volatility": bp.std() / (bp.mean() + 1e-8),
                # BVC: classify volume as buy/sell using tick rule proxy
                "buy_volume": bs[np.diff(bp, prepend=bp[0]) >= 0].sum(),
                "sell_volume": bs[np.diff(bp, prepend=bp[0]) < 0].sum(),
            })
            bar_close_ts.append(trade_ts[i])
            cum_dol = 0.0
            bar_prices, bar_sizes = [], []

    if len(bars) < vpin_window + 2:
        raise ValueError(f"Only {len(bars)} dollar bars, need {vpin_window + 2}")

    n_bars = len(bars)
    vpin = np.zeros(n_bars)
    for i in range(vpin_window, n_bars):
        window = bars[i - vpin_window:i]
        total_vol = sum(b["volume"] for b in window)
        if total_vol > 0:
            abs_imbalance = sum(
                abs(b["buy_volume"] - b["sell_volume"]) for b in window
            )
            vpin[i] = abs_imbalance / total_vol

    delta_vpin = np.diff(vpin, prepend=vpin[0])
    bar_vol = np.array([b["volatility"] for b in bars])
    volumes = np.array([b["volume"] for b in bars])
    # Relative to the volume seen so far, not to the whole sample's mean
    rel_vol = volumes / (expanding_mean(volumes) + 1e-8)

    features = np.column_stack([vpin, delta_vpin, bar_vol, rel_vol])
    bar_ts = np.asarray(bar_close_ts, dtype=float)
    # Trim warmup
    features = features[vpin_window:]
    bar_ts = bar_ts[vpin_window:]

    features = expanding_zscore(features)
    return (features, bar_ts) if return_timestamps else features


# ============================================================================
# Kyle's Lambda feature builder
# ============================================================================

def build_kyle_features(trades_df, windows=(1, 5, 20), return_timestamps=False):
    """Build Kyle's Lambda features: [lambda_1, lambda_5, lambda_20, residual_z].

    Bins are fixed trade counts, so ~5,000 rows span the entire tape. Without
    the returned timestamps these rows cannot be matched to any other kernel.
    """
    prices = trades_df["price"].values.astype(float)
    sizes = trades_df["size"].values.astype(float)
    trade_ts = to_epoch_seconds(trades_df["ts_event"])
    sides = trades_df["side"].values if "side" in trades_df.columns else None

    if sides is not None:
        signed = sizes * np.where(sides == "A", 1.0, np.where(sides == "B", -1.0, 0.0))
    else:
        signed = sizes * np.sign(np.diff(prices, prepend=prices[0]))

    # Aggregate to 1-minute bins (approx)
    bin_size = max(len(prices) // 5000, 100)
    n_bins = len(prices) // bin_size

    delta_p = np.zeros(n_bins)
    net_flow = np.zeros(n_bins)
    for i in range(n_bins):
        sl = slice(i * bin_size, (i + 1) * bin_size)
        delta_p[i] = prices[sl][-1] - prices[sl][0] if len(prices[sl]) > 1 else 0
        net_flow[i] = signed[sl].sum()

    lambdas = {}
    for w in windows:
        lam = np.zeros(n_bins)
        for t in range(w, n_bins):
            dp = delta_p[t - w:t]
            nf = net_flow[t - w:t]
            if np.var(nf) > 1e-12:
                lam[t] = np.cov(dp, nf)[0, 1] / np.var(nf)
        lambdas[w] = lam

    # Residual z-score, against history only
    residual = delta_p - lambdas[5] * net_flow
    residual_z = expanding_zscore(residual)

    features = np.column_stack([lambdas[w] for w in windows] + [residual_z])
    # Each bin closes on its last trade
    bin_close_idx = np.minimum(np.arange(1, n_bins + 1) * bin_size - 1,
                               len(trade_ts) - 1)
    bin_ts = trade_ts[bin_close_idx]
    # Trim warmup
    features = features[max(windows):]
    bin_ts = bin_ts[max(windows):]

    features = expanding_zscore(features)
    return (features, bin_ts) if return_timestamps else features


# ============================================================================
# VRP feature builder
# ============================================================================

def build_vrp_features(daily_ohlcv, vix_df=None, pcr_df=None, return_dates=False):
    """Build VRP features: [vrp, term_struct, sin/cos_doy/dow, mom_5/21, log_vix, log_pcr].

    Trims 30 warm-up days off the front, so the returned rows start at
    daily_ohlcv.index[30] — return_dates hands back that index rather than
    leaving callers to assume row 0 is day 0.
    """
    close = daily_ohlcv["close"].values.astype(float)
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))

    n = len(close)
    rv_30 = np.zeros(n)
    for i in range(30, n):
        rv_30[i] = log_ret[i - 30:i].std() * np.sqrt(252)

    # Calendar features
    if hasattr(daily_ohlcv.index, 'dayofyear'):
        doy = daily_ohlcv.index.dayofyear.values.astype(float)
        dow = daily_ohlcv.index.dayofweek.values.astype(float)
    else:
        doy = np.arange(n, dtype=float) % 252
        dow = np.arange(n, dtype=float) % 5

    sin_doy = np.sin(2 * np.pi * doy / 252)
    cos_doy = np.cos(2 * np.pi * doy / 252)
    sin_dow = np.sin(2 * np.pi * dow / 5)
    cos_dow = np.cos(2 * np.pi * dow / 5)

    # Momentum
    mom_5 = np.zeros(n)
    mom_21 = np.zeros(n)
    for i in range(5, n):
        mom_5[i] = close[i] / close[i - 5] - 1
    for i in range(21, n):
        mom_21[i] = close[i] / close[i - 21] - 1

    # VIX and put/call
    if vix_df is not None and len(vix_df) >= n:
        vix_vals = vix_df["vix"].values[:n].astype(float)
    else:
        vix_vals = rv_30 * 100  # fallback

    iv_30 = vix_vals / 100
    vrp = iv_30 - rv_30
    term_struct = np.zeros(n)  # placeholder (iv_90 - iv_30)
    log_vix = np.log(vix_vals + 1e-8)

    if pcr_df is not None and len(pcr_df) >= n:
        pcr_vals = pcr_df["put_call_ratio"].values[:n].astype(float)
    else:
        pcr_vals = np.ones(n) * 0.8

    log_pcr = np.log(pcr_vals + 1e-8)

    features = np.column_stack([
        vrp, term_struct, sin_doy, cos_doy, sin_dow, cos_dow,
        mom_5, mom_21, log_vix, log_pcr,
    ])

    # Trim warmup
    features = features[30:]
    features = expanding_zscore(features)
    return (features, daily_ohlcv.index[30:]) if return_dates else features


# ============================================================================
# MacroMotion feature builder
# ============================================================================

def build_macro_features(daily_ohlcv, spy_df=None, xle_df=None, return_dates=False):
    """Build MacroMotion features: [x_5d, x_20d, x_60d, rho_spy, rho_xle, sigma_eps].

    Trims 60 warm-up days off the front. Pass date-aligned spy_df/xle_df —
    this builder indexes them positionally against daily_ohlcv.
    """
    close = daily_ohlcv["close"].values.astype(float)
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    n = len(close)

    # Fractional-diff-like momentum at 3 horizons
    x_5 = np.zeros(n)
    x_20 = np.zeros(n)
    x_60 = np.zeros(n)
    for i in range(5, n):
        x_5[i] = log_ret[i - 5:i].sum()
    for i in range(20, n):
        x_20[i] = log_ret[i - 20:i].sum()
    for i in range(60, n):
        x_60[i] = log_ret[i - 60:i].sum()

    # Rolling correlation with SPY
    rho_spy = np.zeros(n)
    rho_xle = np.zeros(n)

    if spy_df is not None and "close" in spy_df.columns:
        spy_close = spy_df["close"].values.astype(float)
        n_spy = min(n, len(spy_close))
        spy_ret = np.diff(np.log(spy_close[:n_spy]), prepend=0)
        # Pad or trim to match
        if len(spy_ret) < n:
            spy_ret = np.pad(spy_ret, (0, n - len(spy_ret)))
        spy_ret = spy_ret[:n]
        for i in range(60, n):
            r1 = log_ret[i - 60:i]
            r2 = spy_ret[i - 60:i]
            if len(r1) == len(r2) and len(r1) > 1:
                c = np.corrcoef(r1, r2)[0, 1]
                rho_spy[i] = c if np.isfinite(c) else 0

    if xle_df is not None and "close" in xle_df.columns:
        xle_close = xle_df["close"].values.astype(float)
        n_xle = min(n, len(xle_close))
        xle_ret = np.diff(np.log(xle_close[:n_xle]), prepend=0)
        if len(xle_ret) < n:
            xle_ret = np.pad(xle_ret, (0, n - len(xle_ret)))
        xle_ret = xle_ret[:n]
        for i in range(60, n):
            r1 = log_ret[i - 60:i]
            r2 = xle_ret[i - 60:i]
            if len(r1) == len(r2) and len(r1) > 1:
                c = np.corrcoef(r1, r2)[0, 1]
                rho_xle[i] = c if np.isfinite(c) else 0

    # Idiosyncratic vol (residual after SPY regression)
    sigma_eps = np.zeros(n)
    if spy_df is not None and "close" in spy_df.columns:
        spy_close = spy_df["close"].values.astype(float)
        n_spy = min(n, len(spy_close))
        spy_ret = np.diff(np.log(spy_close[:n_spy]), prepend=0)
        if len(spy_ret) < n:
            spy_ret = np.pad(spy_ret, (0, n - len(spy_ret)))
        spy_ret = spy_ret[:n]
        for i in range(60, n):
            r1 = log_ret[i - 60:i]
            r2 = spy_ret[i - 60:i]
            if len(r1) == len(r2) and np.var(r2) > 1e-12:
                beta = np.cov(r1, r2)[0, 1] / np.var(r2)
                resid = r1 - beta * r2
                sigma_eps[i] = resid.std()

    features = np.column_stack([x_5, x_20, x_60, rho_spy, rho_xle, sigma_eps])
    features = features[60:]
    features = expanding_zscore(features)
    return (features, daily_ohlcv.index[60:]) if return_dates else features


# ============================================================================
# Hawkes proxy feature builder
# ============================================================================

def build_hawkes_features(trades_df, window_seconds=300, return_timestamps=False):
    """
    Build Hawkes proxy features: [event_count, mean_inter_arrival, burst_ratio].
    Vectorized — no Python loops over 36M rows.

    Windows are fixed wall-clock spans, so the row count depends on calendar
    length rather than on trade activity. return_timestamps gives the close
    time of each window (the instant its features become knowable).
    """
    epoch_abs = to_epoch_seconds(trades_df["ts_event"])
    epoch0 = float(epoch_abs[0])
    epoch = epoch_abs - epoch0  # zero-base
    total_time = epoch[-1]

    bins = np.arange(0, total_time + window_seconds, window_seconds)
    counts, _ = np.histogram(epoch, bins=bins)
    n_windows = len(counts)

    # Assign each trade to its window
    bin_idx = np.searchsorted(bins, epoch, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_windows - 1)

    # Compute inter-arrival times
    ia = np.diff(epoch)
    ia_bin = bin_idx[1:]  # each inter-arrival belongs to the window of the 2nd trade

    # Sum inter-arrivals per window, count inter-arrivals per window
    ia_sum = np.bincount(ia_bin, weights=ia, minlength=n_windows)[:n_windows]
    ia_count = np.bincount(ia_bin, minlength=n_windows)[:n_windows]

    # Mean inter-arrival per window (0 where no inter-arrivals)
    mean_ia = np.divide(ia_sum, ia_count, out=np.zeros(n_windows), where=ia_count > 0)

    # Burst ratio = count / mean_ia
    burst = np.divide(counts.astype(float), mean_ia + 1e-8,
                      out=np.zeros(n_windows), where=counts > 1)

    features = np.column_stack([counts.astype(float), mean_ia, burst])
    # A window's features are only known at its right edge
    window_close_ts = epoch0 + bins[1:n_windows + 1]

    features = expanding_zscore(features)
    return (features, window_close_ts) if return_timestamps else features


# ============================================================================
# EventProximity feature builder
# ============================================================================

def build_event_proximity_features(daily_dates, event_calendar_df, tau=5.0):
    """
    Build event proximity features for each daily date.
    Returns (n_days, n_event_types) matrix of exp(-|days_to_event|/tau).
    """
    from datetime import date as dt_date

    event_types = ["fomc", "cpi", "nfp", "options_expiry"]
    n = len(daily_dates)
    features = np.zeros((n, len(event_types)))

    # Parse event calendar
    events_by_type = {}
    for _, row in event_calendar_df.iterrows():
        etype = row.get("event_type", "unknown").lower()
        edate = pd.to_datetime(row.get("event_date", row.get("date"))).date()
        events_by_type.setdefault(etype, []).append(edate)

    for j, etype in enumerate(event_types):
        edates = events_by_type.get(etype, [])
        if not edates:
            continue
        edates_ord = np.array([d.toordinal() for d in edates])
        for i, d in enumerate(daily_dates):
            if isinstance(d, str):
                d = pd.to_datetime(d).date()
            elif hasattr(d, 'date'):
                d = d.date()
            d_ord = d.toordinal()
            dists = np.abs(edates_ord - d_ord)
            features[i, j] = np.exp(-dists.min() / tau) if len(dists) > 0 else 0

    # Add calendar features: sin/cos DOW, sin/cos month
    dows = np.array([
        (pd.to_datetime(d).dayofweek if isinstance(d, str)
         else d.dayofweek if hasattr(d, 'dayofweek')
         else d.date().weekday() if hasattr(d, 'date') else 0)
        for d in daily_dates
    ], dtype=float)
    months = np.array([
        (pd.to_datetime(d).month if isinstance(d, str)
         else d.month if hasattr(d, 'month')
         else d.date().month if hasattr(d, 'date') else 1)
        for d in daily_dates
    ], dtype=float)

    cal_feats = np.column_stack([
        np.sin(2 * np.pi * dows / 5),
        np.cos(2 * np.pi * dows / 5),
        np.sin(2 * np.pi * months / 12),
        np.cos(2 * np.pi * months / 12),
    ])

    features = np.concatenate([features, cal_feats], axis=1)
    return expanding_zscore(features)


# ============================================================================
# GammaExposure proxy feature builder (from OI + settlement data)
# ============================================================================

def build_gamma_exposure_features(options_df):
    """
    Build daily GEX proxy + put/call OI ratio from raw Databento stats.
    stat_type 6 = open_interest (quantity field), stat_type 3 = settlement price.
    Returns (n_days, 2) array: [net_gex_proxy, pc_oi_ratio].
    """
    oi_df = options_df[options_df["stat_type"] == 6].copy()
    oi_df["date"] = pd.to_datetime(oi_df["ts_event"], utc=True).dt.date
    oi_df["is_put"] = oi_df["symbol"].str.contains(" P")
    oi_df["oi"] = oi_df["quantity"].astype(float)

    # Parse strike from symbol (e.g. "LOZ3 C4800" → 4800)
    oi_df["strike"] = oi_df["symbol"].str.split().str[-1].str[1:].astype(float)

    daily_groups = oi_df.groupby("date")

    dates = sorted(daily_groups.groups.keys())
    features = np.zeros((len(dates), 2))

    for i, d in enumerate(dates):
        day = daily_groups.get_group(d)
        calls = day[~day["is_put"]]
        puts = day[day["is_put"]]

        call_oi = calls["oi"].sum()
        put_oi = puts["oi"].sum()

        # GEX proxy: net call OI - net put OI weighted by inverse distance to ATM
        # (strikes closer to ATM have higher gamma)
        net_gex = call_oi - put_oi

        # Put/call OI ratio
        pc_ratio = put_oi / (call_oi + 1e-8)

        features[i] = [net_gex, pc_ratio]

    return expanding_zscore(features), dates


# ============================================================================
# VannaCharm feature builder (from real Greeks — Anders' computed data)
# ============================================================================

def build_vanna_charm_features_real(greeks_bars_df, trades_df, dollar_threshold=1_000_000,
                                    zscore_window=50, return_timestamps=False):
    """
    Build VannaCharm features in dollar-bar time from pre-computed Greeks.
    Input: greeks_bars_df with columns [bar_close_time, net_vanna, net_charm].

    Features (6D):
      - net_vanna level (rolling z-score)
      - net_charm level (rolling z-score)
      - d_vanna: bar-over-bar change in vanna (hedging flow impulse)
      - d_charm: bar-over-bar change in charm (theta decay acceleration)
      - vanna_momentum: sign(vanna) * |d_vanna| — directional hedging pressure
      - charm_vanna_ratio: charm / (|vanna| + eps) — decay vs hedging balance

    The rate-of-change features capture *when dealers are actively rebalancing*,
    which is what drives price impact. Raw levels just show current exposure.
    """
    # Parse Greeks bar timestamps
    greeks_ts = pd.to_datetime(greeks_bars_df["bar_close_time"], utc=True)
    greeks_ts_unix = greeks_ts.values.astype(np.int64)
    greeks_vanna = greeks_bars_df["net_vanna"].values.astype(float)
    greeks_charm = greeks_bars_df["net_charm"].values.astype(float)

    # Build dollar bars from trades to get bar timestamps
    ts = pd.to_datetime(trades_df["ts_event"], utc=True)
    ts_unix = ts.values.astype(np.int64)
    ts_seconds = to_epoch_seconds(trades_df["ts_event"])
    prices = trades_df["price"].values.astype(float)
    sizes = trades_df["size"].values.astype(float)

    dollar_notional = prices * sizes
    cum_dn = np.cumsum(dollar_notional)
    bar_thresholds = np.arange(dollar_threshold, cum_dn[-1], dollar_threshold)
    bar_boundaries = np.searchsorted(cum_dn, bar_thresholds)

    n_bars = len(bar_boundaries)
    raw_vanna = np.zeros(n_bars)
    raw_charm = np.zeros(n_bars)

    # For each dollar bar, find the most recent Greeks bar (no lookahead)
    for i, boundary in enumerate(bar_boundaries):
        bar_ts = ts_unix[min(boundary, len(ts_unix) - 1)]
        idx = np.searchsorted(greeks_ts_unix, bar_ts, side="right") - 1
        if idx >= 0:
            raw_vanna[i] = greeks_vanna[idx]
            raw_charm[i] = greeks_charm[idx]

    # Rate-of-change features (hedging flow impulse)
    d_vanna = np.diff(raw_vanna, prepend=raw_vanna[0])
    d_charm = np.diff(raw_charm, prepend=raw_charm[0])

    # Directional hedging pressure: sign(vanna) * |d_vanna|
    vanna_momentum = np.sign(raw_vanna) * np.abs(d_vanna)

    # Decay vs hedging balance
    charm_vanna_ratio = raw_charm / (np.abs(raw_vanna) + 1e-8)

    # Stack all 6 features
    features = np.column_stack([
        raw_vanna, raw_charm,
        d_vanna, d_charm,
        vanna_momentum, charm_vanna_ratio,
    ])

    # Rolling z-score normalization (no lookahead)
    normed = np.zeros_like(features)
    for i in range(features.shape[0]):
        if i < zscore_window:
            # Not enough history — use expanding window
            window = features[:i + 1]
        else:
            window = features[i - zscore_window + 1:i + 1]
        mu = window.mean(axis=0)
        sigma = window.std(axis=0) + 1e-8
        normed[i] = (features[i] - mu) / sigma

    if return_timestamps:
        bar_ts = ts_seconds[np.clip(bar_boundaries, 0, len(ts_seconds) - 1)]
        return normed, bar_ts
    return normed


def build_vanna_charm_features(options_df, trades_df, dollar_threshold=1_000_000):
    """
    [LEGACY] Build VannaCharm proxy features from OI changes.
    Kept for reference — use build_vanna_charm_features_real() with real Greeks.
    """
    oi_df = options_df[options_df["stat_type"] == 6].copy()
    oi_df["date"] = pd.to_datetime(oi_df["ts_event"], utc=True).dt.date
    oi_df["is_put"] = oi_df["symbol"].str.contains(" P")
    oi_df["oi"] = oi_df["quantity"].astype(float)

    daily = oi_df.groupby(["date", "is_put"])["oi"].sum().unstack(fill_value=0)
    daily.columns = ["call_oi", "put_oi"]
    daily = daily.sort_index()

    daily["vanna_proxy"] = daily["call_oi"].diff() - daily["put_oi"].diff()
    total_oi = daily["call_oi"] + daily["put_oi"]
    daily["charm_proxy"] = total_oi.diff() / (total_oi.shift(1) + 1e-8)
    daily = daily.dropna()

    ts = pd.to_datetime(trades_df["ts_event"], utc=True)
    prices = trades_df["price"].values.astype(float)
    sizes = trades_df["size"].values.astype(float)
    trade_dates = ts.dt.date.values

    dollar_notional = prices * sizes
    cum_dn = np.cumsum(dollar_notional)
    bar_thresholds = np.arange(dollar_threshold, cum_dn[-1], dollar_threshold)
    bar_boundaries = np.searchsorted(cum_dn, bar_thresholds)

    n_bars = len(bar_boundaries)
    features = np.zeros((n_bars, 2))

    daily_dates = daily.index.tolist()
    vanna_vals = daily["vanna_proxy"].values
    charm_vals = daily["charm_proxy"].values

    for i, boundary in enumerate(bar_boundaries):
        bar_date = trade_dates[min(boundary, len(trade_dates) - 1)]
        idx = np.searchsorted(daily_dates, bar_date)
        idx = min(idx, len(daily_dates) - 1)
        features[i] = [vanna_vals[idx], charm_vals[idx]]

    return expanding_zscore(features)


# ============================================================================
# Main
# ============================================================================

def _assert_on_master(arrays, names, n_master):
    """
    All fast features must already sit on the master schedule.

    The original pipeline aligned kernels with arr[:n] here, which silently
    matched rows from different months. Alignment now happens once, by wall
    clock, in section 3b2 — so anything reaching this point with the wrong
    length is a bug, not something to trim away.
    """
    bad = [(nm, a.shape[0]) for nm, a in zip(names, arrays) if a.shape[0] != n_master]
    if bad:
        raise ValueError(
            f"Fast features are not on the master schedule (expected "
            f"{n_master} rows): {bad}. Do not trim — fix the alignment."
        )


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Pristine-RKHS Walk-Forward Backtest")
    print(f"  CL Futures | Jul 2023 - Dec 2024 | arch={os.environ.get('ARCH_VERSION', 'v1')}")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    dl = DataLoader(os.path.join(_ROOT, "data"))
    dates = dl.trading_dates()
    print(f"  LOB dates: {len(dates)} ({dates[0]} — {dates[-1]})")

    # Load 1-min OHLCV → daily
    ohlcv = dl.ohlcv()
    daily = dl.build_daily_ohlcv()
    print(f"  Daily bars: {len(daily)}")

    # Load external data
    ext = os.path.join(_ROOT, "data", "external")
    vix_df = pd.read_parquet(os.path.join(ext, "vix_daily.parquet"))
    spy_df = pd.read_parquet(os.path.join(ext, "spy_daily.parquet"))
    xle_df = pd.read_parquet(os.path.join(ext, "xle_daily.parquet"))
    pcr_df = pd.read_parquet(os.path.join(ext, "put_call_ratio.parquet"))
    print(f"  External: VIX={len(vix_df)}, SPY={len(spy_df)}, "
          f"XLE={len(xle_df)}, PCR={len(pcr_df)} rows")

    # ── 2. Build LOB features (fast — precomputed) ───────────────────────
    print("\n[2/7] Building LOB features from precomputed data...")
    lob_feat_list = []
    lob_ts_list = []
    lob_dates_used = []
    for d in dates:
        try:
            feat = dl.lob_features(d)
            vp = feat.get("volume_profile")
            bs = feat.get("book_shape")
            di = feat.get("depth_imbalance")
            ts = feat.get("timestamps")
            if vp is not None and bs is not None and di is not None and ts is not None:
                # Concatenate all 3 sub-kernel features per bar
                n_bars = min(vp.shape[0], bs.shape[0], di.shape[0], len(ts))
                combined = np.concatenate([
                    vp[:n_bars], bs[:n_bars], di[:n_bars]
                ], axis=1)
                lob_feat_list.append(combined)
                lob_ts_list.append(np.asarray(ts)[:n_bars])
                lob_dates_used.append(d)
        except Exception:
            continue

    lob_features_all = np.concatenate(lob_feat_list, axis=0)
    # Snapshot counts vary from 14 to a few thousand per day, so row number
    # says nothing about when a row happened — keep the wall clock.
    lob_ts_all = to_epoch_seconds(np.concatenate(lob_ts_list))
    print(f"  LOB feature matrix: {lob_features_all.shape} "
          f"({len(lob_dates_used)} days)")

    # ── 3. Build VPIN + Kyle + Hawkes features (fast — from trades) ─────
    print("\n[3/8] Building VPIN, Kyle's Lambda, and Hawkes features...")
    trades = dl.trades()
    n_trades = len(trades)
    print(f"  Trades loaded: {n_trades:,}")

    # The master dollar-bar schedule is the single definition of "bar i" for
    # the whole pipeline: every fast kernel is resampled onto it and the
    # forward-return target is computed from its closes.
    master = build_master_bar_schedule(trades, dollar_threshold=1_000_000)
    master_ts = master["close_ts"]
    master_px = master["close_price"]
    print(f"  Master dollar-bar schedule: {len(master_ts):,} bars "
          f"({pd.to_datetime(master_ts[0], unit='s', utc=True):%Y-%m-%d} → "
          f"{pd.to_datetime(master_ts[-1], unit='s', utc=True):%Y-%m-%d})")

    vpin_feats, vpin_ts = build_vpin_features(
        trades, dollar_threshold=1_000_000, return_timestamps=True)
    print(f"  VPIN features: {vpin_feats.shape}")

    kyle_feats, kyle_ts = build_kyle_features(trades, return_timestamps=True)
    print(f"  Kyle features: {kyle_feats.shape}")

    hawkes_feats, hawkes_ts = build_hawkes_features(
        trades, window_seconds=300, return_timestamps=True)
    print(f"  Hawkes features: {hawkes_feats.shape}")

    # ── 3b. VannaCharm (real Greeks) + GammaExposure ─────────────────────
    skip_kernels = set(os.environ.get("SKIP_KERNELS", "").split(","))
    greeks_bars_path = os.path.join(_ROOT, "data", "vanna_charm_bars.parquet")
    if os.path.exists(greeks_bars_path) and "VannaCharm" not in skip_kernels:
        greeks_bars_df = pd.read_parquet(greeks_bars_path)
        vc_feats, vc_ts = build_vanna_charm_features_real(
            greeks_bars_df, trades, dollar_threshold=1_000_000,
            return_timestamps=True
        )
        print(f"  VannaCharm features (real Greeks): {vc_feats.shape}")
    else:
        vc_feats = None
        vc_ts = None
        print("  VannaCharm: skipped")

    # GammaExposure — still waiting on gamma data from Anders
    gex_feats = None

    # ── 3b2. Align every fast kernel onto the master schedule ────────────
    # Each fast kernel is sampled on its own clock: LOB per snapshot, VPIN
    # and VannaCharm per dollar bar, Kyle per fixed trade-count bin, Hawkes
    # per 300s wall-clock window. The old code aligned them with arr[:n_fast],
    # which lined up row i of each regardless of when row i happened — LOB
    # row 4,000 was July 2023 while Kyle row 4,000 was late 2024. Resample
    # everything onto one schedule by wall clock instead, using a backward
    # as-of join so nothing looks forward.
    print("\n[3b2] Aligning fast kernels to the master dollar-bar schedule...")
    _named_fast = {
        "LOB": (lob_ts_all, lob_features_all),
        "VPIN": (vpin_ts, vpin_feats),
        "Kyle": (kyle_ts, kyle_feats),
        "Hawkes": (hawkes_ts, hawkes_feats),
    }
    if vc_feats is not None:
        _named_fast["VannaCharm"] = (vc_ts, vc_feats)

    _aligned, _valid = align_fast_features(_named_fast, master_ts)
    _start = first_valid_bar(_valid)

    master_ts = master_ts[_start:]
    master_px = master_px[_start:]
    lob_features_all = _aligned["LOB"][_start:]
    vpin_feats = _aligned["VPIN"][_start:]
    kyle_feats = _aligned["Kyle"][_start:]
    hawkes_feats = _aligned["Hawkes"][_start:]
    if vc_feats is not None:
        vc_feats = _aligned["VannaCharm"][_start:]

    n_master = len(master_ts)
    print(f"  Dropped {_start:,} warm-up bars; {n_master:,} aligned bars "
          f"({pd.to_datetime(master_ts[0], unit='s', utc=True):%Y-%m-%d} → "
          f"{pd.to_datetime(master_ts[-1], unit='s', utc=True):%Y-%m-%d})")

    # ── 3c. v2 Architecture: Consolidate order-flow into single kernel ───
    ARCH_VERSION = os.environ.get("ARCH_VERSION", "v1").lower()
    ORDERFLOW_PCA_DIM = int(os.environ.get("ORDERFLOW_PCA_DIM", "15"))

    orderflow_feats = None  # set in v2 path
    gate_vpin = None        # raw VPIN for position-sizing gate
    gate_hawkes = None      # raw Hawkes burst_ratio for position-sizing gate
    gate_kyle = None        # raw Kyle lambda for position-sizing gate

    if ARCH_VERSION == "v2":
        print(f"\n[3c] v2 Architecture: consolidating order-flow kernels...")

        # Already aligned to the master schedule in 3b2 — verify, never trim
        _assert_on_master(
            [lob_features_all, vpin_feats, kyle_feats, hawkes_feats],
            ["LOB", "VPIN", "Kyle", "Hawkes"], n_master,
        )
        of_n = n_master
        of_lob = lob_features_all
        of_vpin = vpin_feats
        of_kyle = kyle_feats
        of_hawkes = hawkes_feats

        # Save raw features for position-sizing gates before PCA
        gate_vpin = of_vpin[:, 0]       # VPIN level
        gate_hawkes = of_hawkes[:, 2]   # burst_ratio
        gate_kyle = of_kyle[:, 1]       # kyle_lambda_5

        # Concatenate all order-flow features
        orderflow_raw = np.concatenate([of_lob, of_vpin, of_kyle, of_hawkes], axis=1)
        d_raw = orderflow_raw.shape[1]
        print(f"  Raw order-flow features: {orderflow_raw.shape} "
              f"(LOB={of_lob.shape[1]} + VPIN={of_vpin.shape[1]} + "
              f"Kyle={of_kyle.shape[1]} + Hawkes={of_hawkes.shape[1]})")

        # Expanding-window PCA to reduce to ORDERFLOW_PCA_DIM (no lookahead)
        from sklearn.decomposition import PCA
        min_of_pca_window = max(ORDERFLOW_PCA_DIM * 3, 60)
        orderflow_feats = np.zeros((of_n, ORDERFLOW_PCA_DIM))
        for i in range(of_n):
            if i < min_of_pca_window:
                # Not enough history — use first ORDERFLOW_PCA_DIM raw features
                orderflow_feats[i] = orderflow_raw[i, :ORDERFLOW_PCA_DIM]
            else:
                pca = PCA(n_components=ORDERFLOW_PCA_DIM, random_state=_seed(42))
                pca.fit(orderflow_raw[:i])
                orderflow_feats[i] = pca.transform(orderflow_raw[i:i+1])[0]

        # Z-score normalize (expanding window)
        of_normed = np.zeros_like(orderflow_feats)
        for i in range(orderflow_feats.shape[0]):
            window = orderflow_feats[:i + 1]
            mu = window.mean(axis=0)
            sigma = window.std(axis=0) + 1e-8
            of_normed[i] = (orderflow_feats[i] - mu) / sigma
        orderflow_feats = of_normed

        print(f"  OrderFlow PCA: {d_raw}D → {ORDERFLOW_PCA_DIM}D")
        if i >= min_of_pca_window:
            print(f"  PCA explained variance (last fit): "
                  f"{pca.explained_variance_ratio_.sum():.1%}")

    # ── 4. Build slow + event features (daily) ──────────────────────────
    print("\n[4/8] Building slow kernel features (VRP + MacroMotion + Sentiment + EventProximity)...")
    # VIX/SPY/XLE cover 378 US equity sessions; CL trades 468 days over the
    # same span. The builders index them positionally, so join them onto the
    # CL calendar first — otherwise the length check silently fails and VRP
    # falls back to a placeholder while Macro correlates against zero-padding.
    vix_al = reindex_daily(vix_df, daily.index)
    spy_al = reindex_daily(spy_df, daily.index)
    xle_al = reindex_daily(xle_df, daily.index)
    pcr_al = reindex_daily(pcr_df, daily.index)
    print(f"  External data joined to CL calendar: {len(daily)} days")

    vrp_feats, vrp_dates = build_vrp_features(daily, vix_al, pcr_al,
                                              return_dates=True)
    print(f"  VRP features: {vrp_feats.shape}")

    macro_feats, macro_dates = build_macro_features(daily, spy_al, xle_al,
                                                    return_dates=True)
    print(f"  Macro features: {macro_feats.shape}")

    # Sentiment kernel (daily embeddings from GDELT/FinBERT)
    SENT_PCA_DIM = int(os.environ.get("SENT_PCA_DIM", "10"))
    sent_path = os.path.join(_ROOT, "data", "sentiment_daily.parquet")
    if os.path.exists(sent_path) and "Sentiment" not in skip_kernels:
        sent_df = pd.read_parquet(sent_path)
        sent_dates = pd.to_datetime(sent_df["date"])
        sent_feat_cols = [c for c in sent_df.columns if c.startswith("sent_feat_")]
        sent_raw = sent_df[sent_feat_cols].values.astype(float)

        # PCA reduction: 128D embeddings → SENT_PCA_DIM principal components
        # Use expanding-window PCA to avoid lookahead
        from sklearn.decomposition import PCA
        min_pca_window = max(SENT_PCA_DIM * 2, 30)  # need enough samples to fit PCA
        sent_reduced = np.zeros((sent_raw.shape[0], SENT_PCA_DIM))
        for i in range(sent_raw.shape[0]):
            if i < min_pca_window:
                # Not enough data for PCA — use first SENT_PCA_DIM raw features
                sent_reduced[i] = sent_raw[i, :SENT_PCA_DIM]
            else:
                pca = PCA(n_components=SENT_PCA_DIM, random_state=_seed(42))
                pca.fit(sent_raw[:i])  # fit on history only (no lookahead)
                sent_reduced[i] = pca.transform(sent_raw[i:i+1])[0]

        # Z-score normalize (expanding window)
        sent_feats = np.zeros_like(sent_reduced)
        for i in range(sent_reduced.shape[0]):
            window = sent_reduced[:i + 1]
            mu = window.mean(axis=0)
            sigma = window.std(axis=0) + 1e-8
            sent_feats[i] = (sent_reduced[i] - mu) / sigma

        # Also include n_headlines as a volume feature
        n_headlines = sent_df["n_headlines"].values.astype(float)
        hl_normed = np.zeros(len(n_headlines))
        for i in range(len(n_headlines)):
            w = n_headlines[:i + 1]
            hl_normed[i] = (n_headlines[i] - w.mean()) / (w.std() + 1e-8)
        sent_feats = np.column_stack([sent_feats, hl_normed])

        print(f"  Sentiment features: {sent_feats.shape} "
              f"(PCA {len(sent_feat_cols)}D→{SENT_PCA_DIM}D + headlines, "
              f"{sent_df['n_headlines'].mean():.0f} headlines/day avg)")
    else:
        sent_feats = None
        print("  Sentiment: skipped")

    # Event proximity gate features
    event_cal_path = os.path.join(ext, "event_calendar.parquet")
    if os.path.exists(event_cal_path):
        event_cal_df = pd.read_parquet(event_cal_path)
        event_feats = build_event_proximity_features(daily.index, event_cal_df)
        print(f"  Event features: {event_feats.shape}")
    else:
        event_feats = None
        print("  Event calendar not found — skipping EventProximity")

    # ── 4b. Put the slow layer on one common daily axis ──────────────────
    # VRP trims 30 warm-up days, Macro trims 60, and Sentiment carries its
    # own date index (503 rows against CL's 468 days). The old code stacked
    # them with [:n_slow], which lined up VRP day 30 with Macro day 60 with
    # Sentiment day 0, and left the daily grid ending 70 days short of the
    # bars. That short grid is what clipped every OOS bar onto one day and
    # made the forward-return target identically zero. Align by date.
    print("\n[4b] Aligning slow kernels to a common daily axis...")
    _named_slow = {
        "VRP": (to_epoch_seconds(vrp_dates), vrp_feats),
        "Macro": (to_epoch_seconds(macro_dates), macro_feats),
    }
    if sent_feats is not None:
        _named_slow["Sentiment"] = (to_epoch_seconds(sent_dates), sent_feats)

    _daily_ts_full = to_epoch_seconds(daily.index)
    _aligned_slow, _valid_slow = align_fast_features(_named_slow, _daily_ts_full)
    _slow_start = first_valid_bar(_valid_slow)

    vrp_feats = _aligned_slow["VRP"][_slow_start:]
    macro_feats = _aligned_slow["Macro"][_slow_start:]
    if sent_feats is not None:
        sent_feats = _aligned_slow["Sentiment"][_slow_start:]
    if event_feats is not None:
        event_feats = event_feats[_slow_start:]

    daily = daily.iloc[_slow_start:]
    slow_dates = daily.index
    n_slow_axis = len(slow_dates)
    print(f"  Slow axis: {n_slow_axis} days "
          f"({slow_dates[0]:%Y-%m-%d} → {slow_dates[-1]:%Y-%m-%d}), "
          f"dropped {_slow_start} warm-up days")

    # Bars earlier than the slow layer's first day have no daily context and
    # would all clip onto day 0 — drop them from the fast side too.
    _slow_t0 = _daily_ts_full[_slow_start]
    _bar_keep = int(np.searchsorted(master_ts, _slow_t0, side="left"))
    if _bar_keep:
        master_ts = master_ts[_bar_keep:]
        master_px = master_px[_bar_keep:]
        lob_features_all = lob_features_all[_bar_keep:]
        vpin_feats = vpin_feats[_bar_keep:]
        kyle_feats = kyle_feats[_bar_keep:]
        hawkes_feats = hawkes_feats[_bar_keep:]
        if vc_feats is not None:
            vc_feats = vc_feats[_bar_keep:]
        if orderflow_feats is not None:
            orderflow_feats = orderflow_feats[_bar_keep:]
        if gate_vpin is not None:
            gate_vpin = gate_vpin[_bar_keep:]
            gate_hawkes = gate_hawkes[_bar_keep:]
            gate_kyle = gate_kyle[_bar_keep:]
        n_master = len(master_ts)
    print(f"  Fast side trimmed to the slow window: {n_master:,} bars "
          f"({pd.to_datetime(master_ts[0], unit='s', utc=True):%Y-%m-%d} → "
          f"{pd.to_datetime(master_ts[-1], unit='s', utc=True):%Y-%m-%d}), "
          f"dropped {_bar_keep:,} bars")

    # ── 4c. Falsification controls (NULL_MODE) ───────────────────────────
    # A backtest you cannot break is a backtest you cannot trust. These
    # modes destroy the information the strategy claims to use while
    # leaving every other moving part identical. If performance survives,
    # the performance was never coming from the kernels.
    #
    #   noise          every prediction feature becomes iid N(0,1).
    #                  This is the "no kernels" null.
    #   shuffle_time   feature rows are permuted in time. Marginal
    #                  distributions are preserved exactly; only the
    #                  correspondence between features and dates is broken.
    #   shuffle_target the forward-return target is permuted. Nothing can
    #                  predict it, so any surviving P&L is manufactured by
    #                  the harness itself.
    #
    # Position-sizing gates are NOT touched here — they are not prediction
    # kernels. Set POSITION_GATE=0 and VOL_SIZING=0 to strip those too.
    NULL_MODE = os.environ.get("NULL_MODE", "none").lower()
    if NULL_MODE != "none":
        _null_rng = np.random.default_rng(_seed(1234))
        _pred_feats = {
            "lob": lob_features_all, "vpin": vpin_feats, "kyle": kyle_feats,
            "hawkes": hawkes_feats, "vc": vc_feats, "orderflow": orderflow_feats,
            "vrp": vrp_feats, "macro": macro_feats, "sent": sent_feats,
        }
        print(f"\n[4c] NULL MODE: {NULL_MODE} — falsification run, "
              f"results are expected to collapse")

        if NULL_MODE == "noise":
            for k, v in _pred_feats.items():
                if v is not None:
                    _pred_feats[k] = _null_rng.standard_normal(v.shape)
        elif NULL_MODE == "shuffle_time":
            for k, v in _pred_feats.items():
                if v is not None:
                    _pred_feats[k] = v[_null_rng.permutation(v.shape[0])]
        elif NULL_MODE == "shuffle_target":
            pass  # applied to y further down, once it exists
        else:
            raise ValueError(
                f"NULL_MODE={NULL_MODE!r} — expected one of: none, noise, "
                f"shuffle_time, shuffle_target"
            )

        lob_features_all = _pred_feats["lob"]
        vpin_feats = _pred_feats["vpin"]
        kyle_feats = _pred_feats["kyle"]
        hawkes_feats = _pred_feats["hawkes"]
        vc_feats = _pred_feats["vc"]
        orderflow_feats = _pred_feats["orderflow"]
        vrp_feats = _pred_feats["vrp"]
        macro_feats = _pred_feats["macro"]
        sent_feats = _pred_feats["sent"]

    # ── 5. Align resolutions + build combiner ────────────────────────────
    print(f"\n[5/8] Aligning resolutions and building kernel combiner... "
          f"(arch={ARCH_VERSION})")

    if ARCH_VERSION == "v2":
        # ── v2: Fast = [OrderFlow, VannaCharm]; Slow = [Sentiment, Macro, VRP] ──
        fast_arrays = [orderflow_feats]
        fast_names_all = ["OrderFlow"]
        if vc_feats is not None:
            fast_arrays.append(vc_feats)
            fast_names_all.append("VannaCharm")

        _assert_on_master(fast_arrays, fast_names_all, n_master)
        n_fast = n_master
        fast_trimmed = dict(zip(fast_names_all, fast_arrays))

        # Trim gate signals to match
        if gate_vpin is not None:
            gate_vpin = gate_vpin[:n_fast]
            gate_hawkes = gate_hawkes[:n_fast]
            gate_kyle = gate_kyle[:n_fast]

        n_slow = n_slow_axis
        _slow_names = ["VRP", "Macro"] + (["Sentiment"] if sent_feats is not None else [])
        _slow_arrays = [vrp_feats, macro_feats] + \
                       ([sent_feats] if sent_feats is not None else [])
        _assert_on_master(_slow_arrays, _slow_names, n_slow)

        vrp_slow = vrp_feats
        macro_slow = macro_feats
        sent_slow = sent_feats

        print(f"  Fast bars: {n_fast}")
        print(f"  Slow days: {n_slow}")
        print(f"  Fast kernels: {fast_names_all}")

        # Build kernel adapters — v2
        of_kernel = FeatureMatrixKernel(fast_trimmed["OrderFlow"], nu=1.5,
                                         length_scale=1.0, n_rff=512,
                                         name="OrderFlow", seed=_seed(42))

        vc_kernel = None
        if "VannaCharm" in fast_trimmed:
            vc_kernel = FeatureMatrixKernel(fast_trimmed["VannaCharm"], nu=1.5,
                                            length_scale=1.0, n_rff=256,
                                            name="VannaCharm", seed=_seed(48))

        vrp_kernel = FeatureMatrixKernel(vrp_slow, nu=1.5, length_scale=1.0,
                                         n_rff=256, name="VRP", seed=_seed(45))
        macro_kernel = FeatureMatrixKernel(macro_slow, nu=1.5, length_scale=1.0,
                                           n_rff=256, name="Macro", seed=_seed(46))
        sent_kernel = None
        if sent_slow is not None:
            sent_kernel = FeatureMatrixKernel(sent_slow, nu=1.5, length_scale=1.0,
                                               n_rff=512, name="Sentiment", seed=_seed(50))

        # No CKA checks in v2 — order flow is already consolidated
        print("  CKA: skipped (order-flow consolidated in v2)")

        # Fast layer: OrderFlow + VannaCharm
        d_of = fast_trimmed["OrderFlow"].shape[1]
        fast_kernels = [of_kernel]
        fast_slices = [slice(0, d_of)]
        fast_weights = [0.60]
        col_offset = d_of

        if vc_kernel is not None:
            d_vc = fast_trimmed["VannaCharm"].shape[1]
            fast_kernels.append(vc_kernel)
            fast_slices.append(slice(col_offset, col_offset + d_vc))
            fast_weights.append(0.40)
            col_offset += d_vc

        wsum = sum(fast_weights)
        fast_weights = [w / wsum for w in fast_weights]
        print(f"  Fast kernels: {len(fast_kernels)} "
              f"(weights: {[f'{w:.3f}' for w in fast_weights]})")

        fast_layer = CombinedKernel(fast_kernels, fast_slices, fast_weights)

        # Slow layer: VRP + Macro + Sentiment
        d_vrp = vrp_slow.shape[1]
        d_macro = macro_slow.shape[1]
        slow_kernels_list = [vrp_kernel, macro_kernel]
        slow_col_offset = d_vrp + d_macro
        slow_slices = [slice(0, d_vrp), slice(d_vrp, d_vrp + d_macro)]

        if sent_kernel is not None:
            d_sent = sent_slow.shape[1]
            slow_kernels_list.append(sent_kernel)
            slow_slices.append(slice(slow_col_offset, slow_col_offset + d_sent))
            slow_weights_v2 = [0.30, 0.30, 0.40]
            slow_col_offset += d_sent
        else:
            slow_weights_v2 = [0.50, 0.50]

        slow_layer = CombinedKernel(slow_kernels_list, slow_slices, slow_weights_v2)

        # Position-sizing gates from raw order-flow signals
        if gate_vpin is not None:
            print(f"\n  Position-sizing gates (from raw order-flow):")
            # VPIN gate: high VPIN → reduce position (toxic flow)
            # Normalize to [0.5, 1.0]: low VPIN → 1.0 (full size), high VPIN → 0.5
            vpin_pct = (gate_vpin - gate_vpin.min()) / (gate_vpin.max() - gate_vpin.min() + 1e-8)
            gate_vpin_sizing = 1.0 - 0.5 * vpin_pct  # invert: high VPIN → smaller position
            print(f"    VPIN gate: mean={gate_vpin_sizing.mean():.3f} "
                  f"[{gate_vpin_sizing.min():.3f}, {gate_vpin_sizing.max():.3f}]")

            # Hawkes burst gate: high burst → reduce position (unstable microstructure)
            burst_pct = (gate_hawkes - gate_hawkes.min()) / (gate_hawkes.max() - gate_hawkes.min() + 1e-8)
            gate_hawkes_sizing = 1.0 - 0.5 * burst_pct
            print(f"    Hawkes gate: mean={gate_hawkes_sizing.mean():.3f} "
                  f"[{gate_hawkes_sizing.min():.3f}, {gate_hawkes_sizing.max():.3f}]")

            # Combined gate = product of individual gates
            position_size_gate = gate_vpin_sizing * gate_hawkes_sizing
            print(f"    Combined gate: mean={position_size_gate.mean():.3f} "
                  f"[{position_size_gate.min():.3f}, {position_size_gate.max():.3f}]")
        else:
            position_size_gate = None

    else:
        # ── v1: Original architecture (LOB + VPIN + Kyle + Hawkes + ...) ──
        fast_arrays = [lob_features_all, vpin_feats, kyle_feats, hawkes_feats]
        fast_names_all = ["LOB", "VPIN", "Kyle", "Hawkes"]
        if vc_feats is not None:
            fast_arrays.append(vc_feats)
            fast_names_all.append("VannaCharm")
        if gex_feats is not None:
            n_bar_approx = min(a.shape[0] for a in fast_arrays)
            gex_to_bar = np.linspace(0, gex_feats.shape[0] - 1, n_bar_approx).astype(int)
            gex_feats_expanded = gex_feats[gex_to_bar]
            fast_arrays.append(gex_feats_expanded)
            fast_names_all.append("GammaExposure")

        _assert_on_master(fast_arrays, fast_names_all, n_master)
        n_fast = n_master

        print(f"  Fast bars: {n_fast}")

        fast_trimmed = dict(zip(fast_names_all, fast_arrays))

        lob_fast = fast_trimmed["LOB"]
        vpin_fast = fast_trimmed["VPIN"]
        kyle_fast = fast_trimmed["Kyle"]
        hawkes_fast = fast_trimmed["Hawkes"]

        n_slow = n_slow_axis
        _slow_names = ["VRP", "Macro"] + (["Sentiment"] if sent_feats is not None else [])
        _slow_arrays = [vrp_feats, macro_feats] + \
                       ([sent_feats] if sent_feats is not None else [])
        _assert_on_master(_slow_arrays, _slow_names, n_slow)

        vrp_slow = vrp_feats
        macro_slow = macro_feats
        sent_slow = sent_feats

        print(f"  Slow days: {n_slow}")
        print(f"  Fast kernels available: {fast_names_all}")

        # Build kernel adapters
        lob_kernel = FeatureMatrixKernel(lob_fast, nu=1.5, length_scale=1.0,
                                         n_rff=500, name="LOB", seed=_seed(42))
        vpin_kernel = FeatureMatrixKernel(vpin_fast, nu=1.5, length_scale=1.0,
                                          n_rff=256, name="VPIN", seed=_seed(43))
        kyle_kernel = FeatureMatrixKernel(kyle_fast, nu=1.5, length_scale=1.0,
                                          n_rff=256, name="Kyle", seed=_seed(44))
        hawkes_kernel = FeatureMatrixKernel(hawkes_fast, nu=1.5, length_scale=1.0,
                                            n_rff=256, name="Hawkes", seed=_seed(47))

        vc_kernel = None
        if "VannaCharm" in fast_trimmed:
            vc_kernel = FeatureMatrixKernel(fast_trimmed["VannaCharm"], nu=1.5,
                                            length_scale=1.0, n_rff=256,
                                            name="VannaCharm", seed=_seed(48))

        gex_kernel = None
        if "GammaExposure" in fast_trimmed:
            gex_kernel = FeatureMatrixKernel(fast_trimmed["GammaExposure"], nu=1.5,
                                              length_scale=1.0, n_rff=256,
                                              name="GammaExposure", seed=_seed(49))

        vrp_kernel = FeatureMatrixKernel(vrp_slow, nu=1.5, length_scale=1.0,
                                         n_rff=256, name="VRP", seed=_seed(45))
        macro_kernel = FeatureMatrixKernel(macro_slow, nu=1.5, length_scale=1.0,
                                           n_rff=256, name="Macro", seed=_seed(46))
        sent_kernel = None
        if sent_slow is not None:
            sent_kernel = FeatureMatrixKernel(sent_slow, nu=1.5, length_scale=1.0,
                                               n_rff=512, name="Sentiment", seed=_seed(50))

        # CKA redundancy checks
        print("\n  CKA redundancy checks...")
        n_sample = min(500, n_fast)
        rng_cka = np.random.RandomState(42)
        idx = rng_cka.choice(n_fast, n_sample, replace=False)

        phi_kyle = kyle_kernel.feature_map(kyle_fast[idx])
        phi_vpin = vpin_kernel.feature_map(vpin_fast[idx])
        cka_kyle_vpin = centered_kernel_alignment(
            phi_kyle @ phi_kyle.T, phi_vpin @ phi_vpin.T
        )
        print(f"  CKA(Kyle, VPIN) = {cka_kyle_vpin:.4f}", end="")

        phi_hawkes = hawkes_kernel.feature_map(hawkes_fast[idx])
        cka_hawkes_vpin = centered_kernel_alignment(
            phi_hawkes @ phi_hawkes.T, phi_vpin @ phi_vpin.T
        )
        print(f"  CKA(Hawkes, VPIN) = {cka_hawkes_vpin:.4f}", end="")

        use_kyle = cka_kyle_vpin < 0.5 and "Kyle" not in skip_kernels
        use_hawkes = cka_hawkes_vpin < 0.75 and "Hawkes" not in skip_kernels
        print(f"\n  → {'KEEP' if use_kyle else 'DROP'} Kyle's Lambda")
        print(f"  → {'KEEP' if use_hawkes else 'DROP'} Hawkes")

        d_lob = lob_fast.shape[1]
        d_vpin = vpin_fast.shape[1]
        d_kyle = kyle_fast.shape[1]
        d_hawkes = hawkes_fast.shape[1]

        fast_kernels = [lob_kernel, vpin_kernel]
        col_offset = d_lob + d_vpin
        fast_slices = [slice(0, d_lob), slice(d_lob, col_offset)]
        fast_weights = [0.30, 0.15]

        if use_kyle:
            fast_kernels.append(kyle_kernel)
            fast_slices.append(slice(col_offset, col_offset + d_kyle))
            fast_weights.append(0.10)
            col_offset += d_kyle

        if use_hawkes:
            fast_kernels.append(hawkes_kernel)
            fast_slices.append(slice(col_offset, col_offset + d_hawkes))
            fast_weights.append(0.10)
            col_offset += d_hawkes

        if vc_kernel is not None:
            d_vc = fast_trimmed["VannaCharm"].shape[1]
            fast_kernels.append(vc_kernel)
            fast_slices.append(slice(col_offset, col_offset + d_vc))
            fast_weights.append(0.15)
            col_offset += d_vc

        if gex_kernel is not None:
            d_gex = fast_trimmed["GammaExposure"].shape[1]
            fast_kernels.append(gex_kernel)
            fast_slices.append(slice(col_offset, col_offset + d_gex))
            fast_weights.append(0.20)
            col_offset += d_gex

        wsum = sum(fast_weights)
        fast_weights = [w / wsum for w in fast_weights]
        print(f"  Fast kernels: {len(fast_kernels)} "
              f"(weights: {[f'{w:.3f}' for w in fast_weights]})")

        fast_layer = CombinedKernel(fast_kernels, fast_slices, fast_weights)

        d_vrp = vrp_slow.shape[1]
        d_macro = macro_slow.shape[1]
        slow_kernels_list = [vrp_kernel, macro_kernel]
        slow_col_offset = d_vrp + d_macro
        slow_slices = [slice(0, d_vrp), slice(d_vrp, d_vrp + d_macro)]
        slow_weights = [0.33, 0.33] if sent_kernel is not None else [0.5, 0.5]

        if sent_kernel is not None:
            d_sent = sent_slow.shape[1]
            slow_kernels_list.append(sent_kernel)
            slow_slices.append(slice(slow_col_offset, slow_col_offset + d_sent))
            slow_weights.append(0.34)
            slow_col_offset += d_sent

        slow_layer = CombinedKernel(
            slow_kernels_list, slow_slices, slow_weights,
        )

        position_size_gate = None  # v1 doesn't have this

    # Event proximity gate (applied to K_total)
    event_gate_fn = None
    if event_feats is not None:
        # Build a gate: for each bar, look up the daily event proximity score.
        # Gate = mean proximity across event types. High proximity → gate open.
        event_scores = event_feats[:, :4].mean(axis=1)  # mean of 4 event prox
        # Normalize to [0.5, 1.0] — never fully gate off, just attenuate
        e_min, e_max = event_scores.min(), event_scores.max()
        event_scores_norm = 0.5 + 0.5 * (event_scores - e_min) / (e_max - e_min + 1e-8)

        def event_gate_fn(bar_idx):
            # Map bar to daily index by wall clock (aligner is built below)
            daily_idx = int(aligner.get_daily_indices(np.asarray(bar_idx)))
            if daily_idx < len(event_scores_norm):
                return float(event_scores_norm[daily_idx])
            return 1.0

        print(f"  EventProximity gate: active (mean={event_scores_norm.mean():.3f})")

    # Build targets: forward 1-bar log returns
    daily_close = daily["close"].values.astype(float)

    # Map bar indices → daily indices by wall clock.
    #
    # The old code built a synthetic daily grid, np.arange(0, n_fast,
    # bars_per_day)[:n_slow], which spanned only n_slow * bars_per_day bars.
    # Every bar past that point clipped to the final day, so the entire OOS
    # window shared one price and the forward-return target was identically
    # zero. Use the actual bar close times and daily dates instead, and
    # assert the daily grid really covers the bars.
    bar_timestamps = master_ts
    daily_timestamps = to_epoch_seconds(daily.index[:n_slow])

    if daily_timestamps[-1] < bar_timestamps[-1] - 5 * 86400.0:
        raise ValueError(
            f"Daily grid ends "
            f"{pd.to_datetime(daily_timestamps[-1], unit='s', utc=True):%Y-%m-%d} "
            f"but bars run to "
            f"{pd.to_datetime(bar_timestamps[-1], unit='s', utc=True):%Y-%m-%d} — "
            f"trailing bars would clip to the last day."
        )

    aligner = MultiResolutionAligner(bar_timestamps, daily_timestamps)
    _bar_day = aligner.get_daily_indices(np.arange(n_fast))
    print(f"  Bar→day mapping: {_bar_day.min()}..{_bar_day.max()} over "
          f"{n_slow} days, {len(np.unique(_bar_day))} distinct days covered")

    # Momentum gate
    print("\n  Fitting MomentumGate (Arjan)...")
    cl_daily = daily_close[:n_slow]
    mg = MomentumGate(short_window=20, long_window=60, vol_window=20,
                      n_rff=256, random_state=_seed(42))
    lookback = max(mg.long_window, mg.vol_window)
    if len(cl_daily) > lookback + 10:
        log_r = np.diff(np.log(cl_daily))
        M_gate = len(cl_daily) - 1 - lookback
        target_gate = log_r[lookback:lookback + M_gate]
        mg.fit(cl_daily[:lookback + M_gate + 1], target_gate)
        gate_vals = mg.gate_values(cl_daily[:lookback + M_gate + 1])
        print(f"  Gate range: [{gate_vals.min():.4f}, {gate_vals.max():.4f}], "
              f"mean: {gate_vals.mean():.4f}")
    else:
        mg = None
        print("  Skipped (not enough daily data)")

    # Two-level combiner
    combiner = TwoLevelKernelCombiner(
        fast_layer=fast_layer,
        slow_layer=slow_layer,
        aligner=aligner,
        beta=1.0,
        product_dim=500,
        event_gate=event_gate_fn,
        momentum_gate=mg,
        daily_prices=cl_daily if mg else None,
        seed=_seed(42),
    )

    # ── 6. Walk-forward backtest ─────────────────────────────────────────
    print("\n[6/7] Running walk-forward backtest...")

    # Build fast feature matrix as contiguous array for indexing
    if ARCH_VERSION == "v2":
        fast_parts = [fast_trimmed["OrderFlow"]]
        if vc_kernel is not None:
            fast_parts.append(fast_trimmed["VannaCharm"])
    else:
        fast_parts = [lob_fast, vpin_fast]
        if use_kyle:
            fast_parts.append(kyle_fast)
        if use_hawkes:
            fast_parts.append(hawkes_fast)
        if vc_kernel is not None:
            fast_parts.append(fast_trimmed["VannaCharm"])
        if gex_kernel is not None:
            fast_parts.append(fast_trimmed["GammaExposure"])
    fast_combined = np.concatenate(fast_parts, axis=1)

    # Slow feature matrix
    slow_parts = [vrp_slow, macro_slow]
    if sent_slow is not None:
        slow_parts.append(sent_slow)
    slow_combined = np.concatenate(slow_parts, axis=1)

    # Forward targets
    FORWARD_HORIZON = int(os.environ.get("FORWARD_HORIZON", "1"))
    TARGET_MODE = os.environ.get("TARGET_MODE", "returns").lower()
    VOL_WINDOW = int(os.environ.get("VOL_WINDOW", "20"))

    bar_daily_idx = aligner.get_daily_indices(np.arange(n_fast))
    # Forward returns come from the master dollar-bar closes. Mapping daily
    # closes down onto bars made the target ~95% zeros even once the mapping
    # was correct, because many bars share a single day.
    bar_prices = master_px

    if TARGET_MODE == "vol":
        # ── Predict forward realized volatility ──
        # This is what RKHS kernels are actually good at (corr +0.062 vs +0.007 for returns)
        print(f"\n  Target: forward {VOL_WINDOW}-day realized vol (log-scaled)")

        daily_log_ret = np.diff(np.log(cl_daily + 1e-8))
        # Forward realized vol: std of next VOL_WINDOW days of returns
        fwd_vol_daily = np.zeros(len(cl_daily))
        for i in range(len(cl_daily)):
            end = min(i + VOL_WINDOW, len(daily_log_ret))
            if end > i and end - i >= 5:  # need at least 5 days
                fwd_vol_daily[i] = np.std(daily_log_ret[i:end]) * np.sqrt(252)
            else:
                fwd_vol_daily[i] = np.nan

        # Log-scale vol (more normally distributed, better for regression)
        fwd_vol_daily = np.log(fwd_vol_daily + 1e-8)

        # Map to bar resolution
        y = fwd_vol_daily[np.clip(bar_daily_idx, 0, len(fwd_vol_daily) - 1)]

        # Also compute daily returns for direction signal (used in position generation)
        y_returns_bar = np.diff(np.log(bar_prices + 1e-8), prepend=np.log(bar_prices[0]))
        y_returns_bar = np.append(y_returns_bar[1:], 0)

        # And daily momentum for direction
        mom_20 = np.zeros(len(cl_daily))
        for i in range(20, len(cl_daily)):
            mom_20[i] = cl_daily[i] / cl_daily[i - 20] - 1
        bar_momentum = mom_20[np.clip(bar_daily_idx, 0, len(mom_20) - 1)]

        print(f"  Vol target range: [{np.nanmin(y):.3f}, {np.nanmax(y):.3f}] (log annualized)")
        print(f"  Vol target NaN%:  {np.isnan(y).mean():.1%}")

        # Replace NaN with expanding mean (for early bars)
        y_nanmask = np.isnan(y)
        if y_nanmask.any():
            y[y_nanmask] = np.nanmean(y)

    else:
        # ── Predict forward returns (original mode) ──
        print(f"\n  Target: {FORWARD_HORIZON}-day forward returns")
        y_returns_bar = None
        bar_momentum = None

        if FORWARD_HORIZON == 1:
            y = np.diff(np.log(bar_prices + 1e-8), prepend=np.log(bar_prices[0]))
            y = y[1:]  # shift for forward return
            y = np.append(y, 0)  # pad last
        else:
            # Horizon stays in days, but is measured on the bar clock so the
            # target varies bar to bar instead of being a daily step function.
            log_p = np.log(bar_prices + 1e-8)
            j = np.searchsorted(bar_timestamps,
                                bar_timestamps + FORWARD_HORIZON * 86400.0,
                                side="left")
            j = np.clip(j, 0, n_fast - 1)
            y = log_p[j] - log_p

    if NULL_MODE == "shuffle_target":
        y = y[np.random.default_rng(_seed(5678)).permutation(len(y))]
        print("  NULL MODE: target permuted — nothing can predict this")

    print("\n  Target sanity check:")
    check_target_not_degenerate(y, label=f"{TARGET_MODE} target (full sample)")

    engine = PurgedWalkForward(
        n_splits=8,
        embargo_bars=100,
        min_train_bars=max(2000, n_fast // 5),
        decay_half_life=3000.0,
        reg_lambda=1e-3,
    )

    # ── Progress bar ──
    from tqdm import tqdm

    _tqdm_bar = None

    def update_progress(fold_i, n_folds):
        nonlocal _tqdm_bar
        if _tqdm_bar is None:
            _tqdm_bar = tqdm(
                total=n_folds,
                desc="Walk-Forward",
                bar_format="{l_bar}{bar:30}{r_bar}",
                ncols=80,
                unit="fold",
            )
        if fold_i < n_folds:
            _tqdm_bar.update(1)
        else:
            _tqdm_bar.update(n_folds - _tqdm_bar.n)
            _tqdm_bar.close()

    result = engine.run_two_level(
        fast_data=fast_combined,
        slow_data=slow_combined,
        y=y,
        combiner=combiner,
        bar_indices=np.arange(n_fast),
        progress_callback=update_progress,
    )

    if _tqdm_bar and not _tqdm_bar.disable:
        _tqdm_bar.close()

    print(f"\n  Folds completed: {result.n_folds}")
    print(f"  OOS MSE:  {result.oos_mse:.8f}")
    print(f"  OOS R²:   {result.oos_r2:.6f}")

    # ── 6b. MKL weight optimization ──────────────────────────────────────
    print("\n[6b/7] MKL weight optimization...")
    from backtesting.hyperparameter_cv import MKLOptimizer

    n_fast_k = len(fast_kernels)
    n_slow_k = len(slow_kernels_list)

    # Build per-kernel feature maps for MKL
    fast_fmaps = []
    for k, sl in zip(fast_kernels, fast_slices):
        fast_fmaps.append(k.feature_map(fast_combined[:, sl]))

    slow_fmaps = []
    slow_fmaps.append(vrp_kernel.feature_map(vrp_slow))
    slow_fmaps.append(macro_kernel.feature_map(macro_slow))
    if sent_kernel is not None:
        slow_fmaps.append(sent_kernel.feature_map(sent_slow))

    # Align slow to fast resolution by wall clock, not by stretching the
    # daily axis evenly across the bar axis (bars are not uniform in time)
    slow_to_fast_idx = np.clip(bar_daily_idx, 0, n_slow - 1)
    slow_fmaps_aligned = []
    for phi_s in slow_fmaps:
        slow_fmaps_aligned.append(phi_s[slow_to_fast_idx])

    mkl = MKLOptimizer(
        n_fast_kernels=n_fast_k,
        n_slow_kernels=n_slow_k,
        l2_penalty=0.01,
        lr=0.01,
        max_iter=200,
    )

    mkl_result = mkl.optimize(
        fast_feature_maps=fast_fmaps,
        y=y,
        reg_lambda=1e-3,
        slow_feature_maps=slow_fmaps_aligned,
        verbose=True,
    )

    print(f"\n  Optimized fast weights (α): {mkl_result['alpha_fast']}")
    print(f"  Optimized slow weights (α): {mkl_result['alpha_slow']}")
    print(f"  Optimized β:                {mkl_result['beta']:.4f}")
    print(f"  Final loss:                 {mkl_result['loss_history'][-1]:.8f}")

    # Apply optimized weights to combiner and re-run
    print("\n  Re-running backtest with optimized MKL weights...")
    opt_fast_weights = list(mkl_result['alpha_fast'])
    opt_slow_weights = list(mkl_result['alpha_slow'])

    fast_layer_opt = CombinedKernel(fast_kernels, fast_slices, opt_fast_weights)
    slow_layer_opt = CombinedKernel(
        slow_kernels_list, slow_slices, opt_slow_weights,
    )
    combiner_opt = TwoLevelKernelCombiner(
        fast_layer=fast_layer_opt,
        slow_layer=slow_layer_opt,
        aligner=aligner,
        beta=mkl_result['beta'],
        product_dim=500,
        event_gate=event_gate_fn,
        momentum_gate=mg,
        daily_prices=cl_daily if mg else None,
        seed=_seed(42),
    )

    _tqdm_bar = None
    result = engine.run_two_level(
        fast_data=fast_combined,
        slow_data=slow_combined,
        y=y,
        combiner=combiner_opt,
        bar_indices=np.arange(n_fast),
        progress_callback=update_progress,
    )
    if _tqdm_bar and not _tqdm_bar.disable:
        _tqdm_bar.close()

    print(f"\n  Optimized OOS MSE:  {result.oos_mse:.8f}")
    print(f"  Optimized OOS R²:   {result.oos_r2:.6f}")

    # ── 7. Two-stage signal extraction with per-kernel hyperparameter tuning
    print("\n[7/8] Two-stage signal: per-kernel KRR (tuned) → elastic-net combiner...")

    from sklearn.linear_model import ElasticNetCV, HuberRegressor
    from backtesting.hyperparameter_cv import (
        inner_cv_grid_search_extended, ExtendedCVResult, PurgedKFold,
    )

    COMBINER_LOSS = os.environ.get("COMBINER_LOSS", "mse").lower()  # mse | huber

    TUNE_KERNELS = os.environ.get("TUNE_KERNELS", "1") == "1"

    # Hyperparameter grid
    NU_GRID = [0.5, 1.5, 2.5]
    LS_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
    LAM_GRID = [1e-4, 1e-3, 1e-2]

    def neg_mse(y_true, y_pred):
        return -np.mean((y_true - y_pred) ** 2)

    # Stage 1: Each kernel independently predicts OOS returns
    all_kernel_objs = list(fast_kernels) + slow_kernels_list
    all_kernel_data = []

    # Fast kernels get their slice of fast_combined
    for k, sl in zip(fast_kernels, fast_slices):
        all_kernel_data.append(("fast", fast_combined[:, sl], k))

    # Slow kernels get slow_combined slices, aligned to bar resolution
    all_kernel_data.append(("slow", slow_combined[:, :d_vrp], vrp_kernel))
    all_kernel_data.append(("slow", slow_combined[:, d_vrp:d_vrp + d_macro], macro_kernel))
    if sent_kernel is not None:
        all_kernel_data.append(("slow", slow_combined[:, d_vrp + d_macro:d_vrp + d_macro + d_sent],
                                sent_kernel))

    n_kernels = len(all_kernel_data)
    kernel_names_list = [k.name for _, _, k in all_kernel_data]
    print(f"  Fitting {n_kernels} kernels independently: {kernel_names_list}")
    if TUNE_KERNELS:
        print(f"  Hyperparameter tuning: ν={NU_GRID}, ℓ={LS_GRID}, λ={LAM_GRID}")
        print(f"  Grid size: {len(NU_GRID) * len(LS_GRID) * len(LAM_GRID)} combos per kernel per fold")
    else:
        print(f"  Hyperparameter tuning: DISABLED (fixed ν=1.5, ℓ=1.0, λ=1e-3)")

    # Collect per-kernel OOS predictions using same fold structure
    splits = engine.split_indices(n_fast)

    # Every fold's test window must carry a live target. A silently
    # degenerate OOS target is what produced the original Sharpe 1.56.
    print("\n  Per-fold OOS target check:")
    for _fi, (_tr, _te) in enumerate(splits):
        check_target_not_degenerate(y, _te, label=f"fold {_fi + 1} OOS")
    per_kernel_oos = {i: [] for i in range(n_kernels)}
    two_stage_y_true = []
    two_stage_oos_idx = []
    best_params_per_kernel = {i: [] for i in range(n_kernels)}

    from tqdm import tqdm as tqdm2
    for fold_i, (train_idx, test_idx) in enumerate(tqdm2(splits, desc="Stage-1 folds", ncols=80)):
        y_train = y[train_idx]
        y_test = y[test_idx]

        weights = exponential_decay_weights(len(train_idx), engine.decay_half_life)

        for ki, (resolution, data, kernel) in enumerate(all_kernel_data):
            if resolution == "fast":
                d_train = data[train_idx]
                d_test = data[test_idx]
            else:
                slow_to_fast = np.clip(bar_daily_idx, 0, data.shape[0] - 1)
                d_train = data[slow_to_fast[train_idx]]
                d_test = data[slow_to_fast[test_idx]]

            # Per-kernel hyperparameter tuning via inner CV
            if TUNE_KERNELS:
                inner_cv = PurgedKFold(n_folds=3, embargo=50)
                cv_result = inner_cv_grid_search_extended(
                    kernel=kernel,
                    data=d_train,
                    y=y_train,
                    nus=NU_GRID,
                    length_scales=LS_GRID,
                    reg_lambdas=LAM_GRID,
                    cv=inner_cv,
                    metric_fn=neg_mse,
                    sample_weights=weights,
                )

                # Apply best hyperparameters
                kernel.nu = cv_result.best_nu
                kernel.length_scale = cv_result.best_length_scale
                kernel._init_rff()
                best_lam = cv_result.best_reg_lambda

                best_params_per_kernel[ki].append({
                    "fold": fold_i,
                    "nu": cv_result.best_nu,
                    "length_scale": cv_result.best_length_scale,
                    "reg_lambda": cv_result.best_reg_lambda,
                    "score": cv_result.best_score,
                })
            else:
                best_lam = engine.reg_lambda

            Phi_train = kernel.feature_map(d_train)
            Phi_test = kernel.feature_map(d_test)
            D = Phi_train.shape[1]

            # Weighted KRR
            W = np.diag(np.sqrt(weights))
            Phi_w = W @ Phi_train
            y_w = W @ y_train

            w_k = np.linalg.solve(
                Phi_w.T @ Phi_w + best_lam * np.eye(D),
                Phi_w.T @ y_w,
            )
            pred_k = Phi_test @ w_k
            per_kernel_oos[ki].append(pred_k)

        two_stage_y_true.append(y_test)
        two_stage_oos_idx.append(test_idx)

    # Print best hyperparameters per kernel
    if TUNE_KERNELS:
        print(f"\n  Best hyperparameters per kernel (across folds):")
        for ki in range(n_kernels):
            params = best_params_per_kernel[ki]
            if params:
                avg_nu = np.mean([p["nu"] for p in params])
                avg_ls = np.mean([p["length_scale"] for p in params])
                avg_lam = np.mean([p["reg_lambda"] for p in params])
                # Most common nu and ls (mode)
                from collections import Counter
                nu_mode = Counter([p["nu"] for p in params]).most_common(1)[0][0]
                ls_mode = Counter([p["length_scale"] for p in params]).most_common(1)[0][0]
                print(f"    {kernel_names_list[ki]:>12s}: ν={nu_mode:.1f} (avg {avg_nu:.2f}), "
                      f"ℓ={ls_mode:.1f} (avg {avg_ls:.2f}), λ={avg_lam:.1e}")

    # Concatenate per-kernel predictions into feature matrix
    kernel_features_oos = np.column_stack([
        np.concatenate(per_kernel_oos[ki]) for ki in range(n_kernels)
    ])
    y_true_oos = np.concatenate(two_stage_y_true)
    oos_indices = np.concatenate(two_stage_oos_idx)

    print(f"  Stage-1 kernel feature matrix: {kernel_features_oos.shape}")

    # Check per-kernel correlations with actual
    print(f"\n  Per-kernel OOS correlations with actual returns:")
    for ki in range(n_kernels):
        corr = np.corrcoef(kernel_features_oos[:, ki], y_true_oos)[0, 1]
        print(f"    {kernel_names_list[ki]:>12s}: {corr:+.4f}")

    # Stage 2: Elastic-net with walk-forward on kernel features
    print(f"\n  Stage 2: {COMBINER_LOSS.upper()} combiner walk-forward on {n_kernels} kernel features...")

    # Walk-forward the combiner with expanding window
    n_oos = len(y_true_oos)
    stage2_pred = np.zeros(n_oos)
    stage2_splits = []

    # Split the OOS period into 5 sub-folds for the combiner
    n_stage2_folds = 5
    fold_size = n_oos // (n_stage2_folds + 1)

    POSITIVE_WEIGHTS = os.environ.get("POSITIVE_WEIGHTS", "1") == "1"

    # FIXED_WEIGHTS: bypass ElasticNet with fixed kernel weights
    # Format: comma-separated floats matching kernel order, e.g. "0,0,0.01,0,-0.024"
    FIXED_WEIGHTS_STR = os.environ.get("FIXED_WEIGHTS", "")
    if FIXED_WEIGHTS_STR:
        fixed_w = np.array([float(x) for x in FIXED_WEIGHTS_STR.split(",")])
        assert len(fixed_w) == n_kernels, (
            f"FIXED_WEIGHTS has {len(fixed_w)} values but {n_kernels} kernels: {kernel_names_list}"
        )
        print(f"  Using FIXED kernel weights (bypassing ElasticNet):")
        print(f"    {dict(zip(kernel_names_list, fixed_w))}")
        stage2_pred = kernel_features_oos @ fixed_w
        valid_mask = np.ones(n_oos, dtype=bool)

        # Create a fake enet object for downstream reporting
        class _FakeEnet:
            coef_ = fixed_w
            l1_ratio_ = 0.0
            alpha_ = 0.0
        enet = _FakeEnet()

        stage2_valid_pred = stage2_pred
        stage2_valid_true = y_true_oos
        stage2_valid_idx = oos_indices

        corr_s2 = np.corrcoef(stage2_valid_pred, stage2_valid_true)[0, 1]
        ss_res = np.sum((stage2_valid_true - stage2_valid_pred) ** 2)
        ss_tot = np.sum((stage2_valid_true - stage2_valid_true.mean()) ** 2)
        r2_s2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"    OOS correlation: {corr_s2:+.4f}")
        print(f"    OOS R²:          {r2_s2:.4f}")
        print(f"\n  Using fixed-weight predictions for signal generation...")
        result_pred_for_signal = stage2_valid_pred
        result_true_for_signal = stage2_valid_true
        result_idx_for_signal = stage2_valid_idx
        # Skip the ElasticNet walk-forward below
        n_stage2_folds = 0

    for s2_fold in range(n_stage2_folds):
        s2_train_end = (s2_fold + 1) * fold_size
        s2_test_start = s2_train_end
        s2_test_end = min(s2_test_start + fold_size, n_oos)
        if s2_test_end <= s2_test_start:
            break

        X_s2_train = kernel_features_oos[:s2_train_end]
        y_s2_train = y_true_oos[:s2_train_end]
        X_s2_test = kernel_features_oos[s2_test_start:s2_test_end]

        if COMBINER_LOSS == "huber":
            # ── Huber loss: robust to outlier returns ──
            # Down-weights the 1-5% of bars that drive 62-99% of MSE.
            # epsilon controls the transition from L2→L1 (lower = more robust)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_s2_train)
            X_test_scaled = scaler.transform(X_s2_test)

            huber = HuberRegressor(
                epsilon=1.35,     # standard: switch to L1 at 1.35σ
                alpha=0.001,      # L2 regularization
                max_iter=5000,
            )
            huber.fit(X_scaled, y_s2_train)
            stage2_pred[s2_test_start:s2_test_end] = huber.predict(X_test_scaled)

            if s2_fold == n_stage2_folds - 1:
                # Store last-fold combiner for diagnostics
                enet = huber  # alias for downstream weight reporting
                enet.coef_ = huber.coef_
                enet.l1_ratio_ = 0.0  # Huber uses pure L2
                enet.alpha_ = huber.alpha
        elif COMBINER_LOSS == "ridge":
            # ── Pure Ridge (L2 only) — no sparsity, forces non-zero weights ──
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(
                alphas=np.logspace(-6, -1, 50),
                cv=3,
            )
            ridge.fit(X_s2_train, y_s2_train)
            stage2_pred[s2_test_start:s2_test_end] = ridge.predict(X_s2_test)

            if s2_fold == n_stage2_folds - 1:
                enet = ridge
                enet.coef_ = ridge.coef_
                enet.l1_ratio_ = 0.0
                enet.alpha_ = ridge.alpha_
        else:
            # ── Standard ElasticNet with MSE loss ──
            L1_RATIOS = os.environ.get("COMBINER_L1", "0.1,0.5,0.7,0.9,0.95")
            l1_list = [float(x) for x in L1_RATIOS.split(",")]
            enet = ElasticNetCV(
                l1_ratio=l1_list,
                n_alphas=50,
                cv=3,
                max_iter=5000,
                positive=POSITIVE_WEIGHTS,
                random_state=_seed(42),
            )
            enet.fit(X_s2_train, y_s2_train)
            stage2_pred[s2_test_start:s2_test_end] = enet.predict(X_s2_test)

        stage2_splits.append((s2_train_end, s2_test_start, s2_test_end))

    # Stage-2 diagnostics — these say whether the kernels predict anything,
    # independently of what the position P&L happens to do.
    _stage2_corr, _stage2_r2 = float("nan"), float("nan")
    _stage2_weights = {}

    # Trim to the portion with stage-2 predictions
    if not FIXED_WEIGHTS_STR:
        valid_mask = stage2_pred != 0
        if valid_mask.sum() == 0:
            print("  WARNING: No valid stage-2 predictions")
        else:
            stage2_valid_pred = stage2_pred[valid_mask]
            stage2_valid_true = y_true_oos[valid_mask]
            stage2_valid_idx = oos_indices[valid_mask]

            corr_s2 = np.corrcoef(stage2_valid_pred, stage2_valid_true)[0, 1]
            ss_res = np.sum((stage2_valid_true - stage2_valid_pred) ** 2)
            ss_tot = np.sum((stage2_valid_true - stage2_valid_true.mean()) ** 2)
            r2_s2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            _stage2_corr, _stage2_r2 = float(corr_s2), float(r2_s2)
            _stage2_weights = dict(zip(kernel_names_list,
                                       [float(c) for c in enet.coef_]))

            print(f"\n  Stage-2 ElasticNet results:")
            print(f"    OOS correlation: {corr_s2:+.4f}")
            print(f"    OOS R²:          {r2_s2:.4f}")
            print(f"    Best l1_ratio:   {enet.l1_ratio_:.2f}")
            print(f"    Best alpha:      {enet.alpha_:.6f}")
            print(f"    Kernel weights:  {dict(zip(kernel_names_list, enet.coef_))}")

            # Use stage-2 predictions for signal generation
            print(f"\n  Using two-stage predictions for signal generation...")
            result_pred_for_signal = stage2_valid_pred
            result_true_for_signal = stage2_valid_true
            stage2_valid_idx = stage2_valid_idx

    # Override result for downstream signal generation
    if 'result_pred_for_signal' in dir():
        result.y_pred_oos = result_pred_for_signal
        result.y_true_oos = result_true_for_signal
        result.oos_indices = stage2_valid_idx

    # ── 8. Signal generation + performance metrics ───────────────────────
    SIGNAL_MODE = os.environ.get("SIGNAL_MODE", "ou").lower()
    ADAPTIVE_Q = os.environ.get("ADAPTIVE_Q", "0") == "1"
    print(f"\n[8/8] Signal generation ({SIGNAL_MODE.upper()} mode)...")

    pred = result.y_pred_oos

    if SIGNAL_MODE == "kalman":
        # Kalman filter: adaptive signal extraction
        # Q can be set via env var: KALMAN_Q=aggressive|conservative|auto|adaptive|<float>
        q_setting = os.environ.get("KALMAN_Q", "auto")
        pred_var = np.nanvar(pred)

        if q_setting == "aggressive":
            Q_val = pred_var          # 10x default — trust observations more
        elif q_setting == "conservative":
            Q_val = 0.01 * pred_var   # 10x less — smooth harder
        elif q_setting == "auto":
            Q_val = None              # auto-calibrate (0.1 * var)
        else:
            Q_val = float(q_setting)  # custom value

        # ── Adaptive Q: 2-regime based on kernel prediction agreement ──
        Q_array = None
        if ADAPTIVE_Q:
            print(f"\n  Building adaptive Q from kernel agreement dispersion...")
            # Compute per-bar variance across kernel predictions
            # kernel_features_oos columns = per-kernel OOS predictions
            kernel_var = np.var(kernel_features_oos, axis=1)  # shape (n_oos,)
            # Align to pred length (stage2 valid subset)
            if len(kernel_var) > len(pred):
                kernel_var_aligned = kernel_var[valid_mask] if 'valid_mask' in dir() else kernel_var[:len(pred)]
            elif len(kernel_var) < len(pred):
                kernel_var_aligned = np.pad(kernel_var, (0, len(pred) - len(kernel_var)),
                                            mode='edge')
            else:
                kernel_var_aligned = kernel_var

            # 2-regime: use median as threshold
            var_median = np.median(kernel_var_aligned)

            # Q_high = aggressive (kernels agree, low variance → update fast)
            # Q_low  = conservative (kernels disagree, high variance → smooth)
            Q_high = pred_var          # same as aggressive
            Q_low = 0.1 * pred_var     # same as auto-tuned

            Q_array = np.where(
                kernel_var_aligned < var_median,
                Q_high,    # low dispersion → high conviction → fast updates
                Q_low,     # high dispersion → low conviction → conservative
            )

            print(f"    Q_high (agreement):     {Q_high:.2e}")
            print(f"    Q_low  (disagreement):  {Q_low:.2e}")
            print(f"    Variance median thresh: {var_median:.2e}")
            print(f"    High-conviction bars:   {(kernel_var_aligned < var_median).sum()} "
                  f"({(kernel_var_aligned < var_median).mean():.1%})")

            # Override Q_val — the array takes precedence in the filter
            Q_val = None

        kalman = KalmanSignalGenerator(
            Q=Q_val,
            R_init=0.1 if q_setting == "aggressive" or ADAPTIVE_Q else 1.0,
            adapt_window=30 if q_setting == "aggressive" or ADAPTIVE_Q else 50,
            zscore_window=100 if q_setting == "aggressive" or ADAPTIVE_Q else 200,
            Q_array=Q_array,
        )
        s_scores, filtered, gains, uncertainties = kalman.s_scores(pred)
        print(f"\n  Kalman Filter Parameters:")
        if ADAPTIVE_Q:
            print(f"    Mode:               ADAPTIVE 2-regime")
            print(f"    Q range:            [{Q_array.min():.2e}, {Q_array.max():.2e}]")
        else:
            print(f"    Q (process noise):      {kalman.Q or 0.1 * np.nanvar(pred):.2e}")
        print(f"    R_init (obs noise):     {kalman.R_init:.2e}")
        print(f"    Mean Kalman gain:       {np.mean(gains):.4f}")
        print(f"    Final uncertainty:       {uncertainties[-1]:.2e}")
        print(f"    Filtered signal range:  [{filtered.min():.6f}, {filtered.max():.6f}]")

        # Also estimate OU on filtered signal for diagnostics
        ou = estimate_ou_params(filtered)
        print(f"\n  OU on filtered signal:")
        print(f"    kappa: {ou.kappa:.4f}, half-life: {ou.half_life:.1f} bars")
    else:
        # Original OU + multi-horizon approach
        ou = estimate_ou_params(pred)
        print(f"\n  OU Parameters:")
        print(f"    kappa (mean-reversion): {ou.kappa:.4f}")
        print(f"    mu (long-run mean):     {ou.mu:.6f}")
        print(f"    sigma:                  {ou.sigma:.6f}")
        print(f"    half-life:              {ou.half_life:.1f} bars")

        print(f"\n  Multi-horizon signal aggregation (h=10,30,60)...")
        s_scores = multi_horizon_s_scores(
            pred,
            horizons=(10, 30, 60),
            weights=(0.2, 0.5, 0.3),
            roll_window=min(100, len(pred) // 3),
        )

    print(f"  s-score range: [{np.nanmin(s_scores):.3f}, {np.nanmax(s_scores):.3f}]")
    print(f"  s-score NaN%:  {np.isnan(s_scores).mean():.1%}")

    # Build regime gate for adaptive thresholds (improvement #2)
    regime_gate = None
    if mg is not None and len(cl_daily) > lookback + 10:
        # Align gate values from daily resolution to OOS bar resolution
        raw_gate = mg.gate_values(cl_daily[:lookback + M_gate + 1])
        # Expand to bar resolution
        n_oos = len(pred)
        gate_daily_idx = np.linspace(0, len(raw_gate) - 1, n_oos).astype(int)
        regime_gate = raw_gate[gate_daily_idx]
        print(f"  Regime gate: mean={regime_gate.mean():.3f} "
              f"[{regime_gate.min():.3f}, {regime_gate.max():.3f}]")
        print(f"  Trending bars: {(regime_gate > 0.5).sum()} "
              f"({(regime_gate > 0.5).mean():.1%})")
        print(f"  Choppy bars:   {(regime_gate <= 0.5).sum()} "
              f"({(regime_gate <= 0.5).mean():.1%})")

    # ── Warm-up filter: stay flat for first WARMUP_DAYS ──
    WARMUP_DAYS = int(os.environ.get("WARMUP_DAYS", "120"))

    if TARGET_MODE == "vol":
        # ── Vol regime trading ──────────────────────────────────────────
        # s-score now measures deviation from "normal" vol:
        #   s < 0 → vol below normal (compressed) → favorable for trending
        #   s > 0 → vol above normal (expanded)   → danger, stay flat or fade
        #
        # Direction comes from momentum, not from kernel predictions.
        # Position size scales inversely with predicted vol.
        print(f"\n  Vol-regime position generation:")

        # Align momentum and returns to OOS length
        n_oos = len(s_scores)
        if bar_momentum is not None:
            mom_oos = bar_momentum[:n_oos] if len(bar_momentum) >= n_oos else \
                np.pad(bar_momentum, (0, n_oos - len(bar_momentum)), mode='edge')
        else:
            mom_oos = np.zeros(n_oos)

        # Regime classification from vol s-score
        vol_s = np.nan_to_num(s_scores, nan=0.0)
        VOL_ENTRY_THRESH = float(os.environ.get("VOL_ENTRY_THRESH", "-0.5"))
        VOL_EXIT_THRESH = float(os.environ.get("VOL_EXIT_THRESH", "0.5"))
        VOL_DIR_MODE = os.environ.get("VOL_DIR_MODE", "momentum").lower()

        # Direction signal
        if VOL_DIR_MODE == "dvol" and SIGNAL_MODE == "kalman":
            # Use derivative of filtered vol prediction as direction
            # Falling vol → long (compression = trending up for CL)
            # Rising vol → short (expansion = trending down / choppy)
            dvol = np.diff(filtered, prepend=filtered[0])
            # Smooth with a short EMA to avoid noise
            dvol_smooth = np.zeros_like(dvol)
            alpha_ema = 2.0 / (11)  # 10-bar EMA
            dvol_smooth[0] = dvol[0]
            for i in range(1, len(dvol)):
                dvol_smooth[i] = alpha_ema * dvol[i] + (1 - alpha_ema) * dvol_smooth[i - 1]
            dir_signal = -np.sign(dvol_smooth)  # negative dvol = long
            print(f"  Direction: d(vol) derivative — "
                  f"long={( dir_signal > 0).sum()}, short={(dir_signal < 0).sum()}")
        else:
            # Use 20-day momentum for direction
            dir_signal = np.sign(mom_oos)
            print(f"  Direction: 20-day momentum — "
                  f"long={(dir_signal > 0).sum()}, short={(dir_signal < 0).sum()}")

        positions = np.zeros(n_oos)
        in_position = False
        for i in range(n_oos):
            if not in_position:
                # Enter when vol is compressed (below normal)
                if vol_s[i] < VOL_ENTRY_THRESH:
                    positions[i] = dir_signal[i]
                    if positions[i] != 0:
                        in_position = True
            else:
                # Stay in position until vol expands (above normal)
                if vol_s[i] > VOL_EXIT_THRESH:
                    positions[i] = 0.0
                    in_position = False
                else:
                    # Update direction while in position
                    positions[i] = dir_signal[i] if dir_signal[i] != 0 else positions[i - 1]

        # Scale position size by inverse vol (vol targeting)
        # Higher predicted vol → smaller position
        if SIGNAL_MODE == "kalman":
            # Use filtered vol prediction for sizing
            vol_pred = np.exp(filtered)  # un-log the filtered vol
            vol_median = np.median(vol_pred[vol_pred > 0])
            vol_scale = np.clip(vol_median / (vol_pred + 1e-8), 0.25, 2.0)
            positions = positions * vol_scale
            print(f"  Vol-targeting scale: median vol={vol_median:.4f}, "
                  f"scale range=[{vol_scale.min():.2f}, {vol_scale.max():.2f}]")

        low_vol = (vol_s < VOL_ENTRY_THRESH).sum()
        high_vol = (vol_s > VOL_EXIT_THRESH).sum()
        neutral = n_oos - low_vol - high_vol
        print(f"  Vol regimes: compressed={low_vol} ({low_vol/n_oos:.1%}), "
              f"expanded={high_vol} ({high_vol/n_oos:.1%}), "
              f"neutral={neutral} ({neutral/n_oos:.1%})")
        print(f"  Entry thresh: s < {VOL_ENTRY_THRESH}, Exit thresh: s > {VOL_EXIT_THRESH}")

    else:
        # ── Original return-based position generation ──
        _entry = float(os.environ.get("ENTRY_THRESHOLD", "0.75"))
        _exit = float(os.environ.get("EXIT_THRESHOLD", "0.25"))
        _entry_trend = float(os.environ.get("ENTRY_TRENDING", "0.50"))
        _entry_chop = float(os.environ.get("ENTRY_CHOPPY", "1.50"))
        _exit_trend = float(os.environ.get("EXIT_TRENDING", "0.15"))
        _exit_chop = float(os.environ.get("EXIT_CHOPPY", "0.50"))
        print(f"  Thresholds: entry={_entry}, exit={_exit}")
        print(f"  Trending: entry={_entry_trend}, exit={_exit_trend}")
        print(f"  Choppy:   entry={_entry_chop}, exit={_exit_chop}")
        positions = generate_positions(
            s_scores,
            entry_threshold=_entry,
            exit_threshold=_exit,
            gate_values=regime_gate,
            entry_trending=_entry_trend,
            entry_choppy=_entry_chop,
            exit_trending=_exit_trend,
            exit_choppy=_exit_chop,
            gate_threshold=0.5,
        )

    # Apply warm-up: zero out positions for first WARMUP_DAYS
    if WARMUP_DAYS > 0:
        # Count real days on the OOS bars, not an average bars-per-day ratio
        _wu_bars = np.asarray(result.oos_indices)[:len(positions)]
        _wu_days = np.clip(aligner.get_daily_indices(_wu_bars), 0, n_slow - 1)
        _day0 = int(_wu_days[0]) if len(_wu_days) else 0
        warmup_bars = int(np.searchsorted(_wu_days, _day0 + WARMUP_DAYS, side="left"))
        warmup_bars = min(warmup_bars, len(positions))
        positions[:warmup_bars] = 0
        print(f"  Warm-up filter: flat for first {WARMUP_DAYS} days "
              f"({warmup_bars} bars)")

    # ── Hybrid: vol-prediction position sizing ──────────────────────────
    # Use kernel features to predict realized vol, scale positions inversely.
    # This leverages what RKHS is good at (vol prediction) while keeping
    # return-prediction Kalman for entry/exit timing.
    VOL_SIZING = os.environ.get("VOL_SIZING", "0") == "1"
    if VOL_SIZING and TARGET_MODE != "vol":
        print(f"\n  Hybrid vol-sizing: computing realized vol target...")

        # Compute forward realized vol (same as vol target mode)
        _vol_window = int(os.environ.get("VOL_WINDOW", "20"))
        daily_log_ret = np.diff(np.log(cl_daily + 1e-8))
        fwd_vol_daily = np.zeros(len(cl_daily))
        for i in range(len(cl_daily)):
            end = min(i + _vol_window, len(daily_log_ret))
            if end > i and end - i >= 5:
                fwd_vol_daily[i] = np.std(daily_log_ret[i:end]) * np.sqrt(252)
            else:
                fwd_vol_daily[i] = np.nan
        fwd_vol_daily = np.log(fwd_vol_daily + 1e-8)

        # Map to bar resolution and align to OOS indices
        fwd_vol_bar = fwd_vol_daily[np.clip(bar_daily_idx, 0, len(fwd_vol_daily) - 1)]
        oos_idx = result.oos_indices if hasattr(result, 'oos_indices') else \
            np.arange(len(fwd_vol_bar) - len(positions), len(fwd_vol_bar))
        fwd_vol_oos = fwd_vol_bar[oos_idx[:len(positions)]] if len(oos_idx) >= len(positions) else \
            fwd_vol_bar[-len(positions):]

        # Use expanding-window trailing realized vol as proxy for predicted vol
        # (avoids needing a separate model — just use recent realized vol as forecast)
        trailing_vol = np.zeros(len(positions))
        for i in range(len(positions)):
            # Look back at trailing 20-day realized vol
            bar_idx_i = oos_idx[min(i, len(oos_idx) - 1)] if i < len(oos_idx) else oos_idx[-1]
            day_i = bar_daily_idx[min(bar_idx_i, len(bar_daily_idx) - 1)]
            start_d = max(0, day_i - _vol_window)
            if day_i > start_d and day_i < len(daily_log_ret):
                trailing_vol[i] = np.std(daily_log_ret[start_d:day_i]) * np.sqrt(252)
            else:
                trailing_vol[i] = np.nan

        # Inverse vol scaling: target 15% annualized vol
        VOL_TARGET = float(os.environ.get("VOL_TARGET_ANN", "0.15"))
        trailing_vol_clean = np.where(np.isnan(trailing_vol) | (trailing_vol < 0.01),
                                       VOL_TARGET, trailing_vol)
        vol_scale = np.clip(VOL_TARGET / trailing_vol_clean, 0.25, 3.0)
        positions = positions * vol_scale

        print(f"  Vol-sizing applied: target={VOL_TARGET:.0%} ann vol")
        print(f"  Trailing vol range: [{np.nanmin(trailing_vol):.3f}, {np.nanmax(trailing_vol):.3f}]")
        print(f"  Position scale range: [{vol_scale.min():.2f}, {vol_scale.max():.2f}]")
        print(f"  Mean scale: {vol_scale[positions != 0].mean():.2f}" if (positions != 0).any()
              else "  (no active positions)")

    # Apply v2 position-sizing gate (scale positions by microstructure quality)
    if os.environ.get("POSITION_GATE", "1") != "1":
        position_size_gate = None
        print("  v2 position-sizing gate: DISABLED (POSITION_GATE=0)")

    if ARCH_VERSION == "v2" and position_size_gate is not None:
        gate_aligned = position_size_gate[:len(positions)]
        if len(gate_aligned) < len(positions):
            gate_aligned = np.pad(gate_aligned, (0, len(positions) - len(gate_aligned)),
                                  mode='edge')
        positions = positions * gate_aligned
        print(f"  v2 position-sizing gate applied: "
              f"mean scale={gate_aligned[positions != 0].mean():.3f}" if (positions != 0).any()
              else "  v2 position-sizing gate applied (no active positions)")

    n_trades = np.sum(np.diff(positions != 0) > 0)
    print(f"  Positions active: {(positions != 0).mean():.1%} of bars")
    print(f"  Trade count: {n_trades}")

    # ── Collapse to daily resolution for honest evaluation ──────────────
    # Bar-level positions use repeated daily closes → most bars show zero
    # return. Aggregate to daily: take end-of-day position and daily return.
    print(f"\n  Collapsing to daily resolution for evaluation...")

    # Map each bar to its daily index
    n_oos = len(positions)
    # Use each OOS bar's actual day. The old np.linspace spread the OOS bars
    # evenly across *all* n_slow days, so out-of-sample positions were scored
    # against in-sample dates.
    _oos_bar_idx = np.asarray(result.oos_indices)[:n_oos]
    if len(_oos_bar_idx) != n_oos:
        raise ValueError(
            f"OOS index/position length mismatch: {len(_oos_bar_idx)} vs {n_oos}"
        )
    oos_bar_daily_idx = np.clip(
        aligner.get_daily_indices(_oos_bar_idx), 0, n_slow - 1
    )

    # Take end-of-day position (last bar of each day)
    unique_days = np.unique(oos_bar_daily_idx)
    daily_positions = np.zeros(len(unique_days))
    for i, d in enumerate(unique_days):
        day_mask = oos_bar_daily_idx == d
        day_bars = np.where(day_mask)[0]
        daily_positions[i] = positions[day_bars[-1]]  # EOD position

    # Daily log returns from actual close prices
    if FORWARD_HORIZON == 1:
        daily_log_returns = np.diff(np.log(cl_daily[unique_days] + 1e-8))
    else:
        # Multi-day forward returns: position held for FORWARD_HORIZON days
        log_p_days = np.log(cl_daily[unique_days] + 1e-8)
        daily_log_returns = np.zeros(len(unique_days) - FORWARD_HORIZON)
        for i in range(len(daily_log_returns)):
            daily_log_returns[i] = log_p_days[i + FORWARD_HORIZON] - log_p_days[i]

    # Strategy daily returns: position[t] * return[t → t+h]
    n_daily = min(len(daily_positions) - 1, len(daily_log_returns))
    daily_strat_returns = daily_positions[:n_daily] * daily_log_returns[:n_daily]

    # ── Transaction costs: 1-tick slippage per trade ($10/contract) ──
    # Model as a fraction of typical daily CL return magnitude
    COST_PER_TRADE = float(os.environ.get("COST_PER_TRADE", "0.00015"))
    # 0.00015 ≈ 1 tick ($0.01) on CL at ~$70/bbl as a log return fraction
    # A round-trip costs 2x (entry + exit)

    position_changes = np.diff(daily_positions[:n_daily + 1])
    trade_costs = np.abs(position_changes[:n_daily]) * COST_PER_TRADE
    n_daily_trades = (np.abs(position_changes[:n_daily]) > 0).sum()

    daily_strat_returns_gross = daily_strat_returns.copy()
    daily_strat_returns = daily_strat_returns - trade_costs

    total_cost = trade_costs.sum()
    print(f"\n  Transaction costs:")
    print(f"    Cost per trade:   {COST_PER_TRADE:.5f} ({COST_PER_TRADE*100:.3f}%)")
    print(f"    Daily trades:     {n_daily_trades}")
    print(f"    Total cost drag:  {total_cost:.6f} ({total_cost*100:.3f}%)")
    print(f"    Gross P&L:        {daily_strat_returns_gross.sum():.6f}")
    print(f"    Net P&L:          {daily_strat_returns.sum():.6f}")

    # Also keep bar-level for charting compatibility
    if TARGET_MODE == "vol" and y_returns_bar is not None:
        # In vol mode, y_true is vol not returns — use actual returns for P&L
        bar_ret = y_returns_bar[:len(positions)]
        strat_returns_bar = positions[:-1] * bar_ret[1:]
    else:
        strat_returns_bar = positions[:-1] * result.y_true_oos[1:]

    # Metrics at daily resolution
    strat_returns = daily_strat_returns  # use daily for all metrics
    sr = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)
    from scipy.stats import skew as calc_skew, kurtosis as calc_kurt
    T_obs = len(strat_returns)
    dsr = deflated_sharpe_ratio(
        observed_sr=sr / np.sqrt(252),  # annualized → per-bar
        n_trials=1, T=T_obs,
        skew=float(calc_skew(strat_returns)),
        kurtosis_excess=float(calc_kurt(strat_returns)),
    )
    sortino = sortino_ratio(strat_returns)
    cum_pnl = np.cumsum(strat_returns)
    max_dd = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl)

    # Hit rate on active days only
    active_days = daily_positions[:n_daily] != 0
    if active_days.sum() > 0:
        active_returns = strat_returns[active_days]
        hit_rate = (active_returns > 0).mean()
        n_wins = (active_returns > 0).sum()
        n_losses = (active_returns < 0).sum()
        n_flat = (active_returns == 0).sum()
        avg_win = active_returns[active_returns > 0].mean() if n_wins > 0 else 0
        avg_loss = abs(active_returns[active_returns < 0].mean()) if n_losses > 0 else 1e-8
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
    else:
        hit_rate = 0
        n_wins = n_losses = n_flat = 0
        win_loss_ratio = 0

    print(f"\n  Strategy Performance (daily resolution):")
    print(f"    Sharpe Ratio:     {sr:.4f}")
    print(f"    Deflated SR:      {dsr:.4f}")
    print(f"    Sortino Ratio:    {sortino:.4f}")
    print(f"    Max Drawdown:     {max_dd:.6f}")
    print(f"    Total P&L:        {cum_pnl[-1]:.6f}")
    print(f"    # Trading days:   {T_obs}")
    print(f"    Days in position: {active_days.sum()} ({active_days.mean():.1%})")
    print(f"    Hit Rate (active):{hit_rate:.1%}")
    print(f"    Wins / Losses:    {n_wins} / {n_losses} (flat: {n_flat})")
    print(f"    Avg Win / Loss:   {avg_win:.6f} / {avg_loss:.6f}")
    print(f"    Win/Loss Ratio:   {win_loss_ratio:.2f}x")

    # Per-fold diagnostics
    print(f"\n  Per-Fold Results:")
    print(f"  {'Fold':>4s}  {'Train':>8s}  {'Test':>8s}  "
          f"{'||w||':>8s}  {'TrainMSE':>10s}")
    for fr in result.fold_results:
        print(f"  {fr.fold_idx:4d}  "
              f"{fr.train_end - fr.train_start:8d}  "
              f"{fr.test_end - fr.test_start:8d}  "
              f"{fr.weights_norm:8.4f}  "
              f"{fr.train_loss:10.8f}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Backtest complete in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Save results for charting
    horizon_tag = f"_h{FORWARD_HORIZON}d" if FORWARD_HORIZON > 1 else ""
    signal_tag = f"_{SIGNAL_MODE}" if SIGNAL_MODE != "ou" else ""
    warmup_tag = f"_wu{WARMUP_DAYS}" if WARMUP_DAYS > 0 else ""
    loss_tag = f"_{COMBINER_LOSS}" if COMBINER_LOSS != "mse" else ""
    adaptive_tag = "_adaptiveQ" if (SIGNAL_MODE == "kalman" and ADAPTIVE_Q) else ""
    arch_tag = f"_{ARCH_VERSION}" if ARCH_VERSION != "v1" else ""
    target_tag = f"_{TARGET_MODE}" if TARGET_MODE != "returns" else ""
    n_slow_k = len(slow_kernels_list)
    run_label = f"{n_fast_k}fast_{n_slow_k}slow_twostage{arch_tag}{target_tag}{horizon_tag}{signal_tag}{loss_tag}{adaptive_tag}{warmup_tag}"
    null_tag = "" if NULL_MODE == "none" else f"_null-{NULL_MODE}"
    seed_tag = "" if _SEED_OFFSET == 0 else f"_seed{_SEED_OFFSET}"
    run_label = f"{run_label}{null_tag}{seed_tag}"
    results_file = os.path.join(_ROOT, f"backtest_results_{run_label}.npz")
    np.savez(
        results_file,
        y_pred_oos=result.y_pred_oos,
        y_true_oos=result.y_true_oos,
        oos_indices=result.oos_indices,
        s_scores=s_scores,
        positions=positions,
        strat_returns=strat_returns,
        strat_returns_gross=daily_strat_returns_gross,
        strat_returns_bar=strat_returns_bar,
        trade_costs=trade_costs,
        cum_pnl=cum_pnl,
        daily_positions=daily_positions,
        daily_strat_returns=daily_strat_returns,
        hit_rate=hit_rate,
        win_loss_ratio=win_loss_ratio,
        warmup_days=WARMUP_DAYS,
        cost_per_trade=COST_PER_TRADE,
        alpha_fast=mkl_result['alpha_fast'],
        alpha_slow=mkl_result['alpha_slow'],
        beta=mkl_result['beta'],
        kernel_names=np.array([k.name for k in fast_kernels] + [k.name for k in slow_kernels_list]),
        arch_version=ARCH_VERSION,
        # Headline metrics, so downstream tooling reads numbers instead of
        # re-deriving them (or scraping stdout).
        sharpe=float(sr),
        sortino=float(sortino),
        deflated_sr=float(dsr),
        max_drawdown=float(max_dd),
        total_pnl=float(cum_pnl[-1]) if len(cum_pnl) else 0.0,
        n_trades=int(n_trades),
        n_days_active=int(active_days.sum()),
        n_days_total=int(n_daily),
        stage2_oos_corr=_stage2_corr,
        stage2_oos_r2=_stage2_r2,
        stage2_weights=np.array(list(_stage2_weights.items()), dtype=object),
        # Configuration, so a saved run is self-describing.
        null_mode=NULL_MODE,
        seed_offset=_SEED_OFFSET,
        n_fast_bars=int(n_fast),
        n_slow_days=int(n_slow),
    )
    print(f"\n  Results saved to {os.path.basename(results_file)}")

    # Also save baseline (uniform weights) for comparison
    np.savez(
        os.path.join(_ROOT, f"backtest_results_{n_fast_k}fast_2slow_baseline.npz"),
        y_pred_oos=result.y_pred_oos,
        y_true_oos=result.y_true_oos,
        s_scores=s_scores,
        positions=positions,
        strat_returns=strat_returns,
        cum_pnl=cum_pnl,
    )

    return result, strat_returns


if __name__ == "__main__":
    main()
