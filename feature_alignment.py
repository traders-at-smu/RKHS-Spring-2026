"""
Timestamp-based alignment for multi-resolution kernel features.

Why this module exists
----------------------
Every "fast" kernel in this pipeline is computed on its own sampling
schedule:

    LOB        — one row per LOB snapshot, concatenated day by day
    VPIN       — one row per $1M dollar bar
    Kyle       — one row per fixed-trade-count bin (~5,000 bins total)
    Hawkes     — one row per fixed 300-second wall-clock window
    VannaCharm — one row per $1M dollar bar

Those schedules have wildly different row counts over the same 471
trading days (LOB ~372k rows vs Kyle ~5k rows). The original pipeline
aligned them by truncating every array to the shortest one:

    n_fast = min(a.shape[0] for a in arrays)
    arrays = [a[:n_fast] for a in arrays]

which lines up row i of each kernel regardless of what instant row i
actually refers to. Row 4,000 of LOB was July 2023 while row 4,000 of
Kyle was late 2024, so the v2 order-flow PCA was concatenating features
from completely different time periods.

This module gives every kernel a wall-clock timestamp and resamples all
of them onto one master dollar-bar schedule using a backward as-of join
(the last observation at or before each bar close). Row i then means the
same instant on every kernel, and no future information leaks backwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Sanity bounds for epoch *seconds*. The original aligner silently mixed
# nanoseconds and seconds (pandas .asi8 on tz-aware vs tz-naive indices),
# which collapsed the whole bar→day mapping onto a single day. Anything
# outside 2000-01-01 .. 2100-01-01 means the unit is wrong, so fail loudly.
_EPOCH_MIN = 946_684_800.0     # 2000-01-01 UTC
_EPOCH_MAX = 4_102_444_800.0   # 2100-01-01 UTC


def to_epoch_seconds(values) -> np.ndarray:
    """
    Convert anything date-like to float64 epoch seconds (UTC).

    Handles tz-aware and tz-naive input, datetime64 arrays, and ISO
    strings. Raises if the result is not a plausible epoch in seconds —
    that check is what catches the ns/s scale mix-up.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    # .asi8 returns the raw integer in the index's own unit, and pandas 3
    # defaults to microseconds — reading it as nanoseconds silently divides
    # every timestamp by 1000 and lands you in 1970. Pin the unit first.
    if hasattr(idx, "as_unit"):
        idx = idx.as_unit("ns")
    secs = idx.asi8.astype(np.float64) / 1e9

    finite = secs[np.isfinite(secs)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        if lo < _EPOCH_MIN or hi > _EPOCH_MAX:
            raise ValueError(
                f"Timestamps are not plausible epoch seconds "
                f"(min={lo:.3e}, max={hi:.3e}). Expected "
                f"{_EPOCH_MIN:.3e}..{_EPOCH_MAX:.3e} — check the time unit."
            )
    return secs


def build_master_bar_schedule(trades_df, dollar_threshold: float = 1_000_000):
    """
    Build the canonical dollar-bar schedule from the raw trade tape.

    Every fast kernel is resampled onto this schedule, and the backtest
    target is computed from its close prices, so it is the single
    definition of "bar i" for the whole pipeline.

    Returns a dict with:
        close_ts    : (n_bars,) float64 — epoch seconds at each bar close
        close_price : (n_bars,) float64 — trade price at each bar close
        boundary    : (n_bars,) int64   — row in trades_df closing the bar
    """
    ts_s = to_epoch_seconds(trades_df["ts_event"])
    prices = trades_df["price"].values.astype(float)
    sizes = trades_df["size"].values.astype(float)

    cum_notional = np.cumsum(prices * sizes)
    if cum_notional[-1] < dollar_threshold:
        raise ValueError(
            f"Trade tape totals ${cum_notional[-1]:,.0f}, less than one "
            f"${dollar_threshold:,.0f} bar."
        )

    thresholds = np.arange(dollar_threshold, cum_notional[-1], dollar_threshold)
    boundaries = np.searchsorted(cum_notional, thresholds)
    boundaries = np.clip(boundaries, 0, len(prices) - 1)

    close_ts = ts_s[boundaries]
    if not np.all(np.diff(close_ts) >= 0):
        raise ValueError("Trade tape is not sorted by timestamp.")

    return {
        "close_ts": close_ts,
        "close_price": prices[boundaries],
        "boundary": boundaries.astype(np.int64),
    }


def align_to_schedule(feature_ts, features, master_ts):
    """
    Backward as-of join of one feature matrix onto the master schedule.

    For each master bar close, take the most recent feature row whose own
    timestamp is at or before that close. Bars earlier than the feature's
    first observation are marked invalid rather than back-filled.

    Rows containing non-finite values are not observations, so they are
    dropped and the previous clean row is carried forward. The LOB feature
    files hold 451 such rows spread over 19 days; positional trimming never
    reached them because it only ever read the first six days of LOB.

    Returns
    -------
    aligned   : (n_master, d) float64 — zeros where invalid
    valid     : (n_master,) bool      — a real observation was available
    staleness : (n_master,) float64   — seconds since that observation
    """
    features = np.asarray(features, dtype=float)
    if features.ndim == 1:
        features = features[:, None]
    feature_ts = np.asarray(feature_ts, dtype=float)
    master_ts = np.asarray(master_ts, dtype=float)

    if feature_ts.shape[0] != features.shape[0]:
        raise ValueError(
            f"Timestamp/feature length mismatch: {feature_ts.shape[0]} "
            f"timestamps vs {features.shape[0]} rows."
        )

    clean = np.isfinite(features).all(axis=1) & np.isfinite(feature_ts)
    feature_ts = feature_ts[clean]
    features = features[clean]
    if feature_ts.size == 0:
        raise ValueError("Every feature row is non-finite — nothing to align.")

    order = np.argsort(feature_ts, kind="mergesort")
    fts = feature_ts[order]
    feats = features[order]

    pos = np.searchsorted(fts, master_ts, side="right") - 1
    valid = pos >= 0

    aligned = np.zeros((master_ts.shape[0], feats.shape[1]), dtype=float)
    aligned[valid] = feats[pos[valid]]

    staleness = np.full(master_ts.shape[0], np.nan)
    staleness[valid] = master_ts[valid] - fts[pos[valid]]

    return aligned, valid, staleness


def align_fast_features(named_features, master_ts, verbose: bool = True):
    """
    Align several fast kernels onto the master schedule at once.

    Parameters
    ----------
    named_features : dict
        name -> (timestamps_epoch_seconds, feature_matrix)
    master_ts : (n_master,) float64
        Bar-close epoch seconds from build_master_bar_schedule().

    Returns
    -------
    aligned  : dict name -> (n_master, d) array
    valid    : (n_master,) bool — True where *every* kernel has an
               observation. Because the join is backward-only this mask is
               monotone: False for a leading warm-up region, then True.
    """
    aligned = {}
    valid_all = np.ones(len(master_ts), dtype=bool)

    for name, (ts, feats) in named_features.items():
        a, v, staleness = align_to_schedule(ts, feats, master_ts)
        aligned[name] = a
        valid_all &= v

        if verbose:
            arr = np.asarray(feats, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
            n_src = arr.shape[0]
            n_bad = int((~np.isfinite(arr).all(axis=1)).sum())
            med = np.nanmedian(staleness) if v.any() else float("nan")
            worst = np.nanmax(staleness) if v.any() else float("nan")
            note = f", {n_bad} non-finite rows skipped" if n_bad else ""
            print(f"    {name:<12} {n_src:>8,} rows → {v.sum():,}/{len(v):,} "
                  f"aligned | staleness median {med:>8.1f}s, "
                  f"max {worst:>9.1f}s{note}")

    return aligned, valid_all


def first_valid_bar(valid: np.ndarray) -> int:
    """
    Index of the first master bar where every kernel has an observation.

    Asserts the mask is monotone (a single leading warm-up gap), which a
    backward as-of join guarantees. A hole in the middle means one of the
    feature timestamp arrays is malformed.
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        raise ValueError("No master bar has an observation from every kernel.")
    start = int(idx[0])
    if not valid[start:].all():
        n_holes = int((~valid[start:]).sum())
        raise ValueError(
            f"Alignment mask has {n_holes} gaps after the warm-up region — "
            f"a feature timestamp array is out of order or has NaNs."
        )
    return start


def check_target_not_degenerate(
    y: np.ndarray,
    indices: np.ndarray | None = None,
    label: str = "target",
    min_std: float = 1e-10,
    max_zero_frac: float = 0.90,
) -> None:
    """
    Fail loudly if a target is constant or almost all zeros.

    This is the guard for the second bug: a truncated bar→day mapping
    clipped every out-of-sample bar to the same day, so the forward-return
    target was identically zero across the whole OOS window and the
    ElasticNet collapsed to an intercept. That produced a Sharpe of 1.56
    from nothing but fold-boundary intercept drift, and nothing in the
    pipeline complained.
    """
    y = np.asarray(y, dtype=float)
    if indices is not None:
        y = y[np.asarray(indices)]

    if y.size == 0:
        raise ValueError(f"{label} is empty.")

    std = float(np.nanstd(y))
    zero_frac = float(np.mean(np.abs(y) < 1e-12))

    if std < min_std:
        raise ValueError(
            f"{label} is degenerate: std={std:.3e} over {y.size} samples. "
            f"The bar→day mapping is almost certainly clipping."
        )
    if zero_frac > max_zero_frac:
        raise ValueError(
            f"{label} is degenerate: {zero_frac:.1%} of {y.size} samples are "
            f"exactly zero (limit {max_zero_frac:.0%}). The bar→day mapping "
            f"is almost certainly clipping."
        )

    print(f"    {label}: std={std:.3e}, zeros={zero_frac:.1%}, n={y.size} — OK")


def reindex_daily(df, target_dates, columns=None):
    """
    Backward as-of join a daily DataFrame onto a target date index.

    VIX, SPY and XLE cover 378 US equity sessions while CL trades 468 days
    over the same span. The original code joined them by row position and
    padded the tail with zeros, so SPY row i was compared against a CL day
    that drifted further away as the sample went on, and the last ~90 days
    of correlation were computed against zeros. Join on the date instead.
    """
    src_dates = to_epoch_seconds(df.index)
    tgt_dates = to_epoch_seconds(target_dates)
    cols = list(df.columns) if columns is None else list(columns)

    values = df[cols].to_numpy(dtype=float)
    aligned, valid, _ = align_to_schedule(src_dates, values, tgt_dates)
    aligned[~valid] = np.nan

    out = pd.DataFrame(aligned, columns=cols, index=pd.Index(target_dates))
    return out.ffill().bfill()


# ============================================================================
# Causal normalisation
# ============================================================================

def expanding_mean(x: np.ndarray) -> np.ndarray:
    """Mean of x[:i + 1] at every row i."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x, axis=0) / np.arange(1, x.shape[0] + 1, dtype=float).reshape(
        (-1,) + (1,) * (x.ndim - 1)
    )


def expanding_zscore(
    features: np.ndarray,
    min_periods: int = 20,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Standardise each row against the rows before it, never the whole sample.

    The feature builders used to finish with

        mu, sigma = features.mean(axis=0), features.std(axis=0) + 1e-8
        return (features - mu) / sigma

    where mu and sigma are computed over every row, including rows that had
    not happened yet at bar i. That leaks the sample's future distribution
    into every training bar. Row i here sees only rows 0..i.

    Rows before `min_periods` have too little history to standardise and are
    returned as zero rather than as an enormous ratio of two tiny numbers.

    Implemented with running sums so it stays O(n) — the fast kernels carry
    150k-370k rows and a Python loop over them is not viable.
    """
    x = np.asarray(features, dtype=float)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]
    n = x.shape[0]
    if n == 0:
        return x[:, 0] if one_d else x

    # Offset by the first row so the running sums of squares stay well
    # conditioned when the raw values are far from zero.
    z = x - x[0]
    counts = np.arange(1, n + 1, dtype=float)[:, None]
    mean = np.cumsum(z, axis=0) / counts
    var = np.maximum(np.cumsum(z * z, axis=0) / counts - mean * mean, 0.0)

    out = (z - mean) / (np.sqrt(var) + eps)
    out[:min(min_periods, n)] = 0.0
    out[~np.isfinite(out)] = 0.0

    return out[:, 0] if one_d else out
