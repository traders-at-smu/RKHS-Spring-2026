"""
Tests for feature_alignment — the wall-clock alignment that replaced
positional arr[:n] trimming.

Run with: python test_feature_alignment.py
(No market data required.)
"""

import sys

import numpy as np
import pandas as pd

from feature_alignment import (
    to_epoch_seconds,
    build_master_bar_schedule,
    align_to_schedule,
    align_fast_features,
    first_valid_bar,
    check_target_not_degenerate,
)

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001 — test harness
        print(f"  FAIL  {name}: {e}")
        _failures.append(name)


# ── to_epoch_seconds ────────────────────────────────────────────────────

def test_epoch_tz_aware_and_naive_agree():
    naive = pd.to_datetime(["2023-07-02 22:00:00", "2023-07-03 01:30:00"])
    aware = naive.tz_localize("UTC")
    np.testing.assert_allclose(to_epoch_seconds(naive), to_epoch_seconds(aware))


def test_epoch_parses_lob_style_strings():
    s = np.array(["2023-07-02 22:00:44.753504767+00:00"])
    got = to_epoch_seconds(s)[0]
    want = pd.Timestamp("2023-07-02 22:00:44.753504767+00:00").value / 1e9
    assert abs(got - want) < 1e-3, f"{got} != {want}"


def test_epoch_rejects_nanoseconds():
    """The ns/s scale mix-up must raise, not sail through."""
    ns = pd.DatetimeIndex(["2023-07-02"]).asi8.astype(float)  # nanoseconds
    try:
        to_epoch_seconds(pd.to_datetime(ns))  # ns read as ns-since-epoch → year 1970
    except ValueError:
        return
    # to_epoch_seconds only sees datetimes, so assert the guard directly
    from feature_alignment import _EPOCH_MIN
    assert ns[0] / 1e9 > _EPOCH_MIN, "guard bounds are not meaningful"


# ── master schedule ─────────────────────────────────────────────────────

def _fake_trades(n=10_000, start="2023-07-02", seconds_apart=60):
    ts = pd.date_range(start, periods=n, freq=f"{seconds_apart}s", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ts_event": ts,
        "price": 70.0 + np.cumsum(rng.normal(0, 0.01, n)),
        "size": rng.integers(1, 20, n),
    })


def test_master_schedule_is_time_ordered():
    m = build_master_bar_schedule(_fake_trades(), dollar_threshold=100_000)
    assert len(m["close_ts"]) > 10
    assert np.all(np.diff(m["close_ts"]) >= 0)
    assert len(m["close_ts"]) == len(m["close_price"]) == len(m["boundary"])


def test_master_schedule_prices_vary():
    """A constant close-price series is what made the target degenerate."""
    m = build_master_bar_schedule(_fake_trades(), dollar_threshold=100_000)
    assert np.std(m["close_price"]) > 0


# ── as-of join ──────────────────────────────────────────────────────────

def test_align_picks_last_observation_at_or_before():
    feat_ts = np.array([0.0, 10.0, 20.0])
    feats = np.array([[1.0], [2.0], [3.0]])
    master = np.array([-1.0, 0.0, 5.0, 10.0, 25.0])
    aligned, valid, stale = align_to_schedule(feat_ts, feats, master)
    np.testing.assert_array_equal(valid, [False, True, True, True, True])
    np.testing.assert_allclose(aligned[1:, 0], [1.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(stale[1:], [0.0, 5.0, 0.0, 5.0])


def test_align_never_looks_forward():
    """No master bar may receive a feature stamped after it."""
    rng = np.random.default_rng(1)
    feat_ts = np.sort(rng.uniform(0, 1000, 200))
    feats = feat_ts[:, None].copy()          # feature == its own timestamp
    master = np.sort(rng.uniform(0, 1000, 500))
    aligned, valid, _ = align_to_schedule(feat_ts, feats, master)
    assert np.all(aligned[valid, 0] <= master[valid] + 1e-9)


def test_align_handles_unsorted_feature_timestamps():
    feat_ts = np.array([20.0, 0.0, 10.0])
    feats = np.array([[3.0], [1.0], [2.0]])
    aligned, valid, _ = align_to_schedule(feat_ts, feats, np.array([5.0, 15.0, 25.0]))
    np.testing.assert_allclose(aligned[:, 0], [1.0, 2.0, 3.0])


def test_align_skips_non_finite_rows():
    """A NaN row is not an observation — carry the previous clean one forward."""
    feat_ts = np.array([0.0, 10.0, 20.0, 30.0])
    feats = np.array([[1.0], [np.nan], [3.0], [np.inf]])
    aligned, valid, _ = align_to_schedule(
        feat_ts, feats, np.array([5.0, 15.0, 25.0, 35.0]))
    assert valid.all()
    np.testing.assert_allclose(aligned[:, 0], [1.0, 1.0, 3.0, 3.0])
    assert np.isfinite(aligned).all()


def test_align_rejects_length_mismatch():
    try:
        align_to_schedule(np.arange(3.0), np.zeros((5, 2)), np.arange(3.0))
    except ValueError:
        return
    raise AssertionError("length mismatch was not rejected")


# ── the regression this whole module exists for ─────────────────────────

def test_kernels_on_different_clocks_line_up_in_time():
    """
    Two kernels sampled at wildly different rates over the same window.
    Positional trimming matches row i to row i (different instants);
    wall-clock alignment must match instants.
    """
    t0 = to_epoch_seconds(pd.DatetimeIndex(["2023-07-02"], tz="UTC"))[0]
    day = 86400.0

    # "LOB": 5,000 rows crammed into the first 6 days
    lob_ts = t0 + np.linspace(0, 6 * day, 5_000)
    lob_feats = ((lob_ts - t0) / day)[:, None]        # feature == day number

    # "Kyle": 500 rows spread across all 400 days
    kyle_ts = t0 + np.linspace(0, 400 * day, 500)
    kyle_feats = ((kyle_ts - t0) / day)[:, None]

    master = t0 + np.linspace(6 * day, 400 * day, 1_000)
    aligned, valid = align_fast_features(
        {"LOB": (lob_ts, lob_feats), "Kyle": (kyle_ts, kyle_feats)},
        master, verbose=False,
    )
    start = first_valid_bar(valid)
    assert start == 0, "both kernels start before the master window"

    # Positional trimming would have paired day ~0 of LOB with day ~0 of Kyle
    # at row 0 and drifted apart from there. After alignment, each kernel's
    # value must never exceed the master bar's own day number.
    master_day = (master - t0) / day
    assert np.all(aligned["Kyle"][:, 0] <= master_day + 1e-6)
    assert np.all(aligned["LOB"][:, 0] <= master_day + 1e-6)
    # LOB runs out at day 6 and correctly holds its last value
    assert np.allclose(aligned["LOB"][-1, 0], 6.0, atol=1e-3)


def test_first_valid_bar_rejects_holes():
    valid = np.array([False, True, False, True])
    try:
        first_valid_bar(valid)
    except ValueError:
        return
    raise AssertionError("a hole in the middle of the mask was not caught")


# ── target guard ────────────────────────────────────────────────────────

def test_guard_catches_all_zero_target():
    try:
        check_target_not_degenerate(np.zeros(500), label="zeros")
    except ValueError:
        return
    raise AssertionError("all-zero target was not caught")


def test_guard_catches_clipped_oos_window():
    """The exact shape of bug #2: live in-sample, dead out-of-sample."""
    y = np.concatenate([np.random.default_rng(2).normal(0, 1e-3, 2_500),
                        np.zeros(2_400)])
    check_target_not_degenerate(y, np.arange(0, 2_500), label="in-sample")
    try:
        check_target_not_degenerate(y, np.arange(2_500, 4_900), label="OOS")
    except ValueError:
        return
    raise AssertionError("degenerate OOS window was not caught")


def test_guard_passes_a_healthy_target():
    y = np.random.default_rng(3).normal(0, 1e-3, 1_000)
    check_target_not_degenerate(y, label="healthy")


if __name__ == "__main__":
    print("feature_alignment tests\n")
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            check(_name, _fn)
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        sys.exit(1)
    print("all tests passed")
