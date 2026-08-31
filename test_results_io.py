"""
Tests for results_io — keeping falsification control runs out of the
tools that auto-discover a results file.

Run with: python test_results_io.py   (no market data required)
"""

import os
import shutil
import sys
import tempfile

import numpy as np

from results_io import is_control_run, select_results_file

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:  # noqa: BLE001 — test harness
        print(f"  FAIL  {name}: {e}")
        _failures.append(name)


def _make(dirpath, name, null_mode="none", keys=("daily_strat_returns",)):
    payload = {k: np.zeros(3) for k in keys}
    payload["null_mode"] = np.array(null_mode)
    path = os.path.join(dirpath, name)
    np.savez(path, **payload)
    return path


def test_control_run_detected_by_filename():
    with tempfile.TemporaryDirectory() as d:
        p = _make(d, "backtest_results_x_null-noise.npz", null_mode="noise")
        assert is_control_run(p)


def test_control_run_detected_by_saved_flag():
    """A renamed file must still be caught — the flag is inside the data."""
    with tempfile.TemporaryDirectory() as d:
        p = _make(d, "backtest_results_innocent_name.npz",
                  null_mode="shuffle_target")
        assert is_control_run(p)


def test_real_run_not_flagged():
    with tempfile.TemporaryDirectory() as d:
        p = _make(d, "backtest_results_real.npz", null_mode="none")
        assert not is_control_run(p)


def test_autodiscovery_skips_controls_even_when_newest():
    with tempfile.TemporaryDirectory() as d:
        real = _make(d, "backtest_results_real.npz")
        os.utime(real, (1_000, 1_000))            # old
        ctrl = _make(d, "backtest_results_z_null-noise.npz",
                     null_mode="noise")
        os.utime(ctrl, (2_000, 2_000))            # newest
        got = select_results_file(root=d, require=("daily_strat_returns",))
        assert os.path.basename(got) == "backtest_results_real.npz", got


def test_autodiscovery_skips_the_baseline_stub():
    with tempfile.TemporaryDirectory() as d:
        real = _make(d, "backtest_results_real.npz")
        os.utime(real, (1_000, 1_000))
        stub = _make(d, "backtest_results_2fast_2slow_baseline.npz")
        os.utime(stub, (2_000, 2_000))
        got = select_results_file(root=d, require=("daily_strat_returns",))
        assert os.path.basename(got) == "backtest_results_real.npz", got


def test_autodiscovery_skips_files_missing_required_arrays():
    with tempfile.TemporaryDirectory() as d:
        full = _make(d, "backtest_results_full.npz",
                     keys=("daily_strat_returns", "daily_positions"))
        os.utime(full, (1_000, 1_000))
        partial = _make(d, "backtest_results_partial.npz",
                        keys=("daily_strat_returns",))
        os.utime(partial, (2_000, 2_000))
        got = select_results_file(
            root=d, require=("daily_strat_returns", "daily_positions"))
        assert os.path.basename(got) == "backtest_results_full.npz", got


def test_only_controls_available_raises_rather_than_returning_one():
    with tempfile.TemporaryDirectory() as d:
        _make(d, "backtest_results_a_null-noise.npz", null_mode="noise")
        try:
            select_results_file(root=d)
        except FileNotFoundError:
            return
        raise AssertionError("returned a control run instead of raising")


def test_explicit_path_is_honoured():
    with tempfile.TemporaryDirectory() as d:
        p = _make(d, "backtest_results_chosen.npz")
        assert select_results_file(p, root=d) == p


if __name__ == "__main__":
    print("results_io tests\n")
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            check(_name, _fn)
    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        sys.exit(1)
    print("all tests passed")
