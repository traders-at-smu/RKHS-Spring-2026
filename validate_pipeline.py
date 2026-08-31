"""
Falsification suite for the RKHS backtest.

A backtest you cannot break is a backtest you cannot trust. This driver
re-runs the pipeline with the information it claims to use destroyed, and
checks that the results collapse. Three controls:

    noise           every prediction feature becomes iid N(0,1) — the
                    "no kernels" null.
    shuffle_time    feature rows permuted in time. Marginal distributions
                    are preserved exactly; only the correspondence between
                    features and dates is broken. Catches alignment bugs
                    specifically: code that ignores time cannot tell the
                    difference between this and the real thing.
    shuffle_target  the forward-return target is permuted, so nothing can
                    predict it. Any surviving out-of-sample correlation is
                    leakage in the harness itself.

Each control runs across several seeds, because a single draw of the
random Fourier features says nothing about stability.

Usage:
    .venv/bin/python validate_pipeline.py             # 3 seeds, ~25 min
    .venv/bin/python validate_pipeline.py --seeds 1   # quick pass
    .venv/bin/python validate_pipeline.py --reuse     # skip completed runs

Exits non-zero if a check that must pass does not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.join(_ROOT, ".venv", "bin", "python")

# The Run 25 configuration. Controls differ from the real run only in
# NULL_MODE, so any difference in the results is attributable to that.
BASE_ENV = {
    "ARCH_VERSION": "v2",
    "SIGNAL_MODE": "kalman",
    "KALMAN_Q": "aggressive",
    "TUNE_KERNELS": "0",
    "WARMUP_DAYS": "60",
    "VOL_SIZING": "1",
    "ENTRY_CHOPPY": "1.50",
    "EXIT_TRENDING": "0.10",
    "EXIT_CHOPPY": "0.35",
}

MODES = ["none", "noise", "shuffle_time", "shuffle_target"]

# A stage-2 out-of-sample correlation this large under a destroyed target
# would mean the harness is leaking. Nothing to do with whether the
# strategy is any good — this is a check on the machinery, and it is the
# only condition that fails the suite. Judgements about the strategy are
# printed as [INFO]: a suite that fails whenever the strategy is bad is a
# suite people learn to ignore.
LEAKAGE_TOLERANCE = 0.05


def results_path(mode: str, seed: int) -> str:
    label = "2fast_3slow_twostage_v2_kalman_wu60"
    if mode != "none":
        label += f"_null-{mode}"
    if seed != 0:
        label += f"_seed{seed}"
    return os.path.join(_ROOT, f"backtest_results_{label}.npz")


def run_one(mode: str, seed: int, reuse: bool) -> dict | None:
    path = results_path(mode, seed)
    if reuse and os.path.exists(path):
        print(f"  [cached] mode={mode:<14} seed={seed}")
    else:
        env = dict(os.environ)
        env.update(BASE_ENV)
        env["NULL_MODE"] = mode
        env["SEED"] = str(seed)
        print(f"  [run]    mode={mode:<14} seed={seed} ... ", end="", flush=True)
        proc = subprocess.run(
            [_PY, os.path.join(_ROOT, "run_backtest.py")],
            env=env, cwd=_ROOT, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
            print("FAILED")
            print(f"    exit {proc.returncode}\n{tail}")
            return None
        print("ok")

    if not os.path.exists(path):
        print(f"    expected {os.path.basename(path)}, not found")
        return None

    d = np.load(path, allow_pickle=True)

    def get(key, default=float("nan")):
        return float(d[key]) if key in d.files else default

    return {
        "mode": mode,
        "seed": seed,
        "sharpe": get("sharpe"),
        "sortino": get("sortino"),
        "deflated_sr": get("deflated_sr"),
        "max_dd": get("max_drawdown"),
        "oos_corr": get("stage2_oos_corr"),
        "oos_r2": get("stage2_oos_r2"),
        "hit_rate": get("hit_rate"),
        "n_days_active": get("n_days_active"),
    }


def summarise(rows: list[dict], mode: str) -> dict:
    sel = [r for r in rows if r["mode"] == mode]
    if not sel:
        return {}
    out = {"n": len(sel)}
    for k in ("sharpe", "oos_corr", "oos_r2", "hit_rate", "n_days_active"):
        vals = np.array([r[k] for r in sel], dtype=float)
        out[k] = float(np.nanmean(vals))
        out[k + "_sd"] = float(np.nanstd(vals))
        out[k + "_max"] = float(np.nanmax(np.abs(vals)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of seeds per mode (default 3)")
    ap.add_argument("--reuse", action="store_true",
                    help="skip runs whose results file already exists")
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    args = ap.parse_args()

    print("=" * 72)
    print("  RKHS pipeline falsification suite")
    print("=" * 72)
    print(f"  {len(args.modes)} modes x {args.seeds} seeds = "
          f"{len(args.modes) * args.seeds} runs\n")

    rows = []
    for mode in args.modes:
        for seed in range(args.seeds):
            r = run_one(mode, seed, args.reuse)
            if r is not None:
                rows.append(r)

    if not rows:
        print("\nNo runs completed.")
        return 1

    print("\n" + "=" * 72)
    print("  Results (mean +/- sd across seeds)")
    print("=" * 72)
    print(f"  {'mode':<15}{'n':>3}{'Sharpe':>16}{'OOS corr':>16}"
          f"{'OOS R2':>12}{'active d':>10}")
    print("  " + "-" * 70)
    stats = {}
    for mode in args.modes:
        st = summarise(rows, mode)
        if not st:
            continue
        stats[mode] = st
        print(f"  {mode:<15}{st['n']:>3}"
              f"{st['sharpe']:>10.2f} +/-{st['sharpe_sd']:<4.2f}"
              f"{st['oos_corr']:>10.4f} +/-{st['oos_corr_sd']:<4.3f}"
              f"{st['oos_r2']:>12.4f}"
              f"{st['n_days_active']:>10.0f}")

    print("\n" + "=" * 72)
    print("  Checks")
    print("=" * 72)
    failures = []

    # Must pass: with the target destroyed, nothing may correlate with it.
    st = stats.get("shuffle_target")
    if st:
        worst = st["oos_corr_max"]
        ok = worst < LEAKAGE_TOLERANCE
        print(f"  [{'PASS' if ok else 'FAIL'}] no target leakage: "
              f"|OOS corr| under a shuffled target = {worst:.4f} "
              f"(tolerance {LEAKAGE_TOLERANCE})")
        if not ok:
            failures.append(
                "shuffle_target still correlates with the target — the "
                "harness is leaking it into training"
            )

    # Correctness of the time alignment itself is asserted directly in
    # test_feature_alignment.py (cross-clock matching, no-lookahead), not
    # inferred from an end-to-end Sharpe comparison. Sharpe turned out to
    # be far too noisy across seeds to discriminate anything — see below.

    print()
    real = stats.get("none")

    # Reported, not gated: everything past this point is a statement about
    # the strategy and the metric, not about the harness.
    def _distinguishable(a, b):
        """Two-sigma separation of the seed means."""
        se = np.sqrt(a["sharpe_sd"] ** 2 / a["n"] + b["sharpe_sd"] ** 2 / b["n"])
        return abs(a["sharpe"] - b["sharpe"]) > 2 * se, 2 * se

    control = stats.get("shuffle_target")
    if real and control:
        sep, band = _distinguishable(real, control)
        print(f"  [INFO] is Sharpe a usable metric here?")
        print(f"         real   {real['sharpe']:+.2f} +/-{real['sharpe_sd']:.2f}"
              f"   (n={real['n']} seeds)")
        print(f"         target-shuffled {control['sharpe']:+.2f} "
              f"+/-{control['sharpe_sd']:.2f}")
        if not sep:
            print(f"         NOT SEPARABLE (needs > {band:.2f}). A target "
                  f"that cannot be predicted\n"
                  f"         scores the same as the real one, so Sharpe here "
                  f"carries no\n"
                  f"         information about predictive skill. Do not use "
                  f"it to choose\n"
                  f"         between configurations, and do not quote a "
                  f"single-seed value.")

    null = stats.get("noise")
    if real and null:
        sep, band = _distinguishable(real, null)
        print(f"\n  [INFO] real features vs pure noise:")
        print(f"         Sharpe   {real['sharpe']:+.2f} vs "
              f"{null['sharpe']:+.2f}"
              f"   ({'separable' if sep else 'NOT separable'})")
        print(f"         OOS corr {real['oos_corr']:+.4f} vs "
              f"{null['oos_corr']:+.4f}")
        if not sep:
            print("         Random features do as well as the real ones. "
                  "The P&L is not\n         coming from the kernels.")

    print()
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  Harness checks passed.")
    print("  These verify the machinery is honest. They do NOT say the")
    print("  strategy has edge — read the [INFO] lines for that, and")
    print("  POSTMORTEM.md for what they mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
