"""
Diagnostic: Are RKHS kernels predictive at ANY horizon?

Tests each kernel's OOS predictions against forward returns at
multiple horizons (1, 5, 10, 20, 60 bars) to check if the signal
is predictive at a different timescale than we're testing.

Also checks: autocorrelation of predictions, information coefficient
stability over time, and whether predictions predict volatility
(regime) instead of direction.

Usage:
    .venv/bin/python diagnostic_predictiveness.py
"""

import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Auto-discovery must not pick up a falsification control run — sorting
# filenames used to land on "..._null-shuffle_time_seed2" and analyse
# deliberately destroyed features as if they were real.
from results_io import select_results_file

results_file = select_results_file(
    sys.argv[1] if len(sys.argv) > 1 else None,
    require=("y_pred_oos", "y_true_oos", "s_scores", "positions"),
)
print(f"Loading: {os.path.basename(results_file)}")
data = np.load(results_file, allow_pickle=True)

y_pred = data["y_pred_oos"]
y_true = data["y_true_oos"]
s_scores = data["s_scores"]
positions = data["positions"]
kernel_names = data["kernel_names"]

print(f"Kernel names: {list(kernel_names)}")
print(f"OOS samples: {len(y_pred)}")
print(f"y_true range: [{y_true.min():.6f}, {y_true.max():.6f}]")
print(f"y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")

# ── 1. Correlation at different lags ──
print("\n" + "=" * 60)
print("1. PREDICTION vs RETURNS AT DIFFERENT LAGS")
print("=" * 60)
print(f"  If correlation peaks at lag != 0, we're predicting the wrong horizon.\n")

for lag in [0, 1, 2, 5, 10, 20, 50]:
    if lag == 0:
        corr = np.corrcoef(y_pred, y_true)[0, 1]
    elif lag < len(y_pred):
        corr = np.corrcoef(y_pred[:-lag], y_true[lag:])[0, 1]
    else:
        corr = float('nan')
    print(f"  Lag {lag:3d} bars: corr = {corr:+.4f}")

# ── 2. Multi-horizon forward returns ──
print("\n" + "=" * 60)
print("2. PREDICTION vs MULTI-BAR FORWARD RETURNS")
print("=" * 60)
print(f"  Testing if signal predicts cumulative returns over longer windows.\n")

for horizon in [1, 5, 10, 20, 60]:
    if horizon < len(y_true):
        # Cumulative forward return
        fwd = np.zeros(len(y_true) - horizon)
        for i in range(len(fwd)):
            fwd[i] = y_true[i:i + horizon].sum()
        corr = np.corrcoef(y_pred[:len(fwd)], fwd)[0, 1]
        print(f"  {horizon:3d}-bar fwd return: corr = {corr:+.4f}")

# ── 3. Does the signal predict VOLATILITY instead of direction? ──
print("\n" + "=" * 60)
print("3. PREDICTION vs ABSOLUTE RETURNS (VOLATILITY)")
print("=" * 60)
print(f"  If |corr with |r|| > |corr with r|, signal predicts vol not direction.\n")

abs_ret = np.abs(y_true)
corr_direction = np.corrcoef(y_pred, y_true)[0, 1]
corr_vol = np.corrcoef(y_pred, abs_ret)[0, 1]
corr_vol_sq = np.corrcoef(y_pred, y_true ** 2)[0, 1]
corr_abs_pred_vol = np.corrcoef(np.abs(y_pred), abs_ret)[0, 1]

print(f"  pred vs return:        {corr_direction:+.4f}  (direction)")
print(f"  pred vs |return|:      {corr_vol:+.4f}  (volatility)")
print(f"  pred vs return²:       {corr_vol_sq:+.4f}  (variance)")
print(f"  |pred| vs |return|:    {corr_abs_pred_vol:+.4f}  (magnitude → vol)")

if abs(corr_vol) > abs(corr_direction) * 1.5:
    print(f"\n  ⚠️  Signal predicts VOLATILITY more than direction!")
    print(f"      This means the Kalman filter is detecting vol regimes,")
    print(f"      not directional edge. Consider using |pred| for sizing.")

# ── 4. Rolling IC stability ──
print("\n" + "=" * 60)
print("4. ROLLING INFORMATION COEFFICIENT (IC)")
print("=" * 60)
print(f"  If IC is unstable / sign-flipping, there's no consistent edge.\n")

window = 100
ics = []
for i in range(window, len(y_pred)):
    ic = np.corrcoef(y_pred[i - window:i], y_true[i - window:i])[0, 1]
    ics.append(ic)
ics = np.array(ics)

print(f"  Rolling IC (window={window}):")
print(f"    Mean:   {np.nanmean(ics):+.4f}")
print(f"    Std:    {np.nanstd(ics):.4f}")
print(f"    Min:    {np.nanmin(ics):+.4f}")
print(f"    Max:    {np.nanmax(ics):+.4f}")
print(f"    % > 0:  {(ics > 0).mean():.1%}")
print(f"    IC/std: {np.nanmean(ics) / (np.nanstd(ics) + 1e-8):.2f}  (> 0.5 is good)")

# ── 5. Autocorrelation of predictions ──
print("\n" + "=" * 60)
print("5. PREDICTION AUTOCORRELATION")
print("=" * 60)
print(f"  High autocorrelation = predictions are smooth/persistent,")
print(f"  which is what the Kalman filter exploits for regime detection.\n")

for lag in [1, 5, 10, 20, 50]:
    ac = np.corrcoef(y_pred[:-lag], y_pred[lag:])[0, 1]
    print(f"  AC(pred, lag={lag:2d}): {ac:+.4f}")

# ── 6. S-score vs forward returns ──
print("\n" + "=" * 60)
print("6. S-SCORE SIGNAL QUALITY")
print("=" * 60)

valid = ~np.isnan(s_scores)
s_valid = s_scores[valid]
y_valid = y_true[valid]

# Quintile analysis
n_q = len(s_valid) // 5
if n_q > 10:
    sorted_idx = np.argsort(s_valid)
    print(f"\n  Quintile analysis (sorted by s-score):")
    print(f"  {'Quintile':>10s}  {'Mean s':>8s}  {'Mean ret':>10s}  {'Hit rate':>10s}  {'N':>5s}")
    for q in range(5):
        q_idx = sorted_idx[q * n_q:(q + 1) * n_q]
        q_s = s_valid[q_idx]
        q_ret = y_valid[q_idx]
        q_hit = (q_ret > 0).mean()
        print(f"  {'Q' + str(q + 1) + ' (low)' if q == 0 else 'Q' + str(q + 1) + ' (high)' if q == 4 else 'Q' + str(q + 1):>10s}"
              f"  {q_s.mean():+8.3f}  {q_ret.mean():+10.6f}  {q_hit:10.1%}  {len(q_idx):5d}")

    # Monotonicity check
    q_means = [y_valid[sorted_idx[q * n_q:(q + 1) * n_q]].mean() for q in range(5)]
    monotonic = all(q_means[i] <= q_means[i + 1] for i in range(4))
    anti_monotonic = all(q_means[i] >= q_means[i + 1] for i in range(4))
    print(f"\n  Monotonic (Q1 < Q5)?      {'YES ✓' if monotonic else 'NO'}")
    print(f"  Anti-monotonic (Q1 > Q5)? {'YES (inverted signal)' if anti_monotonic else 'NO'}")
    if not monotonic and not anti_monotonic:
        print(f"  ⚠️  Non-monotonic → s-score has no consistent directional relationship")
        print(f"      with future returns. The signal is NOT predictive of direction.")

# ── 7. What IS profitable? Decompose the edge. ──
print("\n" + "=" * 60)
print("7. EDGE DECOMPOSITION")
print("=" * 60)

pos = positions[:len(y_true)]
active = pos != 0
if active.sum() > 0:
    strat_ret = pos[:-1] * y_true[1:]
    active_ret = strat_ret[pos[:-1] != 0]

    # When does the strategy trade?
    long_mask = pos[:-1] > 0
    short_mask = pos[:-1] < 0

    if long_mask.sum() > 0:
        long_ret = y_true[1:][long_mask].mean()
        long_n = long_mask.sum()
    else:
        long_ret = 0
        long_n = 0

    if short_mask.sum() > 0:
        short_ret = y_true[1:][short_mask].mean()
        short_n = short_mask.sum()
    else:
        short_ret = 0
        short_n = 0

    # Is the edge from timing (when to trade) or direction (which way)?
    all_mean_ret = y_true[1:].mean()
    active_mean_ret = y_true[1:][pos[:-1] != 0].mean() if (pos[:-1] != 0).sum() > 0 else 0

    print(f"  Mean return (all bars):     {all_mean_ret:+.6f}")
    print(f"  Mean return (active bars):  {active_mean_ret:+.6f}")
    print(f"  Timing edge:               {active_mean_ret - all_mean_ret:+.6f}")
    print(f"  Long bars: {long_n}, mean fwd ret: {long_ret:+.6f}")
    print(f"  Short bars: {short_n}, mean fwd ret: {short_ret:+.6f}")

    if abs(active_mean_ret) > abs(all_mean_ret) * 1.5:
        print(f"\n  → Edge is from TIMING (when to be in the market)")
    else:
        print(f"\n  → Edge is from DIRECTION or just market drift")

    # Is it just a momentum bias?
    # Check: does the strategy mostly go long during uptrends?
    if long_n > 0 and short_n > 0:
        print(f"\n  Long/Short ratio: {long_n}/{short_n} = {long_n / short_n:.1f}x")
        if long_n / (long_n + short_n) > 0.8:
            print(f"  ⚠️  Strategy is predominantly long — may just be capturing market drift")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"""
Key question: Is this RKHS pipeline actually predictive?

If answer is NO (all correlations near zero, non-monotonic quintiles):
  The profitable Sharpe comes from the Kalman filter detecting
  WHEN the ElasticNet's walk-forward dynamics shift — a regime
  detection mechanism, not a return prediction mechanism.

  Options:
  a) Accept it as a regime detector and optimize for that
     (focus on Kalman params, warmup, position sizing)
  b) Replace the return prediction target with a regime label
     (e.g., predict realized vol quintile, trend strength, etc.)
  c) Use kernels for risk management / position sizing only,
     and get directional signal from a different model
""")
