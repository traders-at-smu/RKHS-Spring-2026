"""
MKL Optimizer — learns kernel weights α and β from data.
Re-runs the backtest with optimized weights.

Usage:
    python3 run_mkl_optimize.py
"""

import sys, os, time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "kernels"))
sys.path.insert(0, os.path.join(_ROOT, "backtesting"))

import pandas as pd
from data_loader import DataLoader
from kernels.LOB import MaternRFF
from kernels.momentum_gate import MomentumGate
from backtesting.walk_forward import (
    PurgedWalkForward, MultiResolutionAligner, TwoLevelKernelCombiner,
)
from backtesting.hyperparameter_cv import MKLOptimizer, PurgedKFold, inner_cv_grid_search
from backtesting.signal_definition import estimate_ou_params, generate_positions
from backtesting.metrics import deflated_sharpe_ratio, sortino_ratio

# Import the feature builders from run_backtest
from run_backtest import (
    build_vpin_features, build_kyle_features,
    build_vrp_features, build_macro_features,
    FeatureMatrixKernel, CombinedKernel,
)


def neg_mse(y_true, y_pred):
    """Higher is better metric for CV."""
    return -float(np.mean((y_true - y_pred) ** 2))


def main():
    t0 = time.time()
    print("=" * 70)
    print("  MKL Hyperparameter Optimization")
    print("=" * 70)

    # ── Load data (same as run_backtest) ─────────────────────────────────
    print("\n[1/4] Loading data and building features...")
    dl = DataLoader(os.path.join(_ROOT, "data"))
    dates = dl.trading_dates()
    daily = dl.build_daily_ohlcv()

    ext = os.path.join(_ROOT, "data", "external")
    vix_df = pd.read_parquet(os.path.join(ext, "vix_daily.parquet"))
    spy_df = pd.read_parquet(os.path.join(ext, "spy_daily.parquet"))
    xle_df = pd.read_parquet(os.path.join(ext, "xle_daily.parquet"))
    pcr_df = pd.read_parquet(os.path.join(ext, "put_call_ratio.parquet"))

    # LOB features
    lob_feat_list = []
    for d in dates:
        try:
            feat = dl.lob_features(d)
            vp, bs, di = feat.get("volume_profile"), feat.get("book_shape"), feat.get("depth_imbalance")
            if vp is not None and bs is not None and di is not None:
                n_bars = min(vp.shape[0], bs.shape[0], di.shape[0])
                lob_feat_list.append(np.concatenate([vp[:n_bars], bs[:n_bars], di[:n_bars]], axis=1))
        except Exception:
            continue
    lob_all = np.concatenate(lob_feat_list, axis=0)

    trades = dl.trades()
    vpin_feats = build_vpin_features(trades)
    kyle_feats = build_kyle_features(trades)
    vrp_feats = build_vrp_features(daily, vix_df, pcr_df)
    macro_feats = build_macro_features(daily, spy_df, xle_df)

    n_fast = min(lob_all.shape[0], vpin_feats.shape[0], kyle_feats.shape[0])
    n_slow = min(vrp_feats.shape[0], macro_feats.shape[0])

    lob_fast = lob_all[:n_fast]
    vpin_fast = vpin_feats[:n_fast]
    kyle_fast = kyle_feats[:n_fast]
    vrp_slow = vrp_feats[:n_slow]
    macro_slow = macro_feats[:n_slow]

    print(f"  Fast bars: {n_fast}, Slow days: {n_slow}")

    # ── Build per-kernel RFF feature maps ────────────────────────────────
    print("\n[2/4] Computing per-kernel feature maps for MKL...")

    length_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    reg_lambdas = [1e-4, 1e-3, 1e-2, 0.1]

    # Use a subsample for inner CV (full LOB is too large)
    n_sub = min(3000, n_fast)
    rng = np.random.RandomState(42)
    sub_idx = np.sort(rng.choice(n_fast, n_sub, replace=False))

    lob_sub = lob_fast[sub_idx]
    vpin_sub = vpin_fast[sub_idx]
    kyle_sub = kyle_fast[sub_idx]

    # Targets
    daily_close = daily["close"].values.astype(float)
    bars_per_day = n_fast // len(dates)
    bar_timestamps = np.arange(n_fast, dtype=float)
    daily_timestamps = np.arange(0, n_fast, bars_per_day, dtype=float)[:n_slow]
    aligner = MultiResolutionAligner(bar_timestamps, daily_timestamps)
    bar_daily_idx = aligner.get_daily_indices(np.arange(n_fast))
    bar_prices = daily_close[np.clip(bar_daily_idx, 0, len(daily_close) - 1)]
    y_all = np.diff(np.log(bar_prices + 1e-8), prepend=np.log(bar_prices[0]))
    y_all = np.roll(y_all, -1)
    y_all[-1] = 0
    y_sub = y_all[sub_idx]

    cv = PurgedKFold(n_folds=5, embargo=50)

    # Inner CV for each kernel's length scale
    kernels_to_tune = [
        ("LOB",   lob_sub,  50),
        ("VPIN",  vpin_sub, 4),
        ("Kyle",  kyle_sub, 4),
    ]

    best_params = {}
    for name, data, d_in in kernels_to_tune:
        print(f"\n  Tuning {name} (d={d_in})...")
        kernel = FeatureMatrixKernel(data, nu=1.5, length_scale=1.0,
                                     n_rff=256, name=name, seed=42)
        result = inner_cv_grid_search(
            kernel=kernel, data=data, y=y_sub,
            length_scales=length_scales,
            reg_lambdas=reg_lambdas,
            cv=cv, metric_fn=neg_mse,
        )
        best_params[name] = {
            "length_scale": result.best_length_scale,
            "reg_lambda": result.best_reg_lambda,
            "score": result.best_score,
        }
        print(f"    Best: ell={result.best_length_scale}, "
              f"lam={result.best_reg_lambda:.1e}, "
              f"neg_MSE={result.best_score:.8f}")

    # Slow kernels
    for name, data, d_in in [("VRP", vrp_slow, 10), ("Macro", macro_slow, 6)]:
        print(f"\n  Tuning {name} (d={d_in})...")
        n_slow_sub = min(300, len(data))
        slow_sub_idx = np.sort(rng.choice(len(data), n_slow_sub, replace=False))
        data_sub = data[slow_sub_idx]
        y_slow_sub = y_all[:n_slow][slow_sub_idx] if n_slow <= len(y_all) else np.zeros(n_slow_sub)

        kernel = FeatureMatrixKernel(data_sub, nu=1.5, length_scale=1.0,
                                     n_rff=256, name=name, seed=42)
        slow_cv = PurgedKFold(n_folds=3, embargo=10)
        result = inner_cv_grid_search(
            kernel=kernel, data=data_sub, y=y_slow_sub,
            length_scales=length_scales,
            reg_lambdas=reg_lambdas,
            cv=slow_cv, metric_fn=neg_mse,
        )
        best_params[name] = {
            "length_scale": result.best_length_scale,
            "reg_lambda": result.best_reg_lambda,
            "score": result.best_score,
        }
        print(f"    Best: ell={result.best_length_scale}, "
              f"lam={result.best_reg_lambda:.1e}, "
              f"neg_MSE={result.best_score:.8f}")

    # ── MKL weight optimization ──────────────────────────────────────────
    print("\n[3/4] Running MKL weight optimization...")

    # Build feature maps with tuned length scales
    fast_phis = []
    for name, data, seed in [("LOB", lob_sub, 42), ("VPIN", vpin_sub, 43),
                              ("Kyle", kyle_sub, 44)]:
        ls = best_params[name]["length_scale"]
        k = FeatureMatrixKernel(data, nu=1.5, length_scale=ls,
                                n_rff=256, name=name, seed=seed)
        fast_phis.append(k.feature_map(data))

    slow_phis = []
    for name, data, seed in [("VRP", vrp_slow[:n_slow], 45),
                              ("Macro", macro_slow[:n_slow], 46)]:
        ls = best_params[name]["length_scale"]
        k = FeatureMatrixKernel(data, nu=1.5, length_scale=ls,
                                n_rff=256, name=name, seed=seed)
        slow_phis.append(k.feature_map(data))

    # Align slow to fast
    sub_daily_idx = aligner.get_daily_indices(sub_idx)
    slow_phis_aligned = []
    for phi_slow in slow_phis:
        aligned = phi_slow[np.clip(sub_daily_idx, 0, phi_slow.shape[0] - 1)]
        slow_phis_aligned.append(aligned)

    mkl = MKLOptimizer(
        n_fast_kernels=3, n_slow_kernels=2,
        l2_penalty=0.01, lr=0.01, max_iter=200,
    )

    mkl_result = mkl.optimize(
        fast_feature_maps=fast_phis,
        y=y_sub,
        reg_lambda=best_params["LOB"]["reg_lambda"],
        slow_feature_maps=slow_phis_aligned,
        verbose=True,
    )

    print(f"\n  Optimized Weights:")
    print(f"    α_LOB:   {mkl.alpha_fast[0]:.4f}")
    print(f"    α_VPIN:  {mkl.alpha_fast[1]:.4f}")
    print(f"    α_Kyle:  {mkl.alpha_fast[2]:.4f}")
    print(f"    α_VRP:   {mkl.alpha_slow[0]:.4f}")
    print(f"    α_Macro: {mkl.alpha_slow[1]:.4f}")
    print(f"    β:       {mkl.beta:.4f}")
    final_loss = mkl_result["loss_history"][-1] if mkl_result["loss_history"] else float("nan")
    print(f"    Final loss: {final_loss:.8f}")

    # ── Re-run backtest with optimized params ────────────────────────────
    print("\n[4/4] Re-running backtest with optimized parameters...")

    lob_kernel = FeatureMatrixKernel(
        lob_fast, nu=1.5,
        length_scale=best_params["LOB"]["length_scale"],
        n_rff=500, name="LOB", seed=42,
    )
    vpin_kernel = FeatureMatrixKernel(
        vpin_fast, nu=1.5,
        length_scale=best_params["VPIN"]["length_scale"],
        n_rff=256, name="VPIN", seed=43,
    )
    kyle_kernel = FeatureMatrixKernel(
        kyle_fast, nu=1.5,
        length_scale=best_params["Kyle"]["length_scale"],
        n_rff=256, name="Kyle", seed=44,
    )
    vrp_kernel = FeatureMatrixKernel(
        vrp_slow, nu=1.5,
        length_scale=best_params["VRP"]["length_scale"],
        n_rff=256, name="VRP", seed=45,
    )
    macro_kernel = FeatureMatrixKernel(
        macro_slow, nu=1.5,
        length_scale=best_params["Macro"]["length_scale"],
        n_rff=256, name="Macro", seed=46,
    )

    d_lob = lob_fast.shape[1]
    d_vpin = vpin_fast.shape[1]
    d_kyle = kyle_fast.shape[1]
    d_vrp = vrp_slow.shape[1]
    d_macro = macro_slow.shape[1]

    fast_layer = CombinedKernel(
        [lob_kernel, vpin_kernel, kyle_kernel],
        [slice(0, d_lob), slice(d_lob, d_lob + d_vpin),
         slice(d_lob + d_vpin, d_lob + d_vpin + d_kyle)],
        [mkl.alpha_fast[0], mkl.alpha_fast[1], mkl.alpha_fast[2]],
    )
    slow_layer = CombinedKernel(
        [vrp_kernel, macro_kernel],
        [slice(0, d_vrp), slice(d_vrp, d_vrp + d_macro)],
        [mkl.alpha_slow[0], mkl.alpha_slow[1]],
    )

    # Momentum gate
    cl_daily = daily_close[:n_slow]
    mg = MomentumGate(short_window=20, long_window=60, vol_window=20,
                      n_rff=256, random_state=42)
    lookback = max(mg.long_window, mg.vol_window)
    log_r = np.diff(np.log(cl_daily))
    M_gate = len(cl_daily) - 1 - lookback
    target_gate = log_r[lookback:lookback + M_gate]
    mg.fit(cl_daily[:lookback + M_gate + 1], target_gate)

    combiner = TwoLevelKernelCombiner(
        fast_layer=fast_layer,
        slow_layer=slow_layer,
        aligner=aligner,
        beta=mkl.beta,
        product_dim=500,
        momentum_gate=mg,
        daily_prices=cl_daily,
        seed=42,
    )

    fast_combined = np.concatenate([lob_fast, vpin_fast, kyle_fast], axis=1)
    slow_combined = np.concatenate([vrp_slow, macro_slow], axis=1)

    opt_reg = min(best_params[n]["reg_lambda"] for n in best_params)

    engine = PurgedWalkForward(
        n_splits=8,
        embargo_bars=100,
        min_train_bars=max(2000, n_fast // 5),
        decay_half_life=3000.0,
        reg_lambda=opt_reg,
    )

    result = engine.run_two_level(
        fast_data=fast_combined,
        slow_data=slow_combined,
        y=y_all,
        combiner=combiner,
        bar_indices=np.arange(n_fast),
    )

    # Signal generation
    pred = result.y_pred_oos
    roll_w = min(100, len(pred) // 3)
    s_scores = np.zeros(len(pred))
    for t in range(roll_w, len(pred)):
        window = pred[t - roll_w:t]
        mu_w, sig_w = window.mean(), window.std() + 1e-8
        s_scores[t] = (pred[t] - mu_w) / sig_w

    positions = generate_positions(s_scores, entry_threshold=1.25,
                                   exit_threshold=0.5)
    strat_returns = positions[:-1] * result.y_true_oos[1:]
    cum_pnl = np.cumsum(strat_returns)

    sr = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)
    sortino = sortino_ratio(strat_returns)
    max_dd = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl) if len(cum_pnl) > 0 else 0
    hit_rate = np.mean(strat_returns > 0) if len(strat_returns) > 0 else 0

    print(f"\n  Optimized Strategy Performance:")
    print(f"    OOS R²:       {result.oos_r2:.6f}")
    print(f"    Sharpe Ratio:  {sr:.4f}")
    print(f"    Sortino Ratio: {sortino:.4f}")
    print(f"    Max Drawdown:  {max_dd:.6f}")
    print(f"    Hit Rate:      {hit_rate:.2%}")
    print(f"    Total P&L:     {cum_pnl[-1]:.6f}")

    # Save optimized results
    np.savez(
        os.path.join(_ROOT, "backtest_results_optimized.npz"),
        y_pred_oos=result.y_pred_oos,
        y_true_oos=result.y_true_oos,
        oos_indices=result.oos_indices,
        s_scores=s_scores,
        positions=positions,
        strat_returns=strat_returns,
        cum_pnl=cum_pnl,
        alpha_fast=mkl.alpha_fast,
        alpha_slow=mkl.alpha_slow,
        beta=np.array([mkl.beta]),
    )

    # Save params summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Optimization complete in {elapsed:.1f}s")
    print(f"  Results saved to backtest_results_optimized.npz")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
