"""
End-to-End Pipeline Test with Synthetic Data
==============================================
Traders@SMU — Quantitative Strategies Group
Week 5: Backtesting Framework

Verifies the full pipeline before real data arrives:
    1. Generate synthetic mean-reverting price series (known OU params)
    2. Generate synthetic LOB snapshots + daily regime features
    3. Run walk-forward backtest → signal generation → metrics
    4. Check: recovered half-life ≈ true, DSR > 0, CKA < 1

Usage:
    python test_pipeline.py
"""

import sys
import os
import numpy as np

# Ensure both backtesting/ (this dir) and parent lob-kernel/ are on the path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from lob_kernel import (
    LOBSnapshot,
    BaseKernel,
    MaternRFF,
    RKHSLayer,
    VolumeProfileKernel,
    BookShapeKernel,
    DepthImbalanceKernel,
    generate_synthetic_lob,
)
from walk_forward import (
    PurgedWalkForward,
    MultiResolutionAligner,
    TwoLevelKernelCombiner,
    exponential_decay_weights,
    CPCVEvaluator,
    CKARedundancyGate,
)
from signal_definition import (
    OUSignalGenerator,
    estimate_ou_params,
    generate_positions,
    SignalDiagnostics,
)
from metrics import (
    deflated_sharpe_ratio,
    sortino_ratio,
    centered_kernel_alignment,
    effective_dimensionality,
    oos_negative_log_likelihood,
)
from hyperparameter_cv import (
    NestedCV,
    MKLOptimizer,
    ScalerWrapper,
    PurgedKFold,
    inner_cv_grid_search,
)


# ============================================================================
# Synthetic Data Generators
# ============================================================================

def generate_ou_price_series(
    n: int,
    kappa: float = 0.1,
    mu: float = 100.0,
    sigma: float = 0.5,
    dt: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic OU mean-reverting price series.

    dP = κ(μ - P)dt + σ dW
    """
    rng = np.random.RandomState(seed)
    prices = np.zeros(n)
    prices[0] = mu + rng.randn() * sigma

    for t in range(1, n):
        dW = rng.randn() * np.sqrt(dt)
        prices[t] = (
            prices[t - 1]
            + kappa * (mu - prices[t - 1]) * dt
            + sigma * dW
        )

    return prices


def generate_synthetic_lob_from_prices(
    prices: np.ndarray,
    n_levels: int = 5,
    tick_size: float = 0.01,
    seed: int = 42,
) -> list:
    """
    Generate LOBSnapshots consistent with a price series.

    Creates realistic-looking order book states where the mid-price
    tracks the given price series.
    """
    rng = np.random.RandomState(seed)
    n = len(prices)
    snapshots = []

    for t in range(n):
        mid = prices[t]
        spread = tick_size * (1 + rng.exponential(1.0))
        half_spread = spread / 2

        bid_prices = np.array([
            mid - half_spread - i * tick_size for i in range(n_levels)
        ])
        ask_prices = np.array([
            mid + half_spread + i * tick_size for i in range(n_levels)
        ])

        # Volume: higher near the spread, decaying outward
        base_vol = 100 + rng.exponential(200)
        bid_volumes = np.array([
            max(10, base_vol * np.exp(-0.3 * i) + rng.exponential(50))
            for i in range(n_levels)
        ])
        ask_volumes = np.array([
            max(10, base_vol * np.exp(-0.3 * i) + rng.exponential(50))
            for i in range(n_levels)
        ])

        snapshots.append(LOBSnapshot(
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            timestamp=float(t),
        ))

    return snapshots


def generate_synthetic_daily_features(
    n_days: int,
    n_features: int = 3,
    seed: int = 43,
) -> np.ndarray:
    """
    Generate synthetic daily regime features (VRP, GEX, Sentiment proxies).

    Returns array of shape (n_days, n_features).
    """
    rng = np.random.RandomState(seed)
    # Regime features: slowly varying with occasional jumps
    features = np.zeros((n_days, n_features))
    for f in range(n_features):
        features[0, f] = rng.randn()
        for t in range(1, n_days):
            # AR(1) process with occasional jumps
            features[t, f] = (
                0.95 * features[t - 1, f]
                + 0.1 * rng.randn()
                + (rng.rand() < 0.05) * rng.randn() * 2  # rare jumps
            )
    return features


# ============================================================================
# Minimal Kernel for Synthetic Features (no LOBSnapshot dependency)
# ============================================================================

class SyntheticFeatureKernel(BaseKernel):
    """
    A simple kernel that operates on raw feature arrays instead of
    LOBSnapshots. Useful for testing the pipeline with synthetic data.
    """

    def __init__(
        self,
        feature_dim: int = 3,
        nu: float = 1.5,
        length_scale: float = 1.0,
        n_rff: int = 200,
        reg_lambda: float = 1e-3,
        seed: int = 42,
        kernel_name: str = "SyntheticFeature",
    ):
        self.nu = nu
        self.length_scale = length_scale
        self.n_rff = n_rff
        self.reg_lambda = reg_lambda
        self.use_rff = True
        self._rff_seed = seed
        self._feat_dim = feature_dim
        self._kernel_name = kernel_name
        self.rff = None
        self._init_rff()

    def extract_features(self, snapshot) -> np.ndarray:
        return np.asarray(snapshot, dtype=np.float64)

    @property
    def name(self) -> str:
        return self._kernel_name

    @property
    def hyperparameters(self) -> dict:
        return {
            'nu': self.nu,
            'length_scale': self.length_scale,
            'n_rff': self.n_rff,
            'reg_lambda': self.reg_lambda,
        }

    @property
    def _feature_dim(self) -> int:
        return self._feat_dim


# ============================================================================
# Test Functions
# ============================================================================

def test_ou_estimation():
    """Test that OU parameter estimation recovers known parameters."""
    print("=" * 60)
    print("TEST 1: OU Parameter Estimation")
    print("=" * 60)

    true_kappa = 0.1
    true_mu = 100.0
    true_sigma = 0.5
    true_half_life = np.log(2) / true_kappa

    prices = generate_ou_price_series(
        n=5000, kappa=true_kappa, mu=true_mu, sigma=true_sigma, seed=42
    )

    params = estimate_ou_params(prices, dt=1.0)

    print(f"  True:      κ={true_kappa:.3f}, μ={true_mu:.1f}, "
          f"σ={true_sigma:.3f}, t½={true_half_life:.1f}")
    print(f"  Estimated: κ={params.kappa:.3f}, μ={params.mu:.1f}, "
          f"σ={params.sigma:.3f}, t½={params.half_life:.1f}")
    print(f"  Mean-reverting: {params.is_mean_reverting}")

    # Check within tolerance
    kappa_ok = abs(params.kappa - true_kappa) / true_kappa < 0.3
    hl_ok = abs(params.half_life - true_half_life) / true_half_life < 0.3
    print(f"  κ within 30%: {'PASS' if kappa_ok else 'FAIL'}")
    print(f"  t½ within 30%: {'PASS' if hl_ok else 'FAIL'}")
    print()
    return kappa_ok and hl_ok


def test_walk_forward_single_layer():
    """Test single-layer walk-forward backtest with synthetic data."""
    print("=" * 60)
    print("TEST 2: Single-Layer Walk-Forward Backtest")
    print("=" * 60)

    np.random.seed(42)
    n = 3000
    D_feat = 5

    # Synthetic features: AR(1) process
    X = np.zeros((n, D_feat))
    X[0] = np.random.randn(D_feat)
    for t in range(1, n):
        X[t] = 0.9 * X[t - 1] + 0.1 * np.random.randn(D_feat)

    # Target: nonlinear function of features + noise
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] + np.random.randn(n) * 0.5

    # Create kernel
    kernel = SyntheticFeatureKernel(feature_dim=D_feat, n_rff=200)

    # Run walk-forward
    engine = PurgedWalkForward(
        n_splits=5,
        embargo_bars=20,
        min_train_bars=500,
        decay_half_life=1000.0,
        reg_lambda=1e-2,
    )

    result = engine.run(X, y, kernel)

    print(f"  Folds: {result.n_folds}")
    print(f"  OOS predictions: {len(result.y_pred_oos)}")
    print(f"  OOS MSE: {result.oos_mse:.4f}")
    print(f"  OOS R²: {result.oos_r2:.4f}")

    for fr in result.fold_results:
        print(f"    Fold {fr.fold_idx}: train [{fr.train_start}:{fr.train_end}], "
              f"test [{fr.test_start}:{fr.test_end}], ||w||={fr.weights_norm:.3f}")

    ok = result.n_folds == 5 and len(result.y_pred_oos) > 0
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_two_level_walk_forward():
    """Test two-level (fast × slow) walk-forward backtest with 4 fast kernels."""
    print("=" * 60)
    print("TEST 3: Two-Level Walk-Forward Backtest (4 fast + 3 slow)")
    print("=" * 60)

    np.random.seed(42)
    n_bars = 2000
    n_days = 100
    bars_per_day = n_bars // n_days

    # Generate timestamps
    bar_timestamps = np.arange(n_bars, dtype=np.float64)
    daily_timestamps = np.arange(0, n_bars, bars_per_day, dtype=np.float64)

    # Fast features — 4 sub-kernels: LOB(4D) + VPIN(3D) + Lambda(3D) + VannaCharm(3D)
    D_fast = 4 + 3 + 3 + 3  # 13D combined
    X_fast = np.random.randn(n_bars, D_fast) * 0.5
    for t in range(1, n_bars):
        X_fast[t] += 0.8 * X_fast[t - 1]

    # Slow features (daily resolution)
    X_slow = generate_synthetic_daily_features(n_days, n_features=3, seed=43)

    # Target: fast signal + slow regime interaction
    daily_idx = np.searchsorted(daily_timestamps, bar_timestamps, side='right') - 1
    daily_idx = np.clip(daily_idx, 0, n_days - 1)
    y = (
        np.sin(X_fast[:, 0])
        + 0.3 * X_slow[daily_idx, 0] * X_fast[:, 1]
        + 0.2 * X_fast[:, 7]  # Lambda contribution
        + 0.1 * X_fast[:, 10]  # VannaCharm contribution
        + np.random.randn(n_bars) * 0.3
    )

    # Kernels
    fast_kernel = SyntheticFeatureKernel(
        feature_dim=D_fast, n_rff=150, seed=42, kernel_name="Fast"
    )
    slow_kernel = SyntheticFeatureKernel(
        feature_dim=3, n_rff=150, seed=43, kernel_name="Slow"
    )

    # Aligner and combiner
    aligner = MultiResolutionAligner(bar_timestamps, daily_timestamps)
    combiner = TwoLevelKernelCombiner(
        fast_layer=fast_kernel,
        slow_layer=slow_kernel,
        aligner=aligner,
        beta=1.0,
        product_dim=100,
        seed=42,
    )

    # Run two-level walk-forward
    engine = PurgedWalkForward(
        n_splits=4,
        embargo_bars=30,
        min_train_bars=400,
        decay_half_life=800.0,
        reg_lambda=1e-2,
    )

    result = engine.run_two_level(
        fast_data=X_fast,
        slow_data=X_slow,
        y=y,
        combiner=combiner,
    )

    print(f"  Folds: {result.n_folds}")
    print(f"  OOS MSE: {result.oos_mse:.4f}")
    print(f"  OOS R²: {result.oos_r2:.4f}")
    ok = result.n_folds > 0 and len(result.y_pred_oos) > 0
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_signal_generation():
    """Test signal generation and position sizing."""
    print("=" * 60)
    print("TEST 4: Signal Generation (s-score + positions)")
    print("=" * 60)

    np.random.seed(42)

    # Generate mean-reverting series
    prices = generate_ou_price_series(
        n=2000, kappa=0.05, mu=100.0, sigma=0.3, seed=42
    )

    # Create kernel and feature maps
    kernel = SyntheticFeatureKernel(feature_dim=1, n_rff=200)
    phi = kernel.feature_map(prices.reshape(-1, 1))

    # Generate signals
    gen = OUSignalGenerator(rolling_window=50, ou_window=500, ci_level=0.95)
    signal = gen.generate(phi, dt=1.0)

    print(f"  s-scores: {np.nanmin(signal.s_scores):.2f} to "
          f"{np.nanmax(signal.s_scores):.2f}")
    print(f"  OU κ: {signal.ou_params.kappa:.4f}")
    print(f"  OU half-life: {signal.ou_params.half_life:.1f}")
    print(f"  Mean-reverting: {signal.ou_params.is_mean_reverting}")

    # Generate positions
    positions = generate_positions(
        signal.s_scores, entry_threshold=1.5, exit_threshold=0.3
    )
    n_trades = np.sum(np.abs(np.diff(positions)) > 0.01)
    print(f"  Trades: {n_trades}")
    print(f"  Max position: {np.max(np.abs(positions)):.2f}")

    # Compute PnL (simple: position * price change)
    returns = np.diff(prices) / prices[:-1]
    pnl = positions[:-1] * returns
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0

    print(f"  Annualized Sharpe (raw): {sharpe:.2f}")

    ok = signal.ou_params.is_mean_reverting and not np.all(np.isnan(signal.s_scores))
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_metrics():
    """Test all metric computations."""
    print("=" * 60)
    print("TEST 5: Metrics")
    print("=" * 60)

    np.random.seed(42)

    # --- DSR ---
    dsr = deflated_sharpe_ratio(
        observed_sr=1.5, n_trials=20, T=252 * 3, skew=-0.3, kurtosis_excess=1.5
    )
    print(f"  DSR (SR=1.5, 20 trials, 3yr): {dsr:.3f}")

    # --- Sortino ---
    returns = np.random.randn(500) * 0.01 + 0.0003  # slight positive drift
    sort = sortino_ratio(returns, annualization=252.0)
    print(f"  Sortino: {sort:.2f}")

    # --- CKA ---
    n = 100
    X1 = np.random.randn(n, 50)
    X2 = np.random.randn(n, 50)
    K1 = X1 @ X1.T
    K2 = X2 @ X2.T
    K1_copy = K1.copy()
    cka_diff = centered_kernel_alignment(K1, K2)
    cka_same = centered_kernel_alignment(K1, K1_copy)
    print(f"  CKA (different): {cka_diff:.3f}")
    print(f"  CKA (same): {cka_same:.3f}")

    # --- Effective dimensionality ---
    K_full_rank = np.eye(50)
    K_low_rank = np.zeros((50, 50))
    K_low_rank[0, 0] = 1.0
    ed_full = effective_dimensionality(K_full_rank)
    ed_low = effective_dimensionality(K_low_rank)
    print(f"  Eff. dim (identity): {ed_full:.1f}")
    print(f"  Eff. dim (rank-1): {ed_low:.1f}")

    # --- OOS NLL ---
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    y_var = np.ones(100) * 0.1
    nll = oos_negative_log_likelihood(y_true, y_pred, y_var)
    print(f"  OOS NLL: {nll:.3f}")

    # Checks
    ok = (
        0 < dsr <= 1
        and sort > 0
        and 0 < cka_diff < cka_same
        and abs(cka_same - 1.0) < 0.01
        and ed_full > ed_low
    )
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_hyperparameter_cv():
    """Test inner CV grid search."""
    print("=" * 60)
    print("TEST 6: Hyperparameter CV (Inner Grid Search)")
    print("=" * 60)

    np.random.seed(42)
    n = 1000
    D_feat = 3

    X = np.random.randn(n, D_feat)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.randn(n) * 0.3

    kernel = SyntheticFeatureKernel(feature_dim=D_feat, n_rff=150)
    cv = PurgedKFold(n_folds=3, embargo=20)

    result = inner_cv_grid_search(
        kernel=kernel,
        data=X,
        y=y,
        length_scales=[0.5, 1.0, 2.0],
        reg_lambdas=[1e-3, 1e-2],
        cv=cv,
        metric_fn=lambda yt, yp: -float(np.mean((yt - yp) ** 2)),
    )

    print(f"  Best ℓ: {result.best_length_scale:.2f}")
    print(f"  Best λ: {result.best_reg_lambda:.4f}")
    print(f"  Best score (neg MSE): {result.best_score:.4f}")
    print(f"  Grid points evaluated: {len(result.all_scores)}")

    ok = (
        result.best_length_scale > 0
        and result.best_reg_lambda > 0
        and len(result.all_scores) == 6  # 3 * 2
    )
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_cpcv_pbo():
    """Test CPCV and PBO computation."""
    print("=" * 60)
    print("TEST 7: CPCV — Probability of Backtest Overfitting")
    print("=" * 60)

    np.random.seed(42)
    n = 2000
    D_feat = 3

    X = np.random.randn(n, D_feat)
    y = np.sin(X[:, 0]) + np.random.randn(n) * 0.5

    # Two strategies: one with good ℓ, one with bad ℓ
    kernel_good = SyntheticFeatureKernel(
        feature_dim=D_feat, n_rff=150, length_scale=1.0, seed=42,
        kernel_name="Good",
    )
    kernel_bad = SyntheticFeatureKernel(
        feature_dim=D_feat, n_rff=150, length_scale=0.01, seed=42,
        kernel_name="Bad",
    )

    evaluator = CPCVEvaluator(n_groups=6, n_test_groups=2, embargo_bars=20)

    pbo, details = evaluator.compute_pbo(
        data=X,
        y=y,
        strategy_kernels=[kernel_good, kernel_bad],
        metric_fn=lambda yt, yp: -float(np.mean((yt - yp) ** 2)),
        reg_lambda=1e-2,
    )

    print(f"  PBO: {pbo:.3f}")
    print(f"  Total splits: {details['n_splits']}")
    print(f"  Overfit splits: {details['n_overfit']}")

    ok = 0 <= pbo <= 1 and details['n_splits'] > 0
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_lob_kernel_integration():
    """Test with actual LOB kernels from lob_kernel.py."""
    print("=" * 60)
    print("TEST 8: LOB Kernel Integration")
    print("=" * 60)

    # Generate synthetic LOB data
    snapshots = generate_synthetic_lob(
        n_snapshots=1000, n_levels=5, base_price=100.0, seed=42
    )

    # Create LOB layer (aspect mode, RFF)
    vol_kernel = VolumeProfileKernel(
        n_levels=5, sigma=0.5, use_rff=True, nu=1.5, n_rff=200,
    )
    shape_kernel = BookShapeKernel(
        n_levels=5, sigma="auto", use_rff=True, nu=1.5, n_rff=200,
    )
    depth_kernel = DepthImbalanceKernel(
        n_levels=5, sigma=0.8, use_rff=True, nu=1.5, n_rff=200,
    )

    layer = RKHSLayer(
        kernels=[vol_kernel, shape_kernel, depth_kernel],
        weights=[1.0, 1.0, 0.8],
        name="LOB",
    )

    # Feature maps
    phi = layer.feature_map(snapshots)
    print(f"  Feature map shape: {phi.shape}")

    # Gram matrix for a subset
    K = layer.gram_matrix(snapshots[:100])
    is_psd, min_eig = layer.is_psd(snapshots[:100])
    print(f"  Gram matrix PSD: {is_psd} (min eig: {min_eig:.6f})")

    # Effective dimensionality
    ed = effective_dimensionality(K)
    print(f"  Effective dimensionality: {ed:.1f}")

    # CKA between sub-kernels
    K_vol = vol_kernel.gram_matrix(snapshots[:100])
    K_shape = shape_kernel.gram_matrix(snapshots[:100])
    K_depth = depth_kernel.gram_matrix(snapshots[:100])
    cka_vs = centered_kernel_alignment(K_vol, K_shape)
    cka_vd = centered_kernel_alignment(K_vol, K_depth)
    cka_sd = centered_kernel_alignment(K_shape, K_depth)
    print(f"  CKA (Vol, Shape): {cka_vs:.3f}")
    print(f"  CKA (Vol, Depth): {cka_vd:.3f}")
    print(f"  CKA (Shape, Depth): {cka_sd:.3f}")

    # Signal generation on LOB layer
    gen = OUSignalGenerator(rolling_window=50, ou_window=200)
    signal = gen.generate_from_kernel(layer, snapshots)
    print(f"  s-score range: [{np.nanmin(signal.s_scores):.2f}, "
          f"{np.nanmax(signal.s_scores):.2f}]")
    print(f"  OU half-life: {signal.ou_params.half_life:.1f}")

    # Walk-forward with LOB layer + synthetic targets
    mid_prices = np.array([s.mid_price for s in snapshots])
    returns = np.diff(mid_prices) / mid_prices[:-1]
    y = returns  # predict next return

    engine = PurgedWalkForward(
        n_splits=3,
        embargo_bars=10,
        min_train_bars=200,
        decay_half_life=500.0,
        reg_lambda=1e-2,
    )

    # Use pre-computed feature maps (faster for repeated fitting)
    result = engine.run(snapshots[:-1], y, layer)
    print(f"  Walk-forward OOS R²: {result.oos_r2:.4f}")

    # DSR
    oos_returns = result.y_pred_oos * result.y_true_oos  # signal * actual
    if np.std(oos_returns) > 0:
        raw_sr = np.mean(oos_returns) / np.std(oos_returns) * np.sqrt(252)
        dsr = deflated_sharpe_ratio(raw_sr, n_trials=1, T=len(oos_returns))
        print(f"  Raw Sharpe: {raw_sr:.2f}, DSR: {dsr:.3f}")
    else:
        print(f"  Sharpe: N/A (zero variance)")

    ok = is_psd and phi.shape == (1000, 600)  # 3 kernels * 200 RFF
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_cka_redundancy_gate():
    """Test the CKA redundancy gate for Kyle's Lambda vs VPIN."""
    print("=" * 60)
    print("TEST 9: CKA Redundancy Gate (K_Lambda vs K_VPIN)")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Case 1: Independent data → low CKA (different feature structure)
    X_lambda = np.random.randn(n, 3)
    X_vpin = np.random.randn(n, 3)  # truly independent features

    k_cand = SyntheticFeatureKernel(
        feature_dim=3, n_rff=200, seed=42, kernel_name="Lambda_indep",
    )
    k_ref = SyntheticFeatureKernel(
        feature_dim=3, n_rff=200, seed=43, kernel_name="VPIN_indep",
    )

    gate = CKARedundancyGate(threshold=0.5, n_sample=200, seed=42)
    keep_indep, cka_indep = gate.check(
        k_cand, k_ref,
        candidate_data=X_lambda,
        reference_data=X_vpin,
    )
    print(f"  Independent data: CKA={cka_indep:.3f}, keep={keep_indep}")

    # Case 2: Same data, same kernel → CKA ≈ 1.0 (redundant)
    k_same_a = SyntheticFeatureKernel(
        feature_dim=3, n_rff=200, seed=42, kernel_name="Lambda_same",
    )
    k_same_b = SyntheticFeatureKernel(
        feature_dim=3, n_rff=200, seed=42, kernel_name="VPIN_same",
    )
    keep_same, cka_same = gate.check(
        k_same_a, k_same_b,
        candidate_data=X_lambda,
        reference_data=X_lambda,   # same data → identical Gram
    )
    print(f"  Same-seed same-data: CKA={cka_same:.3f}, keep={keep_same}")

    # Case 3: gate_fast_kernels integration — Lambda uses same data as VPIN
    all_kernels = [
        SyntheticFeatureKernel(feature_dim=3, n_rff=200, seed=10, kernel_name="LOB"),
        SyntheticFeatureKernel(feature_dim=3, n_rff=200, seed=20, kernel_name="VPIN"),
        SyntheticFeatureKernel(feature_dim=3, n_rff=200, seed=20, kernel_name="Lambda"),
        SyntheticFeatureKernel(feature_dim=3, n_rff=200, seed=30, kernel_name="VannaCharm"),
    ]
    all_names = ["K_LOB", "K_VPIN", "K_Lambda", "K_VannaCharm"]
    # Lambda and VPIN use the same data (simulates collinear features)
    all_data = [
        np.random.randn(n, 3),   # LOB data
        X_lambda.copy(),          # VPIN data
        X_lambda.copy(),          # Lambda data = same as VPIN
        np.random.randn(n, 3),   # VannaCharm data
    ]

    kept_k, kept_n, report = gate.gate_fast_kernels(
        all_kernels, all_names, all_data,
        pairs_to_check=[(2, 1)],  # Lambda vs VPIN
    )
    print(f"  gate_fast_kernels: kept={kept_n}")
    print(f"    Checks: {report['checks']}")

    # Checks:
    # 1. Independent data → CKA should be low, kernel should be kept
    # 2. Same kernel + same data → CKA ≈ 1.0, should be dropped
    # 3. gate_fast_kernels should drop Lambda when it matches VPIN
    ok = (
        keep_indep is True           # independent data → keep
        and cka_same > 0.9           # same seed + same data → nearly identical
        and keep_same is False       # same → drop
        and "K_Lambda" not in kept_n  # Lambda should be dropped
        and len(kept_n) == 3         # LOB + VPIN + VannaCharm remain
    )
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_event_gate_on_total():
    """Test that event gate applies to entire K_total, not just product term."""
    print("=" * 60)
    print("TEST 10: Event Gate on K_total")
    print("=" * 60)

    np.random.seed(42)
    n_bars = 500
    n_days = 25
    bars_per_day = n_bars // n_days

    X_fast = np.random.randn(n_bars, 5) * 0.5
    for t in range(1, n_bars):
        X_fast[t] += 0.8 * X_fast[t - 1]
    X_slow = generate_synthetic_daily_features(n_days, n_features=3, seed=43)

    bar_timestamps = np.arange(n_bars, dtype=np.float64)
    daily_timestamps = np.arange(0, n_bars, bars_per_day, dtype=np.float64)

    fast_kernel = SyntheticFeatureKernel(feature_dim=5, n_rff=100, seed=42)
    slow_kernel = SyntheticFeatureKernel(feature_dim=3, n_rff=100, seed=43)
    aligner = MultiResolutionAligner(bar_timestamps, daily_timestamps)

    # Gate: zero out bars 100-110
    gate_mask = np.ones(n_bars)
    gate_mask[100:111] = 0.0

    combiner = TwoLevelKernelCombiner(
        fast_layer=fast_kernel, slow_layer=slow_kernel,
        aligner=aligner, beta=1.0, product_dim=50,
        event_gate=lambda idx: float(gate_mask[min(idx, n_bars - 1)]),
        seed=42,
    )

    bar_indices = np.arange(n_bars)
    phi_total = combiner.combined_feature_map(X_fast, X_slow, bar_indices)

    # Gated rows should be entirely zero (gate applied to phi_total)
    gated_norms = np.linalg.norm(phi_total[100:111], axis=1)
    ungated_norms = np.linalg.norm(phi_total[50:61], axis=1)

    all_gated_zero = np.all(gated_norms < 1e-12)
    all_ungated_nonzero = np.all(ungated_norms > 1e-3)

    print(f"  Gated rows (100-110) norm range: [{gated_norms.min():.6f}, {gated_norms.max():.6f}]")
    print(f"  Ungated rows (50-60) norm range: [{ungated_norms.min():.3f}, {ungated_norms.max():.3f}]")
    print(f"  All gated rows zero: {all_gated_zero}")
    print(f"  All ungated rows nonzero: {all_ungated_nonzero}")

    ok = all_gated_zero and all_ungated_nonzero
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_full_pipeline():
    """Full end-to-end pipeline test."""
    print("=" * 60)
    print("TEST 11: Full End-to-End Pipeline")
    print("=" * 60)

    np.random.seed(42)
    n_bars = 3000
    n_days = 150
    bars_per_day = n_bars // n_days

    # Step 1: Generate synthetic data with known OU dynamics
    true_kappa = 0.05
    true_half_life = np.log(2) / true_kappa
    prices = generate_ou_price_series(
        n=n_bars, kappa=true_kappa, mu=100.0, sigma=0.3
    )
    lob_snapshots = generate_synthetic_lob_from_prices(prices, n_levels=5)
    daily_features = generate_synthetic_daily_features(n_days, n_features=3)

    # Step 2: Create two-level kernel architecture
    fast_kernel = SyntheticFeatureKernel(
        feature_dim=10,  # LOB snapshot → 10D (volume, shape, depth)
        n_rff=200, kernel_name="FastLOB",
    )
    slow_kernel = SyntheticFeatureKernel(
        feature_dim=3, n_rff=150, seed=43, kernel_name="SlowRegime",
    )

    # Extract fast features from LOB snapshots
    vol_kernel = VolumeProfileKernel(n_levels=5, sigma=0.5, use_rff=True, nu=1.5, n_rff=200)
    X_fast = vol_kernel.feature_map(lob_snapshots)  # Use volume profile as fast features

    bar_timestamps = np.arange(n_bars, dtype=np.float64)
    daily_timestamps = np.arange(0, n_bars, bars_per_day, dtype=np.float64)

    aligner = MultiResolutionAligner(bar_timestamps, daily_timestamps)

    # Step 3: Signal generation
    gen = OUSignalGenerator(rolling_window=100, ou_window=500)
    signal = gen.generate(X_fast, dt=1.0)
    print(f"  OU params: κ={signal.ou_params.kappa:.4f}, "
          f"t½={signal.ou_params.half_life:.1f} "
          f"(true t½={true_half_life:.1f})")

    # Step 4: Generate positions and compute PnL
    positions = generate_positions(
        signal.s_scores, entry_threshold=1.5, exit_threshold=0.3
    )
    returns = np.diff(prices) / prices[:-1]
    pnl = positions[:-1] * returns
    valid_pnl = pnl[~np.isnan(pnl)]

    # Step 5: Compute metrics
    if len(valid_pnl) > 10 and np.std(valid_pnl) > 0:
        raw_sr = np.mean(valid_pnl) / np.std(valid_pnl) * np.sqrt(252)
        sort = sortino_ratio(valid_pnl, annualization=252.0)
        dsr = deflated_sharpe_ratio(
            raw_sr, n_trials=5, T=len(valid_pnl),
            skew=float(np.mean((valid_pnl - np.mean(valid_pnl)) ** 3) / np.std(valid_pnl) ** 3),
            kurtosis_excess=float(
                np.mean((valid_pnl - np.mean(valid_pnl)) ** 4) / np.std(valid_pnl) ** 4 - 3
            ),
        )
        print(f"  Raw Sharpe: {raw_sr:.2f}")
        print(f"  Sortino: {sort:.2f}")
        print(f"  DSR (5 trials): {dsr:.3f}")
    else:
        print(f"  Insufficient valid PnL data")

    # Step 6: Diagnostics
    acf1 = SignalDiagnostics.autocorrelation(signal.s_scores, lag=1)
    acf5 = SignalDiagnostics.autocorrelation(signal.s_scores, lag=5)
    print(f"  s-score ACF(1): {acf1:.3f}, ACF(5): {acf5:.3f}")

    ok = signal.ou_params.is_mean_reverting and len(valid_pnl) > 100
    print(f"  Result: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 60)
    print("RKHS Multi-Layer Kernel Model — Pipeline Tests")
    print("Traders@SMU — Backtesting Framework (updated architecture)")
    print("K_fast = LOB + VPIN + Lambda(CKA-gated) + VannaCharm")
    print("K_slow = VRP + GEX + Sentiment")
    print("=" * 60 + "\n")

    results = {}

    results['OU Estimation'] = test_ou_estimation()
    results['Single-Layer WF'] = test_walk_forward_single_layer()
    results['Two-Level WF'] = test_two_level_walk_forward()
    results['Signal Generation'] = test_signal_generation()
    results['Metrics'] = test_metrics()
    results['Hyperparameter CV'] = test_hyperparameter_cv()
    results['CPCV/PBO'] = test_cpcv_pbo()
    results['LOB Integration'] = test_lob_kernel_integration()
    results['CKA Redundancy Gate'] = test_cka_redundancy_gate()
    results['Event Gate K_total'] = test_event_gate_on_total()
    results['Full Pipeline'] = test_full_pipeline()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")

    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed")

    if n_pass == n_total:
        print("\n  All tests passed! Pipeline is ready for real data.\n")
    else:
        print("\n  Some tests failed. Review output above.\n")

    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
