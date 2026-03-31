"""
Walk-Forward Backtesting Engine
================================
Traders@SMU — Quantitative Strategies Group
Week 5: Backtesting Framework

Architecture:
    K_total = K_fast + β · K_fast · K_slow_gated
    K_fast  = α₁·K_LOB + α₂·K_VPIN + α₃·K_Lambda + α₄·K_VannaCharm
              (dollar bar resolution)
    K_slow  = α₅·K_VRP + α₆·K_GEX + α₇·K_Sentiment
              (daily resolution)
    K_slow_gated[i,j] = sqrt(gate[i] · gate[j]) · K_slow[i,j]
              where gate(t) ∈ (0,1) from MomentumGate (Arjan)
    Gate    = Event Proximity (optional multiplicative 0/1 mask on K_total)

    Note: K_Lambda includes a CKA redundancy gate against K_VPIN —
          if CKA(K_Lambda, K_VPIN) >= 0.5, K_Lambda is dropped to avoid
          collinearity in the MKL.
    Note: MomentumGate attenuates the slow kernel in choppy regimes
          using TSMOM / vol-scaled momentum / EMA cross signals.

This module provides:
    1. PurgedWalkForward  — expanding-window walk-forward with embargo gap
    2. CPCVEvaluator      — Combinatorial Purged Cross-Validation for PBO
    3. MultiResolutionAligner — aligns fast (bar) and slow (daily) data
    4. CKARedundancyGate  — runtime CKA check to gate redundant kernels

Design principles:
    - RFF frequencies are sampled ONCE and frozen; only linear KRR weights
      retrain at each walk-forward step → O(nD) per step, not O(n³).
    - Expanding window with exponential decay on older observations.
    - Purged train/test split with embargo gap prevents lookahead bias.
    - The two-level (fast × slow) multiplicative kernel combination is
      handled via feature-space tensor product approximation.

References:
    - de Prado (2018), "Advances in Financial Machine Learning", Ch. 7–12
    - Bailey et al. (2017), "The Probability of Backtest Overfitting"
"""

import numpy as np
from typing import (
    List, Optional, Tuple, Dict, Any, Callable, NamedTuple, Protocol,
)
from itertools import combinations
from dataclasses import dataclass, field
import warnings


# ============================================================================
# Protocols for kernel layers (duck-typing with existing BaseKernel/RKHSLayer)
# ============================================================================

class KernelLike(Protocol):
    """Structural type matching BaseKernel and RKHSLayer."""

    def feature_map(self, snapshots) -> np.ndarray: ...
    def fit(self, snapshots, y, **kwargs) -> None: ...
    def predict(self, snapshots) -> np.ndarray: ...


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    y_pred: np.ndarray
    y_true: np.ndarray
    weights_norm: float  # ||w||₂ for regularization diagnostics
    train_loss: float    # in-sample weighted MSE


@dataclass
class BacktestResult:
    """Aggregate result from the full walk-forward backtest."""
    fold_results: List[WalkForwardResult]
    y_pred_oos: np.ndarray      # concatenated OOS predictions
    y_true_oos: np.ndarray      # concatenated OOS targets
    oos_indices: np.ndarray     # indices into the original data array
    n_folds: int

    @property
    def oos_mse(self) -> float:
        return float(np.mean((self.y_pred_oos - self.y_true_oos) ** 2))

    @property
    def oos_r2(self) -> float:
        ss_res = np.sum((self.y_true_oos - self.y_pred_oos) ** 2)
        ss_tot = np.sum((self.y_true_oos - np.mean(self.y_true_oos)) ** 2)
        if ss_tot < 1e-12:
            return 0.0
        return float(1.0 - ss_res / ss_tot)


# ============================================================================
# Multi-Resolution Aligner
# ============================================================================

class MultiResolutionAligner:
    """
    Aligns fast (bar-level) and slow (daily) features for the two-level
    kernel architecture.

    Each bar has a timestamp; each daily observation has a date.
    This class maps each bar index → the corresponding daily index so that
    φ_slow can be looked up for the multiplicative combination:

        φ_total = [φ_fast, √β · (φ_fast ⊗ᵣ φ_slow)]

    where ⊗ᵣ is a rank-r tensor sketch approximation of the outer product.

    Parameters
    ----------
    bar_timestamps : array-like, shape (n_bars,)
        Timestamp (epoch seconds or any monotone float) per bar.
    daily_timestamps : array-like, shape (n_days,)
        Timestamp per daily observation (e.g., market close).
    """

    def __init__(
        self,
        bar_timestamps: np.ndarray,
        daily_timestamps: np.ndarray,
    ):
        self.bar_ts = np.asarray(bar_timestamps, dtype=np.float64)
        self.daily_ts = np.asarray(daily_timestamps, dtype=np.float64)
        # Pre-compute mapping: bar_i → daily_j where daily_j is the most
        # recent daily observation at or before bar_i's timestamp.
        self._bar_to_daily = self._build_mapping()

    def _build_mapping(self) -> np.ndarray:
        """Map each bar to its corresponding daily index via searchsorted."""
        # searchsorted 'right' gives first daily_ts > bar_ts, minus 1 = last <=
        indices = np.searchsorted(self.daily_ts, self.bar_ts, side='right') - 1
        # Bars before the first daily observation get index 0
        indices = np.clip(indices, 0, len(self.daily_ts) - 1)
        return indices

    def get_daily_indices(self, bar_indices: np.ndarray) -> np.ndarray:
        """Return the daily indices corresponding to the given bar indices."""
        return self._bar_to_daily[bar_indices]

    def align_features(
        self,
        fast_features: np.ndarray,
        slow_features: np.ndarray,
        bar_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Align slow features to match fast features at bar resolution.

        Parameters
        ----------
        fast_features : array, shape (n_bars_subset, D_fast)
        slow_features : array, shape (n_days, D_slow)
        bar_indices : array, shape (n_bars_subset,)
            Which bars from the original array we're aligning.

        Returns
        -------
        slow_aligned : array, shape (n_bars_subset, D_slow)
        """
        daily_idx = self.get_daily_indices(bar_indices)
        return slow_features[daily_idx]


# ============================================================================
# Two-Level Kernel Combiner
# ============================================================================

class TwoLevelKernelCombiner:
    """
    Combines fast and slow kernel layers via the architecture:

        K_total = K_fast + β · K_fast · K_slow_gated

    where K_slow_gated is the slow kernel attenuated by Arjan's
    MomentumGate in choppy / mean-reverting regimes:

        K_slow_gated[i,j] = sqrt(gate[i] · gate[j]) · K_slow[i,j]

    In feature space, the momentum gate scales φ_slow before the
    tensor sketch:

        φ_slow_gated = sqrt(gate) ⊙ φ_slow   (element-wise)
        φ_product ≈ TensorSketch(φ_fast, φ_slow_gated)
        φ_total = [φ_fast, √β · φ_product]

    If an event gate is provided, it is applied to the ENTIRE φ_total
    (both the additive fast term and the product cross-term), acting
    as a multiplicative mask on K_total rather than just the cross-term.

    Fast layer (dollar bar resolution):
        K_fast = α₁·K_LOB + α₂·K_VPIN + α₃·K_Lambda + α₄·K_VannaCharm
    Slow layer (daily resolution):
        K_slow = α₅·K_VRP + α₆·K_GEX + α₇·K_Sentiment

    Parameters
    ----------
    fast_layer : KernelLike
        Fast microstructure kernel (LOB + VPIN + Lambda + Vanna/Charm).
    slow_layer : KernelLike
        Slow regime kernel (VRP + GEX + Sentiment at daily resolution).
    aligner : MultiResolutionAligner
        Maps bar indices to daily indices.
    beta : float
        Multiplicative weight for the cross-term. Learned via MKL.
    product_dim : int
        Dimensionality of the tensor sketch approximation.
    event_gate : optional callable
        Maps bar index → 0 or 1 (event proximity mask on K_total).
    momentum_gate : optional MomentumGate
        Arjan's regime gate. If provided and fitted, attenuates φ_slow
        in choppy regimes before the tensor sketch.
    daily_prices : optional np.ndarray
        Daily close prices for the momentum gate. Required if
        momentum_gate is provided.
    """

    def __init__(
        self,
        fast_layer: KernelLike,
        slow_layer: KernelLike,
        aligner: MultiResolutionAligner,
        beta: float = 1.0,
        product_dim: int = 500,
        event_gate: Optional[Callable[[int], float]] = None,
        momentum_gate=None,
        daily_prices: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.fast_layer = fast_layer
        self.slow_layer = slow_layer
        self.aligner = aligner
        self.beta = beta
        self.product_dim = product_dim
        self.event_gate = event_gate
        self.momentum_gate = momentum_gate
        self.daily_prices = daily_prices
        self._rng = np.random.RandomState(seed)
        self._sketch_initialized = False

    def _init_tensor_sketch(self, D_fast: int, D_slow: int):
        """
        Initialize count-sketch hash functions for TensorSketch.

        TensorSketch approximates the tensor product φ_fast ⊗ φ_slow
        in O(D_product * log(D_product)) via FFT-based polynomial
        multiplication of count sketches.
        """
        M = self.product_dim
        # Hash functions: map each feature dimension → a bucket in [0, M)
        self._h1 = self._rng.randint(0, M, size=D_fast)
        self._h2 = self._rng.randint(0, M, size=D_slow)
        # Sign functions: ±1 Rademacher variables
        self._s1 = self._rng.choice([-1, 1], size=D_fast).astype(np.float64)
        self._s2 = self._rng.choice([-1, 1], size=D_slow).astype(np.float64)
        self._sketch_initialized = True

    def _tensor_sketch(
        self,
        phi_fast: np.ndarray,
        phi_slow: np.ndarray,
    ) -> np.ndarray:
        """
        Approximate tensor product via TensorSketch (Pham & Pagh, 2013).

        Input:  φ_fast (n, D_fast), φ_slow (n, D_slow)
        Output: ψ (n, M) approximating the element-wise kernel product.
        """
        n = phi_fast.shape[0]
        M = self.product_dim

        if not self._sketch_initialized:
            self._init_tensor_sketch(phi_fast.shape[1], phi_slow.shape[1])

        # Count sketch for fast features
        sketch1 = np.zeros((n, M))
        for j in range(phi_fast.shape[1]):
            sketch1[:, self._h1[j]] += self._s1[j] * phi_fast[:, j]

        # Count sketch for slow features
        sketch2 = np.zeros((n, M))
        for j in range(phi_slow.shape[1]):
            sketch2[:, self._h2[j]] += self._s2[j] * phi_slow[:, j]

        # Polynomial multiplication via FFT
        fft1 = np.fft.rfft(sketch1, axis=1)
        fft2 = np.fft.rfft(sketch2, axis=1)
        result = np.fft.irfft(fft1 * fft2, n=M, axis=1)

        return result

    def combined_feature_map(
        self,
        fast_data,
        slow_data,
        bar_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Build the combined feature map:
            φ_total = [φ_fast, √β · TensorSketch(φ_fast, φ_slow_aligned)]

        If event_gate is set, rows where gate=0 have the product term zeroed.

        Parameters
        ----------
        fast_data : list-like
            Data for fast_layer.feature_map() (LOB snapshots / feature arrays).
        slow_data : list-like
            Data for slow_layer.feature_map() (daily regime observations).
        bar_indices : array, shape (n,)
            Indices into the original bar array for alignment.

        Returns
        -------
        Phi_total : array, shape (n, D_fast + product_dim)
        """
        phi_fast = self.fast_layer.feature_map(fast_data)  # (n, D_fast)

        # Align slow features to bar resolution
        phi_slow_full = self.slow_layer.feature_map(slow_data)  # (n_days, D_slow)
        phi_slow = self.aligner.align_features(
            phi_fast, phi_slow_full, bar_indices
        )  # (n, D_slow)

        # Apply momentum gate to slow features (Arjan's regime gating)
        # gate(t) ∈ (0,1): ~1 in trending regimes, ~0 in choppy regimes
        # φ_slow_gated = sqrt(gate) ⊙ φ_slow  (preserves PSD via Schur)
        if (self.momentum_gate is not None
                and self.daily_prices is not None
                and getattr(self.momentum_gate, '_fitted', False)):
            daily_idx = self.aligner.get_daily_indices(bar_indices)
            gate_daily = self.momentum_gate.gate_values(self.daily_prices)
            # Map daily gate values to bar resolution
            lookback = max(self.momentum_gate.long_window,
                           self.momentum_gate.vol_window)
            gate_at_bars = np.ones(len(bar_indices))
            for i, di in enumerate(daily_idx):
                gi = di - lookback
                if 0 <= gi < len(gate_daily):
                    gate_at_bars[i] = gate_daily[gi]
            phi_slow = phi_slow * np.sqrt(gate_at_bars)[:, np.newaxis]

        # Tensor sketch approximation of the product kernel
        phi_product = self._tensor_sketch(phi_fast, phi_slow)  # (n, M)

        # Combine: [φ_fast, √β · φ_product]
        phi_total = np.concatenate([
            phi_fast,
            np.sqrt(max(self.beta, 0.0)) * phi_product,
        ], axis=1)

        # Apply event gate to the ENTIRE φ_total (both additive fast
        # and product cross-term), so the gate acts on K_total.
        # Previously the gate was on the product term only; Anders
        # reclassified event proximity as a gate on the final kernel.
        if self.event_gate is not None:
            gate_mask = np.array([
                self.event_gate(int(idx)) for idx in bar_indices
            ], dtype=np.float64)
            phi_total *= gate_mask[:, np.newaxis]

        return phi_total

    def fit(
        self,
        fast_data_train,
        slow_data_all,
        bar_indices_train: np.ndarray,
        y_train: np.ndarray,
        reg_lambda: float = 1e-3,
        sample_weights: Optional[np.ndarray] = None,
    ):
        """
        Fit KRR on the combined two-level feature map.

        With sample weights W = diag(w_i):
            w* = (Φᵀ W Φ + λI)⁻¹ Φᵀ W y

        Parameters
        ----------
        fast_data_train : list-like
            Fast-resolution training data.
        slow_data_all : list-like
            ALL slow (daily) data (aligner selects the right rows).
        bar_indices_train : array, shape (n_train,)
        y_train : array, shape (n_train,)
        reg_lambda : float
        sample_weights : array, shape (n_train,), optional
            Exponential decay weights. If None, uniform weights.
        """
        Phi = self.combined_feature_map(
            fast_data_train, slow_data_all, bar_indices_train
        )
        D = Phi.shape[1]

        if sample_weights is not None:
            W = np.asarray(sample_weights, dtype=np.float64)
            W = W / W.sum() * len(W)  # normalize so mean weight = 1
            W_sqrt = np.sqrt(W)
            Phi_w = Phi * W_sqrt[:, np.newaxis]
            y_w = y_train * W_sqrt
        else:
            Phi_w = Phi
            y_w = y_train

        PhiTPhi = Phi_w.T @ Phi_w
        PhiTy = Phi_w.T @ y_w
        self._w = np.linalg.solve(PhiTPhi + reg_lambda * np.eye(D), PhiTy)
        self._fitted = True

    def predict(
        self,
        fast_data_test,
        slow_data_all,
        bar_indices_test: np.ndarray,
    ) -> np.ndarray:
        """Predict using combined features and fitted weights."""
        assert getattr(self, '_fitted', False), "Must call fit() first"
        Phi = self.combined_feature_map(
            fast_data_test, slow_data_all, bar_indices_test
        )
        return Phi @ self._w


# ============================================================================
# Exponential Decay Weights
# ============================================================================

def exponential_decay_weights(
    n: int,
    half_life: float,
) -> np.ndarray:
    """
    Exponential decay weights for an expanding window.

    w_i = exp(-λ * (n - 1 - i))  where λ = ln(2) / half_life

    The most recent observation (i = n-1) gets weight 1.0.
    An observation half_life steps ago gets weight 0.5.

    Parameters
    ----------
    n : int
        Number of observations in the training window.
    half_life : float
        Number of observations for weight to decay to 0.5.
        Set to np.inf for uniform weights.

    Returns
    -------
    weights : array, shape (n,)
        Non-negative weights, not normalized.
    """
    if half_life <= 0 or np.isinf(half_life):
        return np.ones(n)
    lam = np.log(2.0) / half_life
    t = np.arange(n, dtype=np.float64)
    return np.exp(-lam * (n - 1 - t))


# ============================================================================
# Purged Walk-Forward Engine
# ============================================================================

class PurgedWalkForward:
    """
    Expanding-window walk-forward backtester with purge + embargo.

    At each fold:
        1. Train window expands from the start of data up to the current split.
        2. An embargo gap of `embargo_bars` is inserted between train end
           and test start to prevent information leakage from autocorrelation.
        3. Exponential decay weights downweight older training observations.
        4. Only linear KRR weights are refit (RFF frequencies are frozen).
        5. Predictions are made on the test window.

    Supports both single-layer (fast only) and two-level (fast × slow) modes.

    Parameters
    ----------
    n_splits : int
        Number of walk-forward test windows.
    embargo_bars : int
        Number of bars to purge between train end and test start.
    min_train_bars : int
        Minimum number of training bars before first split.
    decay_half_life : float
        Half-life for exponential decay weighting. np.inf = uniform.
    reg_lambda : float
        KRR regularization parameter.
    """

    def __init__(
        self,
        n_splits: int = 10,
        embargo_bars: int = 50,
        min_train_bars: int = 500,
        decay_half_life: float = 2000.0,
        reg_lambda: float = 1e-3,
    ):
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars
        self.decay_half_life = decay_half_life
        self.reg_lambda = reg_lambda

    def split_indices(self, n_total: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged expanding-window train/test index pairs.

        Returns
        -------
        splits : list of (train_idx, test_idx) tuples
        """
        # Test windows: divide the data after min_train_bars into n_splits chunks
        test_start_min = self.min_train_bars + self.embargo_bars
        if test_start_min >= n_total:
            raise ValueError(
                f"Not enough data: need {test_start_min} bars for first split "
                f"but only have {n_total}"
            )

        # Divide remaining bars into n_splits equal test windows
        remaining = n_total - test_start_min
        test_size = max(remaining // self.n_splits, 1)

        splits = []
        for i in range(self.n_splits):
            test_start = test_start_min + i * test_size
            test_end = min(test_start + test_size, n_total)
            if test_start >= n_total:
                break

            # Train: everything before (test_start - embargo)
            train_end = test_start - self.embargo_bars
            if train_end < self.min_train_bars:
                train_end = self.min_train_bars

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def run(
        self,
        data,
        y: np.ndarray,
        kernel: KernelLike,
    ) -> BacktestResult:
        """
        Run single-layer walk-forward backtest.

        Parameters
        ----------
        data : list-like
            Input data for kernel.feature_map() (e.g., LOBSnapshots).
        y : array, shape (n,)
            Regression targets (e.g., forward returns).
        kernel : KernelLike
            Kernel layer with feature_map/fit/predict.

        Returns
        -------
        BacktestResult
        """
        n = len(y)
        splits = self.split_indices(n)

        fold_results = []
        all_pred = []
        all_true = []
        all_idx = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            # Extract train/test data
            if isinstance(data, np.ndarray):
                train_data = data[train_idx]
                test_data = data[test_idx]
            else:
                train_data = [data[i] for i in train_idx]
                test_data = [data[i] for i in test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            # Compute sample weights (exponential decay)
            weights = exponential_decay_weights(
                len(train_idx), self.decay_half_life
            )

            # Weighted KRR fit: (Φᵀ W Φ + λI)⁻¹ Φᵀ W y
            Phi_train = kernel.feature_map(train_data)
            D = Phi_train.shape[1]
            W_sqrt = np.sqrt(weights / weights.sum() * len(weights))
            Phi_w = Phi_train * W_sqrt[:, np.newaxis]
            y_w = y_train * W_sqrt

            w = np.linalg.solve(
                Phi_w.T @ Phi_w + self.reg_lambda * np.eye(D),
                Phi_w.T @ y_w,
            )

            # Predict on test
            Phi_test = kernel.feature_map(test_data)
            y_pred = Phi_test @ w

            # Diagnostics
            train_pred = Phi_train @ w
            train_loss = float(np.mean(weights * (y_train - train_pred) ** 2))

            fold_results.append(WalkForwardResult(
                fold_idx=fold_i,
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                y_pred=y_pred,
                y_true=y_test,
                weights_norm=float(np.linalg.norm(w)),
                train_loss=train_loss,
            ))

            all_pred.append(y_pred)
            all_true.append(y_test)
            all_idx.append(test_idx)

        return BacktestResult(
            fold_results=fold_results,
            y_pred_oos=np.concatenate(all_pred),
            y_true_oos=np.concatenate(all_true),
            oos_indices=np.concatenate(all_idx),
            n_folds=len(fold_results),
        )

    def run_two_level(
        self,
        fast_data,
        slow_data,
        y: np.ndarray,
        combiner: TwoLevelKernelCombiner,
        bar_indices: Optional[np.ndarray] = None,
        progress_callback=None,
    ) -> BacktestResult:
        """
        Run two-level (fast × slow) walk-forward backtest.

        Parameters
        ----------
        fast_data : list-like
            Bar-resolution data for the fast layer.
        slow_data : list-like
            Daily-resolution data for the slow layer (all days; alignment
            is handled by the combiner's MultiResolutionAligner).
        y : array, shape (n_bars,)
            Regression targets at bar resolution.
        combiner : TwoLevelKernelCombiner
            Handles feature combination and alignment.
        bar_indices : array, shape (n_bars,), optional
            Indices into the original bar array. Defaults to arange(n).

        Returns
        -------
        BacktestResult
        """
        n = len(y)
        if bar_indices is None:
            bar_indices = np.arange(n)

        splits = self.split_indices(n)
        fold_results = []
        all_pred, all_true, all_idx = [], [], []

        n_splits = len(splits)
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            if progress_callback is not None:
                progress_callback(fold_i, n_splits)

            if isinstance(fast_data, np.ndarray):
                fast_train = fast_data[train_idx]
                fast_test = fast_data[test_idx]
            else:
                fast_train = [fast_data[i] for i in train_idx]
                fast_test = [fast_data[i] for i in test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]
            bi_train = bar_indices[train_idx]
            bi_test = bar_indices[test_idx]

            weights = exponential_decay_weights(
                len(train_idx), self.decay_half_life
            )

            combiner.fit(
                fast_train, slow_data, bi_train, y_train,
                reg_lambda=self.reg_lambda,
                sample_weights=weights,
            )

            y_pred = combiner.predict(fast_test, slow_data, bi_test)

            # Diagnostics
            y_train_pred = combiner.predict(fast_train, slow_data, bi_train)
            train_loss = float(np.mean(weights * (y_train - y_train_pred) ** 2))

            fold_results.append(WalkForwardResult(
                fold_idx=fold_i,
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                y_pred=y_pred,
                y_true=y_test,
                weights_norm=float(np.linalg.norm(combiner._w)),
                train_loss=train_loss,
            ))

            all_pred.append(y_pred)
            all_true.append(y_test)
            all_idx.append(test_idx)

        if progress_callback is not None:
            progress_callback(n_splits, n_splits)

        return BacktestResult(
            fold_results=fold_results,
            y_pred_oos=np.concatenate(all_pred),
            y_true_oos=np.concatenate(all_true),
            oos_indices=np.concatenate(all_idx),
            n_folds=len(fold_results),
        )


# ============================================================================
# Combinatorial Purged Cross-Validation (CPCV)
# ============================================================================

class CPCVEvaluator:
    """
    Combinatorial Purged Cross-Validation for computing the Probability
    of Backtest Overfitting (PBO).

    Given N time-sorted groups and a test size of k groups:
    - Generate all C(N, k) train/test splits
    - For each split, purge observations near the train/test boundary
    - Evaluate each strategy on each split
    - PBO = fraction of combinations where the IS-best strategy
      underperforms the median OOS

    Parameters
    ----------
    n_groups : int
        Number of contiguous time groups to divide data into.
    n_test_groups : int
        Number of groups in each test set.
    embargo_bars : int
        Bars to purge at train/test boundary.

    References
    ----------
    Bailey, Borwein, López de Prado & Zhu (2017),
    "The Probability of Backtest Overfitting"
    """

    def __init__(
        self,
        n_groups: int = 10,
        n_test_groups: int = 2,
        embargo_bars: int = 50,
    ):
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_bars = embargo_bars

    def _make_groups(self, n_total: int) -> List[np.ndarray]:
        """Divide n_total observations into n_groups contiguous blocks."""
        boundaries = np.linspace(0, n_total, self.n_groups + 1, dtype=int)
        groups = []
        for i in range(self.n_groups):
            groups.append(np.arange(boundaries[i], boundaries[i + 1]))
        return groups

    def _purge_train(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> np.ndarray:
        """Remove training observations within embargo_bars of any test obs."""
        if self.embargo_bars <= 0:
            return train_idx

        test_min, test_max = test_idx.min(), test_idx.max()

        # Purge: remove train points in [test_min - embargo, test_max + embargo]
        lower = test_min - self.embargo_bars
        upper = test_max + self.embargo_bars

        mask = (train_idx < lower) | (train_idx > upper)
        return train_idx[mask]

    def generate_splits(
        self, n_total: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate all C(N, k) purged train/test splits.

        Returns
        -------
        splits : list of (train_idx, test_idx) tuples
        """
        groups = self._make_groups(n_total)
        splits = []

        for test_group_indices in combinations(
            range(self.n_groups), self.n_test_groups
        ):
            test_idx = np.concatenate([groups[g] for g in test_group_indices])
            train_groups = [
                g for g in range(self.n_groups) if g not in test_group_indices
            ]
            train_idx = np.concatenate([groups[g] for g in train_groups])

            # Purge
            train_idx = self._purge_train(train_idx, test_idx)

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def compute_pbo(
        self,
        data,
        y: np.ndarray,
        strategy_kernels: List[KernelLike],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        reg_lambda: float = 1e-3,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute PBO across all combinatorial splits.

        Parameters
        ----------
        data : list-like
            Input data (snapshots or feature arrays).
        y : array, shape (n,)
            Regression targets.
        strategy_kernels : list of KernelLike
            Different strategy/kernel configurations to compare.
        metric_fn : callable
            metric_fn(y_true, y_pred) -> float, higher is better.
        reg_lambda : float
            Regularization for KRR.

        Returns
        -------
        pbo : float
            Probability of backtest overfitting in [0, 1].
        details : dict
            'is_scores': (n_splits, n_strategies) IS performance matrix
            'oos_scores': (n_splits, n_strategies) OOS performance matrix
            'n_overfit': number of splits where IS-best underperforms OOS median
        """
        n_total = len(y)
        splits = self.generate_splits(n_total)
        n_splits = len(splits)
        n_strategies = len(strategy_kernels)

        is_scores = np.zeros((n_splits, n_strategies))
        oos_scores = np.zeros((n_splits, n_strategies))

        for s_idx, (train_idx, test_idx) in enumerate(splits):
            if isinstance(data, np.ndarray):
                train_data = data[train_idx]
                test_data = data[test_idx]
            else:
                train_data = [data[i] for i in train_idx]
                test_data = [data[i] for i in test_idx]

            y_train, y_test = y[train_idx], y[test_idx]

            for k_idx, kernel in enumerate(strategy_kernels):
                # Fit KRR
                Phi_train = kernel.feature_map(train_data)
                D = Phi_train.shape[1]
                w = np.linalg.solve(
                    Phi_train.T @ Phi_train + reg_lambda * np.eye(D),
                    Phi_train.T @ y_train,
                )

                # IS score
                is_pred = Phi_train @ w
                is_scores[s_idx, k_idx] = metric_fn(y_train, is_pred)

                # OOS score
                Phi_test = kernel.feature_map(test_data)
                oos_pred = Phi_test @ w
                oos_scores[s_idx, k_idx] = metric_fn(y_test, oos_pred)

        # PBO: fraction of splits where IS-best ranks below OOS median
        n_overfit = 0
        for s_idx in range(n_splits):
            is_best = np.argmax(is_scores[s_idx])
            oos_of_best = oos_scores[s_idx, is_best]
            oos_median = np.median(oos_scores[s_idx])
            if oos_of_best <= oos_median:
                n_overfit += 1

        pbo = n_overfit / max(n_splits, 1)

        return pbo, {
            'is_scores': is_scores,
            'oos_scores': oos_scores,
            'n_overfit': n_overfit,
            'n_splits': n_splits,
        }


# ============================================================================
# CKA Redundancy Gate
# ============================================================================

class CKARedundancyGate:
    """
    Runtime CKA check to gate redundant kernels in the fast layer.

    Specifically designed for the Kyle's Lambda / VPIN interaction:
    if CKA(K_Lambda, K_VPIN) >= threshold, Lambda is dropped to avoid
    collinearity that would waste MKL capacity.

    Can also be used for any pair of kernels where redundancy is a concern.

    Parameters
    ----------
    threshold : float
        CKA threshold above which the candidate kernel is dropped.
        Default 0.5 per Anders' specification.
    n_sample : int
        Number of observations to subsample for CKA computation.
        Full Gram matrix is O(n²), so subsampling keeps it fast.
    seed : int
        Random seed for reproducible subsampling.

    Usage
    -----
        gate = CKARedundancyGate(threshold=0.5)
        keep, cka_val = gate.check(lambda_kernel, vpin_kernel, data)
        if keep:
            fast_kernels.append(lambda_kernel)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        n_sample: int = 200,
        seed: int = 42,
    ):
        self.threshold = threshold
        self.n_sample = n_sample
        self._rng = np.random.RandomState(seed)

    def check(
        self,
        candidate_kernel: KernelLike,
        reference_kernel: KernelLike,
        candidate_data,
        reference_data=None,
    ) -> tuple:
        """
        Check whether the candidate kernel is redundant with the reference.

        Each kernel may operate on different feature sets (e.g., K_Lambda
        uses price-impact features while K_VPIN uses flow-toxicity features).
        The CKA measures whether the two kernels' Gram matrices — and thus
        their induced geometries over the same set of time points — are
        redundant, even though the raw inputs differ.

        Parameters
        ----------
        candidate_kernel : KernelLike
            The kernel to potentially drop (e.g., K_Lambda).
        reference_kernel : KernelLike
            The reference kernel to compare against (e.g., K_VPIN).
        candidate_data : list-like or ndarray
            Input data for candidate_kernel.feature_map().
        reference_data : list-like or ndarray, optional
            Input data for reference_kernel.feature_map().
            If None, uses candidate_data for both (same-feature case).

        Returns
        -------
        keep : bool
            True if the candidate should be KEPT (CKA < threshold).
        cka_value : float
            The computed CKA value.
        """
        if reference_data is None:
            reference_data = candidate_data

        # Subsample for efficiency (same indices for both)
        n_cand = len(candidate_data) if hasattr(candidate_data, '__len__') else candidate_data.shape[0]
        n_ref = len(reference_data) if hasattr(reference_data, '__len__') else reference_data.shape[0]
        n = min(n_cand, n_ref)

        if n > self.n_sample:
            idx = self._rng.choice(n, self.n_sample, replace=False)
            idx.sort()
            if isinstance(candidate_data, np.ndarray):
                cand_sub = candidate_data[idx]
            else:
                cand_sub = [candidate_data[i] for i in idx]
            if isinstance(reference_data, np.ndarray):
                ref_sub = reference_data[idx]
            else:
                ref_sub = [reference_data[i] for i in idx]
        else:
            cand_sub = candidate_data
            ref_sub = reference_data

        # Compute feature maps and Gram matrices
        phi_cand = candidate_kernel.feature_map(cand_sub)
        phi_ref = reference_kernel.feature_map(ref_sub)

        K_cand = phi_cand @ phi_cand.T
        K_ref = phi_ref @ phi_ref.T

        # CKA via centered Gram matrices
        cka_value = self._cka(K_cand, K_ref)

        keep = cka_value < self.threshold
        return keep, float(cka_value)

    @staticmethod
    def _center_gram(K: np.ndarray) -> np.ndarray:
        """Center a Gram matrix: H K H where H = I - (1/n) 11^T."""
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        return H @ K @ H

    @classmethod
    def _cka(cls, K1: np.ndarray, K2: np.ndarray) -> float:
        """CKA = HSIC(K1,K2) / sqrt(HSIC(K1,K1) * HSIC(K2,K2))."""
        K1c = cls._center_gram(K1)
        K2c = cls._center_gram(K2)

        hsic_12 = np.sum(K1c * K2c)
        hsic_11 = np.sum(K1c * K1c)
        hsic_22 = np.sum(K2c * K2c)

        denom = np.sqrt(hsic_11 * hsic_22)
        if denom < 1e-12:
            return 0.0
        return float(hsic_12 / denom)

    def gate_fast_kernels(
        self,
        kernels: List,
        kernel_names: List[str],
        kernel_data: List,
        pairs_to_check: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List, List[str], Dict[str, Any]]:
        """
        Gate a list of fast-layer kernels, dropping redundant ones.

        Parameters
        ----------
        kernels : list of KernelLike
            All candidate fast-layer kernels.
        kernel_names : list of str
            Names corresponding to each kernel.
        kernel_data : list
            Per-kernel input data — kernel_data[i] is the data array
            for kernels[i].feature_map(). Each kernel may use different
            feature sets.
        pairs_to_check : list of (candidate_idx, reference_idx), optional
            Specific pairs to check. Default: check Lambda (idx 2) vs
            VPIN (idx 1) per Anders' specification.

        Returns
        -------
        kept_kernels : list of KernelLike
        kept_names : list of str
        report : dict
            'checks': list of dicts with pair info, CKA values, keep/drop
        """
        if pairs_to_check is None:
            # Default: check K_Lambda (idx 2) vs K_VPIN (idx 1)
            if len(kernels) >= 3:
                pairs_to_check = [(2, 1)]
            else:
                pairs_to_check = []

        drop_indices = set()
        checks = []

        for cand_idx, ref_idx in pairs_to_check:
            if cand_idx >= len(kernels) or ref_idx >= len(kernels):
                continue

            keep, cka_val = self.check(
                kernels[cand_idx], kernels[ref_idx],
                candidate_data=kernel_data[cand_idx],
                reference_data=kernel_data[ref_idx],
            )
            check_info = {
                'candidate': kernel_names[cand_idx],
                'reference': kernel_names[ref_idx],
                'cka': cka_val,
                'threshold': self.threshold,
                'keep': keep,
            }
            checks.append(check_info)

            if not keep:
                drop_indices.add(cand_idx)

        kept_kernels = [k for i, k in enumerate(kernels) if i not in drop_indices]
        kept_names = [n for i, n in enumerate(kernel_names) if i not in drop_indices]

        return kept_kernels, kept_names, {'checks': checks}
