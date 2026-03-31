"""
Nested Cross-Validation for Kernel Hyperparameters
====================================================
Traders@SMU — Quantitative Strategies Group
Week 5: Backtesting Framework

Architecture:
    Outer loop: Walk-forward OOS evaluation (PurgedWalkForward)
    Inner loop: Purged k-fold CV over (ℓ, λ) per layer (ν fixed at 1.5)

This module provides:
    1. NestedCV       — nested walk-forward + inner purged CV
    2. MKLOptimizer   — learns α weights and β via L2-regularized opt
    3. ScalerWrapper   — train-only scaler to prevent leakage
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Callable
from itertools import product as cart_product
from dataclasses import dataclass
import warnings

try:
    from .walk_forward import (
        PurgedWalkForward,
        BacktestResult,
        exponential_decay_weights,
    )
except ImportError:
    from walk_forward import (
        PurgedWalkForward,
        BacktestResult,
        exponential_decay_weights,
    )


# ============================================================================
# Scaler Wrapper (leakage prevention)
# ============================================================================

class ScalerWrapper:
    """
    StandardScaler that fits on train only, transforms both train and test.

    Prevents information leakage from test set statistics into training.

    Usage:
        scaler = ScalerWrapper()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> 'ScalerWrapper':
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-10] = 1.0  # prevent div-by-zero
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Must call fit() first"
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================================
# Inner CV: Purged K-Fold
# ============================================================================

class PurgedKFold:
    """
    Purged k-fold cross-validation for time-series data.

    Splits data into k contiguous folds. For each test fold, removes
    training observations within `embargo` bars of the test boundary
    to prevent lookahead bias.

    Parameters
    ----------
    n_folds : int
        Number of CV folds.
    embargo : int
        Bars to purge at boundaries.
    """

    def __init__(self, n_folds: int = 5, embargo: int = 50):
        self.n_folds = n_folds
        self.embargo = embargo

    def split(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate (train_idx, test_idx) pairs."""
        fold_size = n // self.n_folds
        splits = []

        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            test_idx = np.arange(test_start, test_end)

            # Train: all non-test indices, then purge
            train_idx = np.concatenate([
                np.arange(0, test_start),
                np.arange(test_end, n),
            ])

            # Purge: remove train points within embargo of test boundaries
            if self.embargo > 0 and len(train_idx) > 0:
                lower = test_start - self.embargo
                upper = test_end + self.embargo
                mask = (train_idx < lower) | (train_idx >= upper)
                train_idx = train_idx[mask]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits


# ============================================================================
# Inner Loop: Grid Search over (ℓ, λ) for a single kernel
# ============================================================================

@dataclass
class CVResult:
    """Result from inner CV grid search."""
    best_length_scale: float
    best_reg_lambda: float
    best_score: float
    all_scores: Dict[Tuple[float, float], float]


def inner_cv_grid_search(
    kernel,
    data,
    y: np.ndarray,
    length_scales: List[float],
    reg_lambdas: List[float],
    cv: PurgedKFold,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    sample_weights: Optional[np.ndarray] = None,
) -> CVResult:
    """
    Grid search over (ℓ, λ) using purged k-fold CV.

    For each (ℓ, λ) combination:
        1. Update kernel's length scale
        2. Re-initialize RFF with new frequencies (length scale changes ω)
        3. For each fold: fit KRR, compute OOS metric
        4. Average metric across folds

    ν is fixed at 1.5 (not searched).

    Parameters
    ----------
    kernel : BaseKernel
        Kernel with .length_scale, .rff, .n_rff, .nu attributes.
    data : list-like
        Input data for kernel.feature_map().
    y : array, shape (n,)
    length_scales : list of float
        Grid values for ℓ.
    reg_lambdas : list of float
        Grid values for λ.
    cv : PurgedKFold
    metric_fn : callable
        metric_fn(y_true, y_pred) → float (higher is better).
    sample_weights : array, shape (n,), optional

    Returns
    -------
    CVResult
    """
    n = len(y)
    splits = cv.split(n)
    all_scores = {}
    best_score = -np.inf
    best_ls, best_lam = length_scales[0], reg_lambdas[0]

    for ls, lam in cart_product(length_scales, reg_lambdas):
        # Update kernel hyperparameters
        kernel.length_scale = ls
        if hasattr(kernel, 'rff') and kernel.rff is not None:
            kernel.rff.update_params(kernel.nu, ls)
        else:
            kernel._init_rff()

        fold_scores = []
        for train_idx, test_idx in splits:
            if isinstance(data, np.ndarray):
                train_data = data[train_idx]
                test_data = data[test_idx]
            else:
                train_data = [data[i] for i in train_idx]
                test_data = [data[i] for i in test_idx]

            y_train, y_test = y[train_idx], y[test_idx]

            # Feature maps
            Phi_train = kernel.feature_map(train_data)
            Phi_test = kernel.feature_map(test_data)
            D = Phi_train.shape[1]

            # Weighted KRR
            if sample_weights is not None:
                sw = sample_weights[train_idx]
                sw = sw / sw.sum() * len(sw)
                W_sqrt = np.sqrt(sw)
                Phi_w = Phi_train * W_sqrt[:, np.newaxis]
                y_w = y_train * W_sqrt
            else:
                Phi_w = Phi_train
                y_w = y_train

            try:
                w = np.linalg.solve(
                    Phi_w.T @ Phi_w + lam * np.eye(D),
                    Phi_w.T @ y_w,
                )
                y_pred = Phi_test @ w
                score = metric_fn(y_test, y_pred)
            except np.linalg.LinAlgError:
                score = -np.inf

            fold_scores.append(score)

        mean_score = np.mean(fold_scores) if fold_scores else -np.inf
        all_scores[(ls, lam)] = mean_score

        if mean_score > best_score:
            best_score = mean_score
            best_ls = ls
            best_lam = lam

    return CVResult(
        best_length_scale=best_ls,
        best_reg_lambda=best_lam,
        best_score=best_score,
        all_scores=all_scores,
    )


# ============================================================================
# Extended grid search: length_scale + ν + reg_lambda
# ============================================================================

@dataclass
class ExtendedCVResult:
    """Result from extended inner CV grid search (includes ν)."""
    best_nu: float
    best_length_scale: float
    best_reg_lambda: float
    best_score: float
    all_scores: Dict


def inner_cv_grid_search_extended(
    kernel,
    data,
    y: np.ndarray,
    nus: List[float],
    length_scales: List[float],
    reg_lambdas: List[float],
    cv: PurgedKFold,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    sample_weights: Optional[np.ndarray] = None,
) -> ExtendedCVResult:
    """
    Grid search over (ν, ℓ, λ) using purged k-fold CV.

    Parameters
    ----------
    kernel : FeatureMatrixKernel or similar
        Kernel with .nu, .length_scale, ._init_rff() attributes.
    data : np.ndarray, shape (n, d)
    y : array, shape (n,)
    nus : list of float
        Grid values for Matérn smoothness ν.
    length_scales : list of float
        Grid values for length scale ℓ.
    reg_lambdas : list of float
        Grid values for regularization λ.
    cv : PurgedKFold
    metric_fn : callable
        metric_fn(y_true, y_pred) → float (higher is better).
    sample_weights : optional

    Returns
    -------
    ExtendedCVResult
    """
    n = len(y)
    splits = list(cv.split(n))
    all_scores = {}
    best_score = -np.inf
    best_nu = nus[0]
    best_ls = length_scales[0]
    best_lam = reg_lambdas[0]

    for nu in nus:
        for ls in length_scales:
            # Update kernel hyperparameters and re-sample RFF
            kernel.nu = nu
            kernel.length_scale = ls
            if hasattr(kernel, '_init_rff'):
                kernel._init_rff()
            elif hasattr(kernel, 'rff') and kernel.rff is not None:
                kernel.rff.update_params(nu, ls)

            for lam in reg_lambdas:
                fold_scores = []
                for train_idx, test_idx in splits:
                    if isinstance(data, np.ndarray):
                        train_data = data[train_idx]
                        test_data = data[test_idx]
                    else:
                        train_data = [data[i] for i in train_idx]
                        test_data = [data[i] for i in test_idx]

                    y_train, y_test = y[train_idx], y[test_idx]

                    Phi_train = kernel.feature_map(train_data)
                    Phi_test = kernel.feature_map(test_data)
                    D = Phi_train.shape[1]

                    if sample_weights is not None:
                        sw = sample_weights[train_idx]
                        sw = sw / sw.sum() * len(sw)
                        W_sqrt = np.sqrt(sw)
                        Phi_w = Phi_train * W_sqrt[:, np.newaxis]
                        y_w = y_train * W_sqrt
                    else:
                        Phi_w = Phi_train
                        y_w = y_train

                    try:
                        w = np.linalg.solve(
                            Phi_w.T @ Phi_w + lam * np.eye(D),
                            Phi_w.T @ y_w,
                        )
                        y_pred = Phi_test @ w
                        score = metric_fn(y_test, y_pred)
                    except np.linalg.LinAlgError:
                        score = -np.inf

                    fold_scores.append(score)

                mean_score = np.mean(fold_scores) if fold_scores else -np.inf
                all_scores[(nu, ls, lam)] = mean_score

                if mean_score > best_score:
                    best_score = mean_score
                    best_nu = nu
                    best_ls = ls
                    best_lam = lam

    return ExtendedCVResult(
        best_nu=best_nu,
        best_length_scale=best_ls,
        best_reg_lambda=best_lam,
        best_score=best_score,
        all_scores=all_scores,
    )


# ============================================================================
# MKL (Multiple Kernel Learning) Optimizer
# ============================================================================

class MKLOptimizer:
    """
    Learns the layer weights α₁,...,α₇ and β via L2-regularized optimization.

    Architecture:
        K_total = K_fast + β · K_fast · K_slow
        K_fast  = α₁·K_LOB + α₂·K_VPIN + α₃·K_Lambda + α₄·K_VannaCharm
                  (dollar bar resolution)
        K_slow  = α₅·K_VRP + α₆·K_GEX + α₇·K_Sentiment
                  (daily resolution)

    Note: K_Lambda is subject to a CKA redundancy gate against K_VPIN.
    If CKA(K_Lambda, K_VPIN) >= 0.5, K_Lambda is dropped before MKL
    optimization to prevent the optimizer from fighting collinearity.

    The α weights are optimized by projected gradient descent with
    non-negativity constraints.

    Parameters
    ----------
    n_fast_kernels : int
        Number of fast sub-kernels (default 4: LOB, VPIN, Lambda, VannaCharm).
        May be 3 if Lambda was dropped by the CKA redundancy gate.
    n_slow_kernels : int
        Number of slow sub-kernels (default 3: VRP, GEX, Sentiment).
    l2_penalty : float
        L2 penalty on weights to prevent overfitting.
    lr : float
        Learning rate for projected gradient descent.
    max_iter : int
        Maximum optimization iterations.
    """

    def __init__(
        self,
        n_fast_kernels: int = 4,
        n_slow_kernels: int = 3,
        l2_penalty: float = 0.01,
        lr: float = 0.01,
        max_iter: int = 200,
    ):
        self.n_fast = n_fast_kernels
        self.n_slow = n_slow_kernels
        self.l2_penalty = l2_penalty
        self.lr = lr
        self.max_iter = max_iter

        # Initialize weights uniformly
        self.alpha_fast = np.ones(n_fast_kernels) / n_fast_kernels
        self.alpha_slow = np.ones(n_slow_kernels) / n_slow_kernels
        self.beta = 1.0

    def _compute_combined_features(
        self,
        fast_feature_maps: List[np.ndarray],
        slow_feature_maps: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Combine pre-computed feature maps using current weights.

        Parameters
        ----------
        fast_feature_maps : list of arrays, each (n, D_k)
        slow_feature_maps : list of arrays, each (n, D_k), optional
            If provided, aligned to bar resolution already.

        Returns
        -------
        Phi_combined : array, shape (n, D_total)
        """
        # Fast layer: [√α₁ · φ₁, √α₂ · φ₂, ...]
        fast_parts = []
        for i, phi in enumerate(fast_feature_maps):
            fast_parts.append(np.sqrt(max(self.alpha_fast[i], 0)) * phi)
        Phi_fast = np.concatenate(fast_parts, axis=1)

        if slow_feature_maps is None or len(slow_feature_maps) == 0:
            return Phi_fast

        # Slow layer: [√α₃ · φ₃, √α₄ · φ₄, ...]
        slow_parts = []
        for i, phi in enumerate(slow_feature_maps):
            slow_parts.append(np.sqrt(max(self.alpha_slow[i], 0)) * phi)
        Phi_slow = np.concatenate(slow_parts, axis=1)

        # Product approximation: element-wise for same dimension,
        # or pad to max dimension
        n = Phi_fast.shape[0]
        D_fast = Phi_fast.shape[1]
        D_slow = Phi_slow.shape[1]

        # Simple product approximation: use column means as interaction
        # For full tensor sketch, use TwoLevelKernelCombiner
        D_prod = min(D_fast, D_slow)
        phi_product = Phi_fast[:, :D_prod] * Phi_slow[:, :D_prod]

        return np.concatenate([
            Phi_fast,
            np.sqrt(max(self.beta, 0)) * phi_product,
        ], axis=1)

    def optimize(
        self,
        fast_feature_maps: List[np.ndarray],
        y: np.ndarray,
        reg_lambda: float = 1e-3,
        slow_feature_maps: Optional[List[np.ndarray]] = None,
        sample_weights: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Optimize MKL weights via alternating minimization.

        Step 1: Fix α, β → solve KRR for w
        Step 2: Fix w → gradient step on α, β to minimize weighted MSE + L2

        Parameters
        ----------
        fast_feature_maps : list of arrays
        y : array, shape (n,)
        reg_lambda : float
        slow_feature_maps : list of arrays, optional
        sample_weights : array, shape (n,), optional
        verbose : bool

        Returns
        -------
        result : dict with 'alpha_fast', 'alpha_slow', 'beta', 'loss_history'
        """
        n = len(y)
        if sample_weights is not None:
            sw = sample_weights / sample_weights.sum() * n
        else:
            sw = np.ones(n)

        loss_history = []

        for iteration in range(self.max_iter):
            # Step 1: Compute combined features with current weights
            Phi = self._compute_combined_features(
                fast_feature_maps, slow_feature_maps
            )
            D = Phi.shape[1]

            # Weighted KRR
            W_sqrt = np.sqrt(sw)
            Phi_w = Phi * W_sqrt[:, np.newaxis]
            y_w = y * W_sqrt

            try:
                w = np.linalg.solve(
                    Phi_w.T @ Phi_w + reg_lambda * np.eye(D),
                    Phi_w.T @ y_w,
                )
            except np.linalg.LinAlgError:
                break

            # Compute loss
            residual = y - Phi @ w
            loss = float(
                np.mean(sw * residual ** 2)
                + self.l2_penalty * (
                    np.sum(self.alpha_fast ** 2)
                    + np.sum(self.alpha_slow ** 2)
                    + self.beta ** 2
                )
            )
            loss_history.append(loss)

            # Step 2: Numerical gradient on α, β
            eps = 1e-5

            # Gradient for alpha_fast
            grad_af = np.zeros(self.n_fast)
            for i in range(self.n_fast):
                self.alpha_fast[i] += eps
                Phi_p = self._compute_combined_features(
                    fast_feature_maps, slow_feature_maps
                )
                loss_p = np.mean(sw * (y - Phi_p @ w) ** 2)
                self.alpha_fast[i] -= 2 * eps
                Phi_m = self._compute_combined_features(
                    fast_feature_maps, slow_feature_maps
                )
                loss_m = np.mean(sw * (y - Phi_m @ w) ** 2)
                self.alpha_fast[i] += eps  # restore
                grad_af[i] = (loss_p - loss_m) / (2 * eps)
                grad_af[i] += 2 * self.l2_penalty * self.alpha_fast[i]

            # Gradient for alpha_slow
            grad_as = np.zeros(self.n_slow)
            if slow_feature_maps is not None:
                for i in range(self.n_slow):
                    self.alpha_slow[i] += eps
                    Phi_p = self._compute_combined_features(
                        fast_feature_maps, slow_feature_maps
                    )
                    loss_p = np.mean(sw * (y - Phi_p @ w) ** 2)
                    self.alpha_slow[i] -= 2 * eps
                    Phi_m = self._compute_combined_features(
                        fast_feature_maps, slow_feature_maps
                    )
                    loss_m = np.mean(sw * (y - Phi_m @ w) ** 2)
                    self.alpha_slow[i] += eps
                    grad_as[i] = (loss_p - loss_m) / (2 * eps)
                    grad_as[i] += 2 * self.l2_penalty * self.alpha_slow[i]

            # Gradient for beta
            self.beta += eps
            Phi_p = self._compute_combined_features(
                fast_feature_maps, slow_feature_maps
            )
            loss_p = np.mean(sw * (y - Phi_p @ w) ** 2)
            self.beta -= 2 * eps
            Phi_m = self._compute_combined_features(
                fast_feature_maps, slow_feature_maps
            )
            loss_m = np.mean(sw * (y - Phi_m @ w) ** 2)
            self.beta += eps
            grad_beta = (loss_p - loss_m) / (2 * eps)
            grad_beta += 2 * self.l2_penalty * self.beta

            # Projected gradient step (project onto non-negative orthant)
            self.alpha_fast -= self.lr * grad_af
            self.alpha_fast = np.maximum(self.alpha_fast, 0.0)

            self.alpha_slow -= self.lr * grad_as
            self.alpha_slow = np.maximum(self.alpha_slow, 0.0)

            self.beta -= self.lr * grad_beta
            self.beta = max(self.beta, 0.0)

            if verbose and iteration % 50 == 0:
                print(
                    f"  MKL iter {iteration}: loss={loss:.6f}, "
                    f"α_fast={self.alpha_fast}, α_slow={self.alpha_slow}, "
                    f"β={self.beta:.4f}"
                )

            # Convergence check
            if len(loss_history) > 1:
                if abs(loss_history[-1] - loss_history[-2]) < 1e-8:
                    break

        return {
            'alpha_fast': self.alpha_fast.copy(),
            'alpha_slow': self.alpha_slow.copy(),
            'beta': self.beta,
            'loss_history': loss_history,
        }


# ============================================================================
# Nested CV: Outer Walk-Forward + Inner Grid Search
# ============================================================================

@dataclass
class NestedCVResult:
    """Result from the full nested CV procedure."""
    backtest_result: BacktestResult
    inner_cv_results: List[CVResult]  # one per outer fold
    best_params_per_fold: List[Dict[str, float]]


class NestedCV:
    """
    Nested cross-validation: outer walk-forward + inner purged k-fold.

    At each outer fold:
        1. Inner CV: grid search over (ℓ, λ) on the training set
        2. Refit with best hyperparameters on full training set
        3. Predict on OOS test set

    Parameters
    ----------
    outer_engine : PurgedWalkForward
        Walk-forward engine for outer loop.
    inner_n_folds : int
        Number of folds for inner CV.
    inner_embargo : int
        Embargo bars for inner CV.
    length_scales : list of float
        Grid values for ℓ to search.
    reg_lambdas : list of float
        Grid values for λ to search.
    metric_fn : callable
        metric_fn(y_true, y_pred) → float (higher is better).
    """

    def __init__(
        self,
        outer_engine: Optional[PurgedWalkForward] = None,
        inner_n_folds: int = 5,
        inner_embargo: int = 25,
        length_scales: Optional[List[float]] = None,
        reg_lambdas: Optional[List[float]] = None,
        metric_fn: Optional[Callable] = None,
    ):
        self.outer = outer_engine or PurgedWalkForward()
        self.inner_cv = PurgedKFold(n_folds=inner_n_folds, embargo=inner_embargo)
        self.length_scales = length_scales or [0.1, 0.5, 1.0, 2.0, 5.0]
        self.reg_lambdas = reg_lambdas or [1e-4, 1e-3, 1e-2, 1e-1]
        self.metric_fn = metric_fn or self._neg_mse

    @staticmethod
    def _neg_mse(y_true, y_pred):
        return -float(np.mean((y_true - y_pred) ** 2))

    def run(
        self,
        kernel,
        data,
        y: np.ndarray,
    ) -> NestedCVResult:
        """
        Run nested CV.

        Parameters
        ----------
        kernel : BaseKernel
            Kernel to optimize. Must have .length_scale, .rff attributes.
        data : list-like
        y : array, shape (n,)

        Returns
        -------
        NestedCVResult
        """
        n = len(y)
        outer_splits = self.outer.split_indices(n)

        fold_results = []
        inner_results = []
        best_params = []
        all_pred, all_true, all_idx = [], [], []

        # Save original hyperparams to restore after each fold
        original_ls = kernel.length_scale

        for fold_i, (train_idx, test_idx) in enumerate(outer_splits):
            # --- Inner CV on training data only ---
            if isinstance(data, np.ndarray):
                train_data = data[train_idx]
                test_data = data[test_idx]
            else:
                train_data = [data[i] for i in train_idx]
                test_data = [data[i] for i in test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            # Scaler: fit on train, transform both
            scaler = ScalerWrapper()
            if isinstance(train_data, np.ndarray) and train_data.ndim == 2:
                train_data = scaler.fit_transform(train_data)
                test_data = scaler.transform(test_data)

            # Exponential decay weights for training
            weights = exponential_decay_weights(
                len(train_idx), self.outer.decay_half_life
            )

            # Inner grid search
            cv_result = inner_cv_grid_search(
                kernel=kernel,
                data=train_data,
                y=y_train,
                length_scales=self.length_scales,
                reg_lambdas=self.reg_lambdas,
                cv=self.inner_cv,
                metric_fn=self.metric_fn,
                sample_weights=weights,
            )
            inner_results.append(cv_result)
            best_params.append({
                'length_scale': cv_result.best_length_scale,
                'reg_lambda': cv_result.best_reg_lambda,
            })

            # --- Refit on full training set with best params ---
            kernel.length_scale = cv_result.best_length_scale
            if hasattr(kernel, 'rff') and kernel.rff is not None:
                kernel.rff.update_params(kernel.nu, cv_result.best_length_scale)
            else:
                kernel._init_rff()

            Phi_train = kernel.feature_map(train_data)
            D = Phi_train.shape[1]
            lam = cv_result.best_reg_lambda

            W_sqrt = np.sqrt(weights / weights.sum() * len(weights))
            Phi_w = Phi_train * W_sqrt[:, np.newaxis]
            y_w = y_train * W_sqrt

            w = np.linalg.solve(
                Phi_w.T @ Phi_w + lam * np.eye(D),
                Phi_w.T @ y_w,
            )

            # --- OOS prediction ---
            Phi_test = kernel.feature_map(test_data)
            y_pred = Phi_test @ w

            train_pred = Phi_train @ w
            train_loss = float(np.mean(weights * (y_train - train_pred) ** 2))

            from walk_forward import WalkForwardResult
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

        # Restore original kernel params
        kernel.length_scale = original_ls
        if hasattr(kernel, 'rff') and kernel.rff is not None:
            kernel.rff.update_params(kernel.nu, original_ls)

        backtest = BacktestResult(
            fold_results=fold_results,
            y_pred_oos=np.concatenate(all_pred),
            y_true_oos=np.concatenate(all_true),
            oos_indices=np.concatenate(all_idx),
            n_folds=len(fold_results),
        )

        return NestedCVResult(
            backtest_result=backtest,
            inner_cv_results=inner_results,
            best_params_per_fold=best_params,
        )
