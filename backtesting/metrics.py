"""
Performance & Diagnostic Metrics
=================================
Traders@SMU — Quantitative Strategies Group
Week 5: Backtesting Framework

Implements:
    1. Deflated Sharpe Ratio (DSR)  — overfitting detection
    2. Sortino ratio                — downside-adjusted performance
    3. CKA (Centered Kernel Alignment) — kernel redundancy
    4. Effective dimensionality     — eigenvalue concentration
    5. OOS negative log-likelihood  — probabilistic scoring

References:
    - Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
    - Kornblith et al. (2019), "Similarity of Neural Network Representations
      Revisited" (CKA)
"""

import numpy as np
from scipy.stats import norm


# ============================================================================
# 1. Deflated Sharpe Ratio
# ============================================================================

def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    T: int,
    skew: float = 0.0,
    kurtosis_excess: float = 0.0,
) -> float:
    """
    Deflated Sharpe Ratio — probability that the observed SR is genuine
    after accounting for multiple testing.

    DSR = Φ( (SR_obs - E[max SR | H0]) / σ(SR) )

    Under H0 (no skill), the expected maximum SR from n_trials independent
    strategies with T observations each is bounded by:

        E[max SR] ≈ σ(SR) * [ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e)) ]

    where γ ≈ 0.5772 (Euler–Mascheroni), and σ(SR) corrects for skew & kurtosis.

    Parameters
    ----------
    observed_sr : float
        The observed annualized Sharpe ratio.
    n_trials : int
        Number of strategy variations tried (including abandoned ones).
    T : int
        Number of return observations used to estimate the SR.
    skew : float
        Skewness of the return series.
    kurtosis_excess : float
        Excess kurtosis (kurtosis - 3) of the return series.

    Returns
    -------
    dsr : float
        Probability in [0, 1] that the SR is genuine. Values > 0.95 are
        typically considered statistically significant.
    """
    if n_trials < 1 or T < 2:
        return 0.0

    # Variance of SR estimator (Lo 2002, adjusted for non-normality)
    var_sr = (1.0
              - skew * observed_sr
              + ((kurtosis_excess - 1) / 4.0) * observed_sr ** 2) / T

    if var_sr <= 0:
        var_sr = 1.0 / T  # fallback to basic variance

    std_sr = np.sqrt(var_sr)

    # Expected max SR under null (Bailey & López de Prado approximation)
    euler_mascheroni = 0.5772156649
    N = max(n_trials, 1)
    z = norm.ppf(1.0 - 1.0 / N) if N > 1 else 0.0
    e_max_sr = std_sr * (
        (1 - euler_mascheroni) * z
        + euler_mascheroni * norm.ppf(1.0 - 1.0 / (N * np.e))
    )

    # DSR is the probability that observed SR exceeds the null max
    dsr = norm.cdf((observed_sr - e_max_sr) / std_sr)
    return float(dsr)


# ============================================================================
# 2. Sortino Ratio
# ============================================================================

def sortino_ratio(
    returns: np.ndarray,
    target: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """
    Sortino ratio — penalizes only downside deviation.

    Sortino = (mean(r) - target) / downside_std * sqrt(annualization)

    Parameters
    ----------
    returns : array-like, shape (T,)
        Period returns.
    target : float
        Minimum acceptable return per period.
    annualization : float
        Annualization factor (252 for daily, 1 for already annualized).

    Returns
    -------
    sortino : float
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return 0.0

    excess = returns - target
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))

    if downside_std < 1e-12:
        return np.inf if np.mean(excess) > 0 else 0.0

    return float(np.mean(excess) / downside_std * np.sqrt(annualization))


# ============================================================================
# 3. Centered Kernel Alignment (CKA)
# ============================================================================

def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix: H K H where H = I - (1/n) 11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def centered_kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """
    CKA — measures similarity / redundancy between two kernel matrices.

    CKA(K1, K2) = HSIC(K1, K2) / sqrt(HSIC(K1, K1) * HSIC(K2, K2))

    where HSIC is the Hilbert-Schmidt Independence Criterion:
        HSIC(K1, K2) = (1/(n-1)^2) * tr(K1_c @ K2_c)

    Returns
    -------
    cka : float in [0, 1]
        0 = orthogonal (no redundancy), 1 = identical representations.
    """
    K1c = _center_gram(K1)
    K2c = _center_gram(K2)

    hsic_12 = np.sum(K1c * K2c)  # tr(K1c @ K2c) via element-wise
    hsic_11 = np.sum(K1c * K1c)
    hsic_22 = np.sum(K2c * K2c)

    denom = np.sqrt(hsic_11 * hsic_22)
    if denom < 1e-12:
        return 0.0

    return float(hsic_12 / denom)


# ============================================================================
# 4. Effective Dimensionality
# ============================================================================

def effective_dimensionality(K: np.ndarray) -> float:
    """
    Effective dimensionality of a Gram matrix via eigenvalue entropy.

    d_eff = exp( -sum_i p_i log(p_i) )

    where p_i = λ_i / sum(λ), λ_i are eigenvalues.

    Higher = more spread out eigenvalues (richer representation).
    Lower = concentrated on few eigenvectors (potential overfitting).

    Returns
    -------
    d_eff : float >= 1.0
    """
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.maximum(eigenvalues, 0.0)  # clip numerical negatives

    total = eigenvalues.sum()
    if total < 1e-12:
        return 1.0

    p = eigenvalues / total
    # filter out zeros to avoid log(0)
    p = p[p > 1e-15]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


# ============================================================================
# 5. Out-of-Sample Negative Log-Likelihood
# ============================================================================

def oos_negative_log_likelihood(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_var: np.ndarray,
) -> float:
    """
    Gaussian negative log-likelihood on held-out data.

    NLL = (1/n) * sum_i [ 0.5 * log(2π σ²_i) + (y_i - ŷ_i)² / (2 σ²_i) ]

    Parameters
    ----------
    y_true : array, shape (n,)
        Observed values.
    y_pred : array, shape (n,)
        Predicted means.
    y_var : array, shape (n,)
        Predicted variances (must be > 0).

    Returns
    -------
    nll : float
        Mean NLL per observation. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_var = np.asarray(y_var, dtype=np.float64)
    y_var = np.maximum(y_var, 1e-10)  # prevent log(0)

    n = len(y_true)
    nll = 0.5 * np.mean(
        np.log(2 * np.pi * y_var) + (y_true - y_pred) ** 2 / y_var
    )
    return float(nll)
