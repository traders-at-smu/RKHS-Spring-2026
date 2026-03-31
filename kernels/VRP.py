"""
VRP Feature Space for RKHS — Daily Calendar Time
=================================================
Constructs a Reproducing Kernel Hilbert Space (RKHS) feature space
for the Volatility Risk Premium (VRP) to explain stock movement.

RKHS Requirements Satisfied:
  1. Positive Definiteness  — composite kernel (sum/product of PD kernels is PD)
  2. Symmetry               — all kernels are radial/periodic (K(x,x') = K(x',x))
  3. Regularization         — Tikhonov (ridge) regularization to avoid daily-noise overfit
  4. Representer Theorem    — solution f* lives in span of kernel evaluations at training pts

Kernel Composition:
  K_total = K_Periodic(t,t') × K_Matern32(vrp,vrp') + K_RQ(x,x')

Feature Vector x_t (daily):
  [vrp, term_structure, sin_doy, cos_doy, momentum, vix, put_call_ratio]
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  1. FEATURE ENGINEERING  (daily calendar time)
# ─────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build the RKHS input feature matrix X from a daily OHLCV + options DataFrame.

    Required columns in df:
        date          – pandas datetime
        close         – daily close price
        iv_30d        – 30-day implied volatility (annualised, e.g. 0.20 = 20%)
        iv_90d        – 90-day implied volatility
        rv_30d        – 30-day realised volatility  (rolling historical)
        vix           – VIX index level
        put_call_ratio– equity put/call ratio

    Returns
    -------
    X        : (N, D) float64 feature matrix
    y        : (N,)   float64 next-day returns  ΔS_{t+1}
    feat_names: list of feature names (length D)
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    required = ["date", "close", "iv_30d", "iv_90d", "rv_30d", "vix", "put_call_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ── VRP Core: implied − realised spread ──────────────────────────────
    df["vrp"] = df["iv_30d"] - df["rv_30d"]                  # tension signal

    # ── Term Structure: slope of volatility surface ───────────────────────
    # log ratio → captures contango (+) vs backwardation (−)
    df["term_structure"] = np.log(df["iv_90d"] / (df["iv_30d"] + 1e-8))

    # ── Calendar Time Encoding (Day-of-Year → cyclic) ─────────────────────
    # Ensures RKHS periodicity; avoids Dec-31 / Jan-1 discontinuity
    doy = df["date"].dt.dayofyear.values.astype(float)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # ── Day-of-Week Encoding (Monday effect, weekend gap) ─────────────────
    dow = df["date"].dt.dayofweek.values.astype(float)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 5)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 5)

    # ── Momentum / Trend (log-return over n-day window) ───────────────────
    for n in [5, 21]:
        df[f"mom_{n}d"] = np.log(df["close"] / df["close"].shift(n))

    # ── Market Stress: VIX (absolute fear level, log-scaled) ─────────────
    df["log_vix"] = np.log(df["vix"] + 1e-8)

    # ── Put-Call Ratio (market sentiment, log-scaled) ─────────────────────
    df["log_pcr"] = np.log(df["put_call_ratio"] + 1e-8)

    # ── Target: next-day log return ΔS_{t+1} ─────────────────────────────
    df["target"] = np.log(df["close"].shift(-1) / df["close"])

    # Drop rows with NaNs from shifts
    feat_names = [
        "vrp",
        "term_structure",
        "sin_doy", "cos_doy",
        "sin_dow", "cos_dow",
        "mom_5d", "mom_21d",
        "log_vix",
        "log_pcr",
    ]
    df_clean = df[feat_names + ["target"]].dropna()

    X = df_clean[feat_names].values.astype(np.float64)
    y = df_clean["target"].values.astype(np.float64)

    return X, y, feat_names


def standardise(X_train: np.ndarray,
                X_test: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Zero-mean / unit-variance standardisation fit on train, applied to test.
    Required for kernel length-scale consistency in the RKHS.
    """
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mu) / sig
    X_test_s  = (X_test  - mu) / sig if X_test is not None else None
    return X_train_s, X_test_s, mu, sig


# ─────────────────────────────────────────────
#  2. KERNEL DEFINITIONS
# ─────────────────────────────────────────────

def kernel_periodic(t: np.ndarray, t_prime: np.ndarray,
                    ell: float = 1.0, period: float = 1.0) -> np.ndarray:
    """
    Periodic Kernel  (captures January Effect, monthly seasonality, etc.)

        K_per(t,t') = exp( -2 sin²(pi|t-t'|/p) / ell² )

    Positive-definite  | Symmetric
    NOTE: PD is guaranteed in 1D. For multi-dim inputs we take the PRODUCT
    of per-column periodic kernels — product of PD kernels is PD — modelling
    independent periodicity along each time feature dimension.

    t, t_prime : (N, D_time) arrays — one periodic kernel per column.
    """
    t       = np.atleast_2d(t)           # (N, D)
    t_prime = np.atleast_2d(t_prime)     # (M, D)
    N, D = t.shape
    M    = t_prime.shape[0]
    K    = np.ones((N, M))
    for d in range(D):
        diff  = t[:, d:d+1] - t_prime[:, d].reshape(1, -1)   # (N, M)
        K    *= np.exp(-2.0 * np.sin(np.pi * diff / period) ** 2 / ell ** 2)
    return K


def kernel_matern32(v: np.ndarray, v_prime: np.ndarray,
                    ell: float = 1.0) -> np.ndarray:
    """
    Matérn 3/2 Kernel  (rough, heavy-tailed; better than RBF for VRP dynamics)

        K_mat(v,v') = (1 + √3|v-v'|/ℓ) exp(-√3|v-v'|/ℓ)

    Positive-definite ✓ | Symmetric ✓
    v : (N, D_vrp)  — VRP-related features
    """
    v      = np.atleast_2d(v)
    v_prime = np.atleast_2d(v_prime)
    r      = cdist(v, v_prime, metric="euclidean")      # (N, M)
    z      = np.sqrt(3.0) * r / ell
    return (1.0 + z) * np.exp(-z)


def kernel_rational_quadratic(x: np.ndarray, x_prime: np.ndarray,
                               ell: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    """
    Rational Quadratic Kernel  (mixture of RBFs; captures multi-scale VRP decay)

        K_RQ(x,x') = (1 + |x-x'|² / (2αℓ²))^{-α}

    Positive-definite ✓ | Symmetric ✓
    When α → ∞ this reduces to the RBF kernel.
    """
    x      = np.atleast_2d(x)
    x_prime = np.atleast_2d(x_prime)
    r2     = cdist(x, x_prime, metric="sqeuclidean")    # (N, M)
    return (1.0 + r2 / (2.0 * alpha * ell ** 2)) ** (-alpha)


def kernel_total(X_a: np.ndarray, X_b: np.ndarray,
                 idx_time: list,
                 idx_vrp:  list,
                 ell_per:  float = 1.0,
                 ell_mat:  float = 1.0,
                 ell_rq:   float = 1.0,
                 alpha_rq: float = 1.0,
                 period:   float = 1.0) -> np.ndarray:
    """
    Composite RKHS Kernel:
        K_total = K_Periodic(t,t') × K_Matern32(vrp,vrp') + K_RQ(x,x')

    Sums and products of PD kernels are PD  →  RKHS membership preserved.

    Parameters
    ----------
    X_a, X_b   : feature matrices  (N×D) and (M×D)
    idx_time   : column indices for calendar-time features
    idx_vrp    : column indices for VRP-related features
    """
    # Time component (periodic)
    K_per  = kernel_periodic(X_a[:, idx_time], X_b[:, idx_time],
                             ell=ell_per, period=period)

    # VRP component (Matérn 3/2 — rough)
    K_mat  = kernel_matern32(X_a[:, idx_vrp], X_b[:, idx_vrp], ell=ell_mat)

    # Full feature space (RQ — multi-scale)
    K_rq   = kernel_rational_quadratic(X_a, X_b, ell=ell_rq, alpha=alpha_rq)

    # Composite:  interaction term + global multi-scale term
    return K_per * K_mat + K_rq


# ─────────────────────────────────────────────
#  3. RKHS KERNEL RIDGE REGRESSION
#     Tikhonov regularisation  (Representer Theorem)
# ─────────────────────────────────────────────

class VRPKernelRidgeRegression:
    """
    Solves the RKHS regression problem:

        f* = argmin_{f ∈ H}  Σ (ΔS_{t+1} - f(x_t))²  +  λ||f||²_H

    By the Representer Theorem the solution is:

        f*(x) = Σ_i α_i K(x_i, x)

    where α = (K + λI)^{-1} y

    Parameters
    ----------
    lam       : Tikhonov regularisation strength  (λ)
    idx_time  : feature column indices for periodic kernel
    idx_vrp   : feature column indices for Matérn kernel
    ell_per   : length-scale, periodic kernel
    ell_mat   : length-scale, Matérn kernel
    ell_rq    : length-scale, RQ kernel
    alpha_rq  : scale-mixture parameter, RQ kernel
    """

    def __init__(self,
                 lam:      float = 1e-3,
                 idx_time: list  = None,
                 idx_vrp:  list  = None,
                 ell_per:  float = 1.0,
                 ell_mat:  float = 1.0,
                 ell_rq:   float = 1.0,
                 alpha_rq: float = 1.0):
        self.lam      = lam
        self.idx_time = idx_time or [2, 3, 4, 5]   # sin/cos doy + dow
        self.idx_vrp  = idx_vrp  or [0, 1]          # vrp, term_structure
        self.ell_per  = ell_per
        self.ell_mat  = ell_mat
        self.ell_rq   = ell_rq
        self.alpha_rq = alpha_rq
        self.alpha_   = None   # kernel weights  (Representer coefficients)
        self.X_train_ = None

    def _K(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return kernel_total(A, B,
                            idx_time=self.idx_time,
                            idx_vrp=self.idx_vrp,
                            ell_per=self.ell_per,
                            ell_mat=self.ell_mat,
                            ell_rq=self.ell_rq,
                            alpha_rq=self.alpha_rq)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VRPKernelRidgeRegression":
        """
        Fit the RKHS model.  Solves (K + λI)α = y  via Cholesky.
        """
        self.X_train_ = X.copy()
        N = X.shape[0]
        K  = self._K(X, X)                              # (N×N) kernel matrix
        # Tikhonov regularisation  →  (K + λI)α = y
        A  = K + self.lam * np.eye(N)
        self.alpha_ = np.linalg.solve(A, y)             # stable Cholesky solve
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        f*(x) = Σ_i α_i K(x_i, x_test)
        """
        if self.alpha_ is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        K_test = self._K(self.X_train_, X_test)         # (N_train × N_test)
        return K_test.T @ self.alpha_

    def rkhs_norm(self) -> float:
        """
        ||f*||²_H = α^T K α   (scalar, measures model complexity in the RKHS)
        """
        K = self._K(self.X_train_, self.X_train_)
        return float(self.alpha_ @ K @ self.alpha_)

    def representer_weights(self) -> np.ndarray:
        """Return α — the Representer Theorem coefficients."""
        return self.alpha_.copy()


# ─────────────────────────────────────────────
#  4. FEATURE SPACE DIAGNOSTIC TOOLS
# ─────────────────────────────────────────────

def check_positive_definite(K: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Verify RKHS requirement: kernel matrix K must be positive semi-definite.
    Checks all eigenvalues ≥ -tol  (numerical tolerance for floating-point).
    """
    eigvals = np.linalg.eigvalsh(K)
    min_eig = eigvals.min()
    is_pd   = min_eig >= -tol
    print(f"  Min eigenvalue : {min_eig:.6e}   →  PD check: {'✅ PASSED' if is_pd else '❌ FAILED'}")
    return is_pd


def check_symmetry(K: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Verify RKHS requirement: K(x,x') = K(x',x).
    """
    asymmetry = np.max(np.abs(K - K.T))
    ok = asymmetry < tol
    print(f"  Max asymmetry  : {asymmetry:.2e}   →  Symmetry check: {'✅ PASSED' if ok else '❌ FAILED'}")
    return ok


def rkhs_requirements_report(K: np.ndarray) -> None:
    """
    Print a summary RKHS validity report for a given kernel matrix.
    """
    print("\n" + "="*52)
    print("  RKHS VALIDITY REPORT")
    print("="*52)
    pd_ok  = check_positive_definite(K)
    sym_ok = check_symmetry(K)
    print(f"  Kernel size    : {K.shape}")
    print(f"  Trace (sum of eigs): {np.trace(K):.4f}")
    print(f"  Frobenius norm : {np.linalg.norm(K, 'fro'):.4f}")
    status = "✅ VALID RKHS KERNEL" if (pd_ok and sym_ok) else "❌ INVALID"
    print(f"\n  Overall status : {status}")
    print("="*52 + "\n")


# ─────────────────────────────────────────────
#  5. DEMO — Synthetic Daily Data
# ─────────────────────────────────────────────

def generate_synthetic_daily_data(n_days: int = 504, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic daily market data that mimics realistic VRP dynamics.
    Suitable for testing the feature space without live data.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")  # business days

    # Simulate a correlated vol surface with mean-reversion
    rv_base  = 0.15 + 0.05 * rng.standard_normal(n_days).cumsum() * 0.01
    rv_base  = np.clip(rv_base, 0.05, 0.60)
    iv_30d   = rv_base + 0.02 + 0.01 * rng.standard_normal(n_days)   # IV > RV on avg
    iv_90d   = iv_30d  + 0.005 + 0.005 * rng.standard_normal(n_days) # term premium
    vix      = iv_30d * 100 * (1 + 0.1 * rng.standard_normal(n_days))
    pcr      = 0.8 + 0.2 * rng.standard_normal(n_days)

    # Simulate price with VRP-driven mean reversion + random walk
    vrp_signal = iv_30d - rv_base
    returns = -0.5 * vrp_signal + 0.01 * rng.standard_normal(n_days)
    close   = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "date"          : dates,
        "close"         : close,
        "iv_30d"        : np.clip(iv_30d, 0.01, 1.0),
        "iv_90d"        : np.clip(iv_90d, 0.01, 1.0),
        "rv_30d"        : np.clip(rv_base, 0.01, 1.0),
        "vix"           : np.clip(vix, 5.0, 80.0),
        "put_call_ratio": np.clip(pcr, 0.3, 2.5),
    })


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   VRP Feature Space for RKHS  —  Daily Calendar Time ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Generate / load data ─────────────────────────────────────────────
    print("[1] Generating synthetic daily market data (504 days)…")
    df = generate_synthetic_daily_data(n_days=504)
    print(f"    Date range : {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"    Shape      : {df.shape}\n")

    # ── Build RKHS feature matrix ─────────────────────────────────────────
    print("[2] Constructing RKHS feature vector x_t…")
    X, y, feat_names = build_feature_matrix(df)
    print(f"    Feature matrix X : {X.shape}  (N samples × D features)")
    print(f"    Target vector  y : {y.shape}  (next-day log-returns)")
    print(f"    Feature names    : {feat_names}\n")

    # ── Train / test split (walk-forward, no leakage) ─────────────────────
    split     = int(0.8 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    # ── Standardise (fit on train only) ───────────────────────────────────
    print("[3] Standardising features (fit on train, apply to test)…")
    X_tr_s, X_te_s, mu, sig = standardise(X_tr, X_te)
    print(f"    μ  : {np.round(mu, 4)}")
    print(f"    σ  : {np.round(sig, 4)}\n")

    # ── Feature index mapping (after standardisation) ─────────────────────
    # idx_time: sin_doy, cos_doy, sin_dow, cos_dow  (cols 2–5)
    # idx_vrp : vrp, term_structure                 (cols 0–1)
    idx_time = [feat_names.index(f) for f in ["sin_doy", "cos_doy", "sin_dow", "cos_dow"]]
    idx_vrp  = [feat_names.index(f) for f in ["vrp", "term_structure"]]

    # ── Build composite kernel matrix & run RKHS checks ───────────────────
    print("[4] Building composite kernel matrix K_total…")
    K_sample = kernel_total(X_tr_s[:50], X_tr_s[:50],
                            idx_time=idx_time, idx_vrp=idx_vrp,
                            ell_per=1.0, ell_mat=1.0, ell_rq=1.0, alpha_rq=2.0)
    rkhs_requirements_report(K_sample)

    # ── Fit Kernel Ridge Regression (Tikhonov-regularised RKHS) ───────────
    print("[5] Fitting RKHS Kernel Ridge Regression…")
    model = VRPKernelRidgeRegression(
        lam=1e-3,
        idx_time=idx_time,
        idx_vrp=idx_vrp,
        ell_per=1.0,
        ell_mat=0.8,
        ell_rq=1.2,
        alpha_rq=2.0,
    )
    model.fit(X_tr_s, y_tr)

    alpha = model.representer_weights()
    norm  = model.rkhs_norm()
    print(f"    Representer weights α : shape {alpha.shape}")
    print(f"    RKHS norm  ||f*||²_H  : {norm:.6f}")
    print(f"    Top-5 |α| indices     : {np.argsort(np.abs(alpha))[-5:][::-1]}\n")

    # ── Evaluate on out-of-sample test set ────────────────────────────────
    print("[6] Out-of-sample evaluation…")
    y_pred = model.predict(X_te_s)

    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot
    rmse   = np.sqrt(np.mean((y_te - y_pred) ** 2))
    corr   = np.corrcoef(y_te, y_pred)[0, 1]

    print(f"    R²   (out-of-sample) : {r2:.4f}")
    print(f"    RMSE (out-of-sample) : {rmse:.6f}")
    print(f"    Corr (y, ŷ)          : {corr:.4f}\n")

    # ── Feature Importance via kernel weight decomposition ─────────────────
    print("[7] Feature attribution via |α|-weighted kernel contributions…")
    abs_alpha = np.abs(alpha)
    # Weight each training observation's feature contribution
    weighted_X = abs_alpha[:, None] * X_tr_s            # (N_train × D)
    feature_importance = weighted_X.mean(axis=0)
    # Normalise by absolute sum so direction is preserved but magnitudes compare
    abs_sum = np.abs(feature_importance).sum()
    feature_importance_norm = feature_importance / (abs_sum + 1e-8)

    print(f"\n    {'Feature':<20}  {'Importance':>12}")
    print("    " + "-"*34)
    ranked = np.argsort(np.abs(feature_importance_norm))[::-1]
    for i in ranked:
        val = feature_importance_norm[i]
        sign = "+" if val >= 0 else "-"
        bar  = "█" * int(abs(val) * 40)
        print(f"    {feat_names[i]:<20}  {sign}{abs(val):>9.4f}  {bar}")

    print("\n✅  VRP RKHS Feature Space construction complete.\n")
    return model, X_tr_s, X_te_s, y_tr, y_te, feat_names


if __name__ == "__main__":
    main()
