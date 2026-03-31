"""
VPIN Feature Space for RKHS (Reproducing Kernel Hilbert Space)
================================================================
Implements:
  - Dollar Bar construction (homoscedastic sampling)
  - Bulk Volume Classification (BVC) for buy/sell volume split
  - VPIN computation over a rolling window of dollar bars
  - Feature vector assembly: [VPIN, ΔVPIN, bar_volatility, relative_volume]
  - Kernel functions satisfying Mercer's condition (RBF, Laplacian, Polynomial)
  - RKHS model via kernel ridge regression (Representer Theorem)
  - Regularised training + inference pipeline

Dependencies: numpy, pandas, scipy, scikit-learn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ──────────────────────────────────────────────────────────────────────────────
# 1.  DOLLAR BAR CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def build_dollar_bars(
    trades: pd.DataFrame,
    dollar_threshold: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Aggregate raw tick-level trades into Dollar Bars.

    Each bar closes once the cumulative dollar value (price × size) crosses
    `dollar_threshold`.  This gives equal economic exposure per bar, which
    makes the subsequent feature space homoscedastic — a requirement for
    stable kernel matrices in the RKHS.

    Parameters
    ----------
    trades : pd.DataFrame
        Must contain columns: ['timestamp', 'price', 'size']
    dollar_threshold : float
        Dollar value that triggers a new bar (default $1 000 000).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, volume, dollar_volume,
                 open_time, close_time, vwap, bar_return, bar_volatility
    """
    required = {"timestamp", "price", "size"}
    if not required.issubset(trades.columns):
        raise ValueError(f"trades must contain columns: {required}")

    trades = trades.sort_values("timestamp").reset_index(drop=True)
    trades["dollar_value"] = trades["price"] * trades["size"]

    bars: list[dict] = []
    cum_dollar = 0.0
    bar_prices: list[float] = []
    bar_sizes: list[float] = []
    bar_dollars: list[float] = []
    bar_start: pd.Timestamp = trades["timestamp"].iloc[0]

    for _, row in trades.iterrows():
        cum_dollar += row["dollar_value"]
        bar_prices.append(row["price"])
        bar_sizes.append(row["size"])
        bar_dollars.append(row["dollar_value"])

        if cum_dollar >= dollar_threshold:
            prices_arr = np.array(bar_prices)
            sizes_arr  = np.array(bar_sizes)
            dollars_arr = np.array(bar_dollars)
            vwap = np.dot(prices_arr, sizes_arr) / sizes_arr.sum()

            bars.append({
                "open_time":     bar_start,
                "close_time":    row["timestamp"],
                "open":          prices_arr[0],
                "high":          prices_arr.max(),
                "low":           prices_arr.min(),
                "close":         prices_arr[-1],
                "volume":        sizes_arr.sum(),
                "dollar_volume": dollars_arr.sum(),
                "vwap":          vwap,
                "bar_return":    np.log(prices_arr[-1] / prices_arr[0]),
                # Intra-bar log-return std — proxy for realised volatility
                "bar_volatility": np.std(np.log(prices_arr[1:] / prices_arr[:-1]))
                                  if len(prices_arr) > 1 else 0.0,
                "n_trades":      len(prices_arr),
            })
            # Reset accumulator
            cum_dollar = 0.0
            bar_prices.clear()
            bar_sizes.clear()
            bar_dollars.clear()
            bar_start = row["timestamp"]

    return pd.DataFrame(bars)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  BULK VOLUME CLASSIFICATION  (buy / sell split per dollar bar)
# ──────────────────────────────────────────────────────────────────────────────

def bulk_volume_classify(
    bars: pd.DataFrame,
    z_score_window: int = 50,
) -> pd.DataFrame:
    """
    Bulk Volume Classification (Easley et al., 2012).

    For each bar, the fraction of volume classified as buy-initiated is:

        P(buy | bar) = Φ( ΔP / σ_ΔP )

    where  Φ  is the standard normal CDF and  σ_ΔP  is estimated over a
    rolling window of bar returns.

    Parameters
    ----------
    bars : pd.DataFrame
        Output of `build_dollar_bars`.
    z_score_window : int
        Rolling window for normalising bar returns (default 50 bars).

    Returns
    -------
    pd.DataFrame
        Original bars enriched with: buy_volume, sell_volume, order_imbalance
    """
    from scipy.stats import norm

    bars = bars.copy()
    rolling_std = bars["bar_return"].rolling(z_score_window, min_periods=1).std()
    rolling_std = rolling_std.replace(0, np.nan).fillna(1e-8)          # avoid / 0

    z = bars["bar_return"] / rolling_std
    p_buy = norm.cdf(z)                                                 # ∈ (0, 1)

    bars["buy_volume"]       = bars["volume"] * p_buy
    bars["sell_volume"]      = bars["volume"] * (1 - p_buy)
    bars["order_imbalance"]  = np.abs(bars["buy_volume"] - bars["sell_volume"])

    return bars


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VPIN COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_vpin(
    bars: pd.DataFrame,
    window: int = 50,
) -> pd.Series:
    """
    Compute rolling VPIN over dollar bars.

        VPIN_t = Σ |V_i^S − V_i^B|  /  (n × V̄)

    where n = `window` and V̄ is the mean volume per bar over the window.

    Parameters
    ----------
    bars : pd.DataFrame
        Output of `bulk_volume_classify` (must have buy_volume, sell_volume).
    window : int
        Number of bars in the rolling window (default 50).

    Returns
    -------
    pd.Series
        VPIN values indexed to `bars`.
    """
    imbalance = bars["order_imbalance"]
    volume    = bars["volume"]

    rolling_sum_imbalance = imbalance.rolling(window, min_periods=window).sum()
    rolling_mean_volume   = volume.rolling(window, min_periods=window).mean()

    vpin = rolling_sum_imbalance / (window * rolling_mean_volume)
    vpin = vpin.clip(0.0, 1.0)          # VPIN is theoretically bounded in [0, 1]
    return vpin.rename("vpin")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE VECTOR ASSEMBLY  →  input space for the RKHS
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    bars: pd.DataFrame,
    vpin: pd.Series,
    rel_vol_window: int = 20,
) -> pd.DataFrame:
    """
    Assemble the 4-dimensional feature vector required by the RKHS model.

        x = [ VPIN_t,  ΔVPIN_t,  bar_volatility_t,  relative_volume_t ]

    - VPIN_t            : current toxicity estimate           ∈ [0, 1]
    - ΔVPIN_t           : first difference (momentum)         ∈ ℝ
    - bar_volatility_t  : intra-bar realised vol              ≥ 0
    - relative_volume_t : volume / rolling mean volume        ≥ 0

    Parameters
    ----------
    bars : pd.DataFrame
        Output of `bulk_volume_classify`.
    vpin : pd.Series
        Output of `compute_vpin`.
    rel_vol_window : int
        Window for computing relative volume (default 20).

    Returns
    -------
    pd.DataFrame
        Feature matrix with NaN rows dropped.
    """
    features = pd.DataFrame(index=bars.index)

    features["vpin"]             = vpin
    features["delta_vpin"]       = vpin.diff()
    features["bar_volatility"]   = bars["bar_volatility"]

    rolling_mean_vol             = bars["volume"].rolling(rel_vol_window, min_periods=1).mean()
    features["relative_volume"]  = bars["volume"] / rolling_mean_vol.replace(0, np.nan)

    return features.dropna()


# ──────────────────────────────────────────────────────────────────────────────
# 5.  KERNEL FUNCTIONS  (all satisfy Mercer's condition)
# ──────────────────────────────────────────────────────────────────────────────

KernelType = Literal["rbf", "laplacian", "polynomial"]


def compute_kernel_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    kernel: KernelType = "rbf",
    sigma: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
) -> np.ndarray:
    """
    Compute the kernel (Gram) matrix K(X, Y).

    Supported kernels
    -----------------
    rbf         : K(x,y) = exp(−‖x−y‖² / 2σ²)
                  Best for VPIN: captures smooth non-linear regimes.

    laplacian   : K(x,y) = exp(−‖x−y‖₁ / σ)
                  Sharper peak; better for sudden VPIN spikes (crypto, illiquid).

    polynomial  : K(x,y) = (x·y + c)^d
                  Encodes explicit power-law interactions between features.

    All three satisfy Mercer's condition (positive semi-definiteness), which
    guarantees convergence of the RKHS optimisation.

    Parameters
    ----------
    X      : (n, d) array
    Y      : (m, d) array  — if None, computes K(X, X)
    kernel : one of 'rbf', 'laplacian', 'polynomial'
    sigma  : length-scale for RBF / Laplacian
    degree : polynomial degree
    coef0  : polynomial free term c

    Returns
    -------
    np.ndarray of shape (n, m)
    """
    if Y is None:
        Y = X

    if kernel == "rbf":
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-sq_dists / (2.0 * sigma ** 2))

    elif kernel == "laplacian":
        l1_dists = cdist(X, Y, metric="cityblock")
        return np.exp(-l1_dists / sigma)

    elif kernel == "polynomial":
        return (X @ Y.T + coef0) ** degree

    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from: rbf, laplacian, polynomial")


# ──────────────────────────────────────────────────────────────────────────────
# 6.  RKHS MODEL  (kernel ridge regression via the Representer Theorem)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VPINRKHSModel:
    """
    Kernel Ridge Regression in the RKHS spanned by VPIN dollar-bar features.

    The Representer Theorem guarantees that the optimal function takes the form

        f*(x) = Σᵢ αᵢ K(x, xᵢ)

    where αᵢ are learned coefficients and {xᵢ} are the training points.
    We solve for α by minimising the regularised empirical risk:

        min_α  ‖y − Kα‖² + λ αᵀKα

    with closed-form solution:  α = (K + λI)⁻¹ y

    The λ parameter controls the RKHS norm ‖f‖_H, preventing over-fitting
    to transient toxicity spikes.

    Parameters
    ----------
    kernel     : kernel type (see `compute_kernel_matrix`)
    lambda_reg : regularisation strength  λ  (default 1e-3)
    sigma      : kernel length-scale (RBF / Laplacian)
    degree     : polynomial degree
    coef0      : polynomial free term
    """
    kernel    : KernelType = "rbf"
    lambda_reg: float      = 1e-3
    sigma     : float      = 1.0
    degree    : int        = 3
    coef0     : float      = 1.0

    # Learned after fit()
    alpha_      : Optional[np.ndarray] = field(default=None, repr=False)
    X_train_    : Optional[np.ndarray] = field(default=None, repr=False)
    scaler_     : StandardScaler       = field(default_factory=StandardScaler, repr=False)

    def _kernel(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        return compute_kernel_matrix(
            X, Y,
            kernel=self.kernel,
            sigma=self.sigma,
            degree=self.degree,
            coef0=self.coef0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VPINRKHSModel":
        """
        Fit the RKHS model.

        Parameters
        ----------
        X : (n, d)  feature matrix (raw, will be scaled internally)
        y : (n,)    target labels / regression values

        Returns
        -------
        self
        """
        X_scaled       = self.scaler_.fit_transform(X)
        self.X_train_  = X_scaled

        K              = self._kernel(X_scaled)
        n              = K.shape[0]
        # Closed-form solution:  α = (K + λI)⁻¹ y
        self.alpha_    = np.linalg.solve(K + self.lambda_reg * np.eye(n), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using  f*(x) = Σᵢ αᵢ K(x, xᵢ).

        Parameters
        ----------
        X : (m, d)  new feature vectors (raw)

        Returns
        -------
        np.ndarray of shape (m,)
        """
        if self.alpha_ is None or self.X_train_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self.scaler_.transform(X)
        K_star   = self._kernel(X_scaled, self.X_train_)   # (m, n)
        return K_star @ self.alpha_

    def rkhs_norm(self) -> float:
        """
        Compute ‖f*‖²_H = αᵀKα.

        A high RKHS norm after training suggests the model is over-fitting
        toxicity spikes.  Increase `lambda_reg` to penalise this.
        """
        if self.alpha_ is None or self.X_train_ is None:
            raise RuntimeError("Call fit() before rkhs_norm().")
        K = self._kernel(self.X_train_)
        return float(self.alpha_ @ K @ self.alpha_)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  END-TO-END PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    trades: pd.DataFrame,
    dollar_threshold: float = 1_000_000.0,
    vpin_window: int = 50,
    bvc_window: int = 50,
    rel_vol_window: int = 20,
    kernel: KernelType = "rbf",
    lambda_reg: float = 1e-3,
    sigma: float = 1.0,
    target_col: str = "vpin",          # predict future VPIN by default
    forecast_horizon: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Full pipeline: raw trades → VPIN RKHS model.

    Steps
    -----
    1. Build dollar bars
    2. BVC buy/sell classification
    3. VPIN computation
    4. Feature matrix assembly
    5. Target: shifted VPIN (h-step ahead forecast)
    6. Train / test split (no shuffle — preserves time ordering)
    7. Fit RKHS model
    8. Evaluate

    Parameters
    ----------
    trades           : raw tick data
    dollar_threshold : dollar value per bar
    vpin_window      : bars in VPIN rolling window
    bvc_window       : bars for BVC z-score normalisation
    rel_vol_window   : bars for relative volume
    kernel           : 'rbf', 'laplacian', or 'polynomial'
    lambda_reg       : regularisation  λ
    sigma            : kernel length-scale
    target_col       : column in feature df to use as target (shifted)
    forecast_horizon : steps ahead to forecast
    test_size        : fraction of data for test set
    random_state     : reproducibility seed

    Returns
    -------
    dict with keys: model, features, vpin, bars, train_mse, test_mse
    """
    print("── Step 1: Building dollar bars ──")
    bars = build_dollar_bars(trades, dollar_threshold)
    print(f"   {len(bars)} dollar bars constructed.")

    print("── Step 2: Bulk Volume Classification ──")
    bars = bulk_volume_classify(bars, bvc_window)

    print("── Step 3: Computing VPIN ──")
    vpin = compute_vpin(bars, vpin_window)
    print(f"   VPIN range: [{vpin.min():.4f}, {vpin.max():.4f}]")

    print("── Step 4: Building feature matrix ──")
    features = build_feature_matrix(bars, vpin, rel_vol_window)
    print(f"   Feature matrix shape: {features.shape}")

    print("── Step 5: Constructing target (h-step VPIN forecast) ──")
    y = features[target_col].shift(-forecast_horizon).dropna()
    X = features.loc[y.index]
    X_arr = X.values.astype(float)
    y_arr = y.values.astype(float)

    print("── Step 6: Train / test split (temporal) ──")
    split = int(len(X_arr) * (1 - test_size))
    X_train, X_test = X_arr[:split], X_arr[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("── Step 7: Fitting RKHS model ──")
    model = VPINRKHSModel(kernel=kernel, lambda_reg=lambda_reg, sigma=sigma)
    model.fit(X_train, y_train)
    print(f"   RKHS norm ‖f*‖²_H = {model.rkhs_norm():.6f}")

    print("── Step 8: Evaluation ──")
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    train_mse    = mean_squared_error(y_train, y_pred_train)
    test_mse     = mean_squared_error(y_test, y_pred_test)
    print(f"   Train MSE: {train_mse:.6f}")
    print(f"   Test  MSE: {test_mse:.6f}")

    return {
        "model":     model,
        "features":  features,
        "vpin":      vpin,
        "bars":      bars,
        "train_mse": train_mse,
        "test_mse":  test_mse,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 8.  SYNTHETIC DATA GENERATOR  (for testing without live market data)
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_trades(
    n: int = 100_000,
    seed: int = 42,
    toxicity_shock_prob: float = 0.001,
) -> pd.DataFrame:
    """
    Generate a synthetic tick-level trade stream for development / testing.

    Includes occasional "toxicity shocks" (informed trading events) that
    cause price drift and volume spikes — the signal VPIN is designed to detect.

    Parameters
    ----------
    n                  : number of ticks
    seed               : random seed
    toxicity_shock_prob: probability of a shock starting on any tick

    Returns
    -------
    pd.DataFrame with columns: timestamp, price, size
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range("2024-01-01", periods=n, freq="100ms")
    price      = np.zeros(n)
    size       = np.zeros(n)

    price[0]   = 100.0
    in_shock   = False
    shock_dir  = 1
    shock_len  = 0

    for i in range(1, n):
        if not in_shock and rng.random() < toxicity_shock_prob:
            in_shock  = True
            shock_dir = rng.choice([-1, 1])
            shock_len = rng.integers(50, 200)

        if in_shock:
            drift      = shock_dir * rng.uniform(0.02, 0.08)
            vol        = 0.03
            shock_len -= 1
            if shock_len <= 0:
                in_shock = False
        else:
            drift = 0.0
            vol   = 0.01

        log_ret  = rng.normal(drift, vol)
        price[i] = price[i - 1] * np.exp(log_ret)

        # Volume proportional to absolute return + noise
        base_vol  = 500
        shock_mul = 3.0 if in_shock else 1.0
        size[i]   = max(1.0, rng.exponential(base_vol * shock_mul * (1 + 5 * abs(log_ret))))

    return pd.DataFrame({"timestamp": timestamps, "price": price, "size": size})


# ──────────────────────────────────────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VPIN → RKHS Pipeline  (Dollar Bar Edition)")
    print("=" * 60)

    # ── Generate synthetic market data
    print("\nGenerating synthetic tick data …")
    trades = generate_synthetic_trades(n=200_000, seed=0)
    print(f"Ticks: {len(trades):,}  |  Price range: "
          f"[{trades['price'].min():.2f}, {trades['price'].max():.2f}]")

    # ── Run full pipeline with RBF kernel
    print("\n[RBF Kernel]")
    results_rbf = run_pipeline(
        trades,
        dollar_threshold  = 50_000,
        vpin_window       = 50,
        kernel            = "rbf",
        sigma             = 1.0,
        lambda_reg        = 1e-3,
        forecast_horizon  = 5,
    )

    # ── Compare kernels
    print("\n[Laplacian Kernel]")
    results_lap = run_pipeline(
        trades,
        dollar_threshold  = 50_000,
        vpin_window       = 50,
        kernel            = "laplacian",
        sigma             = 1.0,
        lambda_reg        = 1e-3,
        forecast_horizon  = 5,
    )

    # ── Show RKHS norms — higher norm = richer function = watch regularisation
    print("\n── RKHS norms ──")
    print(f"   RBF       ‖f*‖²_H = {results_rbf['model'].rkhs_norm():.6f}")
    print(f"   Laplacian ‖f*‖²_H = {results_lap['model'].rkhs_norm():.6f}")

    # ── Single prediction example
    model   = results_rbf["model"]
    features = results_rbf["features"]
    sample_x = features.values[-10:-5]          # last 5 feature vectors
    preds    = model.predict(sample_x)
    print("\n── Sample 5-step-ahead VPIN forecasts (RBF) ──")
    for i, p in enumerate(preds):
        print(f"   Bar -{10-i}:  predicted VPIN = {p:.4f}")
