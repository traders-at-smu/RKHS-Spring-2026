"""
Signal Generation from Kernel Outputs
=======================================
Traders@SMU — Quantitative Strategies Group
Week 5: Backtesting Framework

Signal pipeline:
    1. Compute rolling mean embedding: μ̂_t = (1/W) Σ_{i=t-W+1}^{t} φ(x_i)
    2. Hilbert distance from mean: d_t = ||φ(x_t) - μ̂_t||_H
    3. Estimate OU parameters on the d_t series
    4. s-score: s_t = (d_t - μ_d) / σ_d
    5. Entry at |s| > entry_threshold, exit at |s| < exit_threshold

This module provides:
    - OUSignalGenerator  — s-score computation + OU parameter estimation
    - SignalDiagnostics  — per-layer SNR, confidence intervals
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass


# ============================================================================
# OU Parameter Estimation
# ============================================================================

@dataclass
class OUParams:
    """Ornstein-Uhlenbeck process parameters: dX = κ(μ - X)dt + σ dW."""
    kappa: float    # Mean-reversion speed
    mu: float       # Long-run mean
    sigma: float    # Volatility
    half_life: float  # ln(2) / κ — time to revert halfway

    @property
    def is_mean_reverting(self) -> bool:
        return self.kappa > 0


def estimate_ou_params(
    series: np.ndarray,
    dt: float = 1.0,
) -> OUParams:
    """
    Estimate OU parameters via OLS on the discrete AR(1) representation.

    The discretized OU process is:
        X_{t+1} - X_t = κ(μ - X_t)Δt + σ√Δt ε_t

    which is a linear regression:
        ΔX_t = a + b·X_t + ε_t
    where b = -κΔt, a = κμΔt, σ² = Var(ε) / Δt

    Parameters
    ----------
    series : array, shape (T,)
        The observed series (e.g., Hilbert distances from mean embedding).
    dt : float
        Time step between observations (1.0 for bar-by-bar).

    Returns
    -------
    OUParams
    """
    series = np.asarray(series, dtype=np.float64)
    T = len(series)
    if T < 3:
        return OUParams(kappa=0.0, mu=0.0, sigma=1.0, half_life=np.inf)

    dX = np.diff(series)              # ΔX_t = X_{t+1} - X_t
    X = series[:-1]                    # X_t

    # OLS: dX = a + b * X + noise
    A = np.column_stack([np.ones(T - 1), X])
    params, residuals, _, _ = np.linalg.lstsq(A, dX, rcond=None)
    a, b = params

    # Extract OU parameters
    kappa = -b / dt
    if kappa < 1e-10:
        # Not mean-reverting; return degenerate params
        return OUParams(
            kappa=max(kappa, 0.0),
            mu=np.mean(series),
            sigma=np.std(dX) / np.sqrt(dt),
            half_life=np.inf,
        )

    mu = a / (kappa * dt)
    noise = dX - (a + b * X)
    sigma = np.std(noise) / np.sqrt(dt)
    half_life = np.log(2.0) / kappa

    return OUParams(kappa=kappa, mu=mu, sigma=sigma, half_life=half_life)


# ============================================================================
# Signal Generator
# ============================================================================

@dataclass
class Signal:
    """Container for signal values and metadata at each time step."""
    s_scores: np.ndarray         # shape (T,) — the s-score
    distances: np.ndarray        # shape (T,) — raw Hilbert distances
    mean_embeddings: np.ndarray  # shape (T, D) — rolling mean embedding
    ou_params: OUParams          # estimated OU parameters
    confidence_upper: np.ndarray  # shape (T,) — upper CI on s-score
    confidence_lower: np.ndarray  # shape (T,) — lower CI on s-score


class OUSignalGenerator:
    """
    Generate mean-reversion signals from RKHS feature maps.

    The s-score measures how far the current state is from the rolling
    mean in Hilbert space, normalized by its historical volatility.

    Parameters
    ----------
    rolling_window : int
        Number of observations for the rolling mean embedding.
    ou_window : int
        Number of observations for OU parameter estimation.
    ci_level : float
        Confidence level for signal confidence intervals (e.g., 0.95).
    """

    def __init__(
        self,
        rolling_window: int = 100,
        ou_window: int = 500,
        ci_level: float = 0.95,
    ):
        self.rolling_window = rolling_window
        self.ou_window = ou_window
        self.ci_level = ci_level

    def generate(
        self,
        feature_maps: np.ndarray,
        dt: float = 1.0,
    ) -> Signal:
        """
        Generate s-scores from pre-computed feature maps.

        Parameters
        ----------
        feature_maps : array, shape (T, D)
            Pre-computed φ(x_t) for each observation. Use
            kernel.feature_map(data) to produce this.
        dt : float
            Time step between observations.

        Returns
        -------
        Signal
        """
        T, D = feature_maps.shape
        W = self.rolling_window

        # --- Rolling mean embedding ---
        mean_embeddings = np.zeros((T, D))
        distances = np.full(T, np.nan)

        for t in range(T):
            start = max(0, t - W + 1)
            mean_embeddings[t] = np.mean(feature_maps[start:t + 1], axis=0)
            distances[t] = np.linalg.norm(
                feature_maps[t] - mean_embeddings[t]
            )

        # --- OU parameter estimation on a trailing window ---
        valid_start = W  # first point with a full rolling window
        if valid_start >= T:
            valid_start = 0
        d_series = distances[valid_start:]

        ou_window = min(self.ou_window, len(d_series))
        if ou_window > 2:
            ou_params = estimate_ou_params(d_series[-ou_window:], dt=dt)
        else:
            ou_params = OUParams(kappa=0.0, mu=0.0, sigma=1.0, half_life=np.inf)

        # --- s-score: rolling z-score of distances ---
        s_scores = np.full(T, np.nan)
        for t in range(W, T):
            window_d = distances[max(0, t - self.ou_window):t + 1]
            mu_d = np.mean(window_d)
            sigma_d = np.std(window_d)
            if sigma_d > 1e-12:
                s_scores[t] = (distances[t] - mu_d) / sigma_d
            else:
                s_scores[t] = 0.0

        # --- Confidence intervals ---
        # Under OU with estimated params, the CI on s_score is ±z * SE
        from scipy.stats import norm
        z_val = norm.ppf(0.5 + self.ci_level / 2.0)

        # SE of s-score ≈ 1 / sqrt(window) (CLT on the z-score)
        se = np.full(T, np.nan)
        for t in range(W, T):
            n_eff = min(t + 1, self.ou_window)
            se[t] = 1.0 / np.sqrt(max(n_eff, 1))

        confidence_upper = s_scores + z_val * se
        confidence_lower = s_scores - z_val * se

        return Signal(
            s_scores=s_scores,
            distances=distances,
            mean_embeddings=mean_embeddings,
            ou_params=ou_params,
            confidence_upper=confidence_upper,
            confidence_lower=confidence_lower,
        )

    def generate_from_kernel(
        self,
        kernel,
        data,
        dt: float = 1.0,
    ) -> Signal:
        """
        Convenience: compute feature maps then generate signals.

        Parameters
        ----------
        kernel : BaseKernel or RKHSLayer
        data : list-like
            Input data for kernel.feature_map().
        dt : float
        """
        phi = kernel.feature_map(data)
        return self.generate(phi, dt=dt)


# ============================================================================
# Trading Signal Positions
# ============================================================================

def generate_positions(
    s_scores: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    max_position: float = 1.0,
    gate_values: Optional[np.ndarray] = None,
    entry_trending: float = 0.75,
    entry_choppy: float = 2.0,
    exit_trending: float = 0.25,
    exit_choppy: float = 0.75,
    gate_threshold: float = 0.5,
) -> np.ndarray:
    """
    Generate position signals from s-scores using OU mean-reversion logic.

    Supports regime-conditional thresholds via gate_values:
        - gate > gate_threshold → trending → use tighter entry/exit
        - gate <= gate_threshold → choppy → use wider entry/exit
        - If gate_values is None, falls back to static thresholds.

    Rules:
        - Enter short when s > +entry_threshold
        - Enter long when s < -entry_threshold
        - Exit when |s| < exit_threshold
        - Position size is proportional to |s| (capped at max_position)

    Parameters
    ----------
    s_scores : array, shape (T,)
    entry_threshold : float
        Static entry threshold (used when gate_values is None).
    exit_threshold : float
        Static exit threshold (used when gate_values is None).
    max_position : float
    gate_values : array, shape (T,), optional
        Regime gate in (0, 1). High = trending, low = choppy.
    entry_trending : float
        Entry threshold in trending regime.
    entry_choppy : float
        Entry threshold in choppy regime.
    exit_trending : float
        Exit threshold in trending regime.
    exit_choppy : float
        Exit threshold in choppy regime.
    gate_threshold : float
        Gate cutoff separating trending from choppy.

    Returns
    -------
    positions : array, shape (T,)
        +1 = long, -1 = short, 0 = flat. Scaled by |s|/entry_threshold.
    """
    T = len(s_scores)
    positions = np.zeros(T)
    current_pos = 0.0
    use_regime = gate_values is not None

    for t in range(T):
        s = s_scores[t]
        if np.isnan(s):
            positions[t] = current_pos
            continue

        # Determine thresholds
        if use_regime and t < len(gate_values) and not np.isnan(gate_values[t]):
            g = gate_values[t]
            # Smooth interpolation between choppy and trending thresholds
            w = min(max((g - gate_threshold) / (1.0 - gate_threshold + 1e-8), 0), 1)
            ent = entry_choppy * (1 - w) + entry_trending * w
            ext = exit_choppy * (1 - w) + exit_trending * w
        else:
            ent = entry_threshold
            ext = exit_threshold

        if current_pos == 0.0:
            # Flat → check for entry
            if s > ent:
                # Far from mean (above) → short (mean reversion)
                current_pos = -min(abs(s) / ent, max_position)
            elif s < -ent:
                # Far from mean (below) → long
                current_pos = min(abs(s) / ent, max_position)
        else:
            # In position → check for exit
            if abs(s) < ext:
                current_pos = 0.0
            elif current_pos > 0 and s > ent:
                # Was long, now signal says go short → flip
                current_pos = -min(abs(s) / ent, max_position)
            elif current_pos < 0 and s < -ent:
                # Was short, now signal says go long → flip
                current_pos = min(abs(s) / ent, max_position)

        positions[t] = current_pos

    return positions


# ============================================================================
# Multi-Horizon Signal Aggregation
# ============================================================================

def multi_horizon_s_scores(
    predictions: np.ndarray,
    horizons: Tuple[int, ...] = (10, 30, 60),
    weights: Optional[Tuple[float, ...]] = None,
    roll_window: int = 100,
) -> np.ndarray:
    """
    Compute s-scores at multiple prediction horizons and aggregate.

    Instead of using raw 1-bar-ahead predictions, this smooths the
    prediction series at multiple lookahead windows and computes a
    weighted average of z-scores. This reduces noise and lets the
    kernel signal express over longer horizons.

    Pipeline:
        1. For each horizon h: compute rolling mean of predictions
           over h bars → smoothed_h[t] = mean(pred[t:t+h])
        2. z-score each smoothed series over roll_window
        3. Weighted average of z-scores

    Parameters
    ----------
    predictions : array, shape (T,)
        Raw 1-bar-ahead OOS predictions from kernel ridge regression.
    horizons : tuple of int
        Lookahead smoothing windows. Default: (10, 30, 60).
    weights : tuple of float, optional
        Weights per horizon. Default: equal weights.
    roll_window : int
        Rolling z-score window for each horizon.

    Returns
    -------
    agg_s_scores : array, shape (T,)
        Aggregated multi-horizon s-scores.
    """
    T = len(predictions)
    n_h = len(horizons)
    if weights is None:
        weights = tuple(1.0 / n_h for _ in horizons)

    assert len(weights) == n_h, "weights must match horizons"

    z_scores = np.zeros((n_h, T))

    for i, h in enumerate(horizons):
        # Forward-looking rolling mean (causal: use past h bars)
        smoothed = np.full(T, np.nan)
        for t in range(h, T):
            smoothed[t] = np.mean(predictions[t - h:t])

        # Rolling z-score
        for t in range(h + roll_window, T):
            window = smoothed[t - roll_window:t]
            valid = window[~np.isnan(window)]
            if len(valid) > 2:
                mu_w = valid.mean()
                sig_w = valid.std() + 1e-8
                z_scores[i, t] = (smoothed[t] - mu_w) / sig_w

    # Weighted average across horizons
    agg = np.zeros(T)
    for i, w in enumerate(weights):
        agg += w * z_scores[i]

    return agg


# ============================================================================
# Kalman Filter Signal Generator
# ============================================================================

@dataclass
class KalmanState:
    """State of the Kalman filter at time t."""
    x: float       # filtered state estimate (true signal level)
    P: float       # state uncertainty (estimation error variance)
    K: float       # Kalman gain at this step
    innovation: float  # prediction error (observed - predicted)


class KalmanSignalGenerator:
    """
    Adaptive signal extraction using a scalar Kalman filter.

    Model:
        State:       x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
        Observation: z_t = x_t + v_t,        v_t ~ N(0, R_t)

    where:
        x_t = "true" signal level (unobserved)
        z_t = noisy kernel prediction (observed)
        Q   = process noise — how much the true signal can move per bar
        R_t = observation noise — how noisy the kernel predictions are
              (estimated adaptively from recent prediction variance)

    The s-score is then:
        s_t = (x_t - mu_t) / sigma_t

    where mu_t and sigma_t are rolling mean/std of the filtered state.

    Advantages over OU:
        - Adapts to changing noise levels (R_t estimated online)
        - No assumption of fixed mean-reversion speed
        - Kalman gain automatically balances responsiveness vs smoothness
        - Produces uncertainty estimates (P_t) for position sizing

    Parameters
    ----------
    Q : float
        Process noise variance. Controls how fast the filter tracks.
        Small Q → smoother signal, slower to react.
        Large Q → noisier signal, faster to react.
        Default: estimated from data as 0.1 * Var(predictions).
    R_init : float
        Initial observation noise. Updated adaptively.
    adapt_window : int
        Window for adaptive R estimation.
    zscore_window : int
        Window for computing s-scores from filtered state.
    """

    def __init__(
        self,
        Q: Optional[float] = None,
        R_init: float = 1.0,
        adapt_window: int = 50,
        zscore_window: int = 200,
        Q_array: Optional[np.ndarray] = None,
    ):
        self.Q = Q
        self.R_init = R_init
        self.adapt_window = adapt_window
        self.zscore_window = zscore_window
        self.Q_array = Q_array  # per-bar Q values for adaptive mode

    def filter(
        self,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Kalman filter on kernel predictions.

        Parameters
        ----------
        predictions : array, shape (T,)
            OOS predictions from KRR or elastic-net.

        Returns
        -------
        filtered : array, shape (T,)
            Filtered state estimates x_t.
        gains : array, shape (T,)
            Kalman gains K_t (diagnostic: high = trusting observations).
        uncertainties : array, shape (T,)
            State uncertainties P_t.
        """
        z = np.asarray(predictions, dtype=np.float64)
        T = len(z)

        # Auto-calibrate Q from data if not set
        Q_scalar = self.Q
        if Q_scalar is None:
            Q_scalar = 0.1 * np.nanvar(z)
            if Q_scalar < 1e-16:
                Q_scalar = 1e-10

        # Per-bar adaptive Q if provided
        use_adaptive_Q = (self.Q_array is not None and len(self.Q_array) >= T)

        # Initialize
        x = z[0] if not np.isnan(z[0]) else 0.0  # state
        P = self.R_init                            # uncertainty

        filtered = np.zeros(T)
        gains = np.zeros(T)
        uncertainties = np.zeros(T)
        innovations = np.zeros(T)

        for t in range(T):
            # --- Predict step ---
            Q_t = self.Q_array[t] if use_adaptive_Q else Q_scalar
            x_pred = x          # state prediction (random walk model)
            P_pred = P + Q_t    # uncertainty grows by process noise

            # --- Adaptive observation noise ---
            if t >= self.adapt_window:
                recent = innovations[t - self.adapt_window:t]
                R_t = np.var(recent) + 1e-12
            else:
                R_t = self.R_init

            # --- Update step ---
            innovation = z[t] - x_pred
            S = P_pred + R_t                    # innovation variance
            K = P_pred / S                      # Kalman gain
            x = x_pred + K * innovation         # filtered state
            P = (1 - K) * P_pred                # updated uncertainty

            filtered[t] = x
            gains[t] = K
            uncertainties[t] = P
            innovations[t] = innovation

        return filtered, gains, uncertainties

    def s_scores(
        self,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute s-scores from Kalman-filtered predictions.

        Returns
        -------
        s_scores : array, shape (T,)
            Rolling z-score of filtered state.
        filtered : array, shape (T,)
            Filtered state estimates.
        gains : array, shape (T,)
            Kalman gains.
        uncertainties : array, shape (T,)
            State uncertainties.
        """
        filtered, gains, uncertainties = self.filter(predictions)
        T = len(filtered)
        W = self.zscore_window

        scores = np.full(T, np.nan)
        for t in range(W, T):
            window = filtered[t - W:t]
            mu = window.mean()
            sigma = window.std()
            if sigma > 1e-12:
                scores[t] = (filtered[t] - mu) / sigma
            else:
                scores[t] = 0.0

        return scores, filtered, gains, uncertainties


# ============================================================================
# Signal Diagnostics
# ============================================================================

class SignalDiagnostics:
    """
    Compute per-layer diagnostics for kernel signal quality.

    Metrics:
        - SNR (signal-to-noise ratio): Var(signal) / Var(residual)
        - Autocorrelation of s-scores (should be high for OU process)
        - Per-layer contribution analysis
    """

    @staticmethod
    def layer_snr(
        signal_component: np.ndarray,
        total_signal: np.ndarray,
    ) -> float:
        """
        Signal-to-noise ratio for one kernel layer.

        SNR = Var(signal_component) / Var(total_signal - signal_component)

        Parameters
        ----------
        signal_component : array, shape (T,)
            The signal contribution from one layer.
        total_signal : array, shape (T,)
            The combined signal from all layers.

        Returns
        -------
        snr : float
        """
        residual = total_signal - signal_component
        var_signal = np.nanvar(signal_component)
        var_noise = np.nanvar(residual)

        if var_noise < 1e-12:
            return np.inf if var_signal > 0 else 0.0
        return float(var_signal / var_noise)

    @staticmethod
    def autocorrelation(series: np.ndarray, lag: int = 1) -> float:
        """
        Sample autocorrelation at a given lag.

        A mean-reverting s-score should have negative autocorrelation
        at short lags.
        """
        series = np.asarray(series, dtype=np.float64)
        valid = ~np.isnan(series)
        s = series[valid]
        if len(s) < lag + 2:
            return 0.0
        s = s - np.mean(s)
        n = len(s)
        c0 = np.sum(s ** 2)
        if c0 < 1e-12:
            return 0.0
        ck = np.sum(s[:-lag] * s[lag:])
        return float(ck / c0)

    @staticmethod
    def per_layer_distances(
        kernels: list,
        data,
        rolling_window: int = 100,
    ) -> dict:
        """
        Compute Hilbert distances from mean embedding per layer.

        Returns
        -------
        results : dict mapping kernel.name → {
            'distances': array,
            'snr_vs_combined': float,
        }
        """
        layer_distances = {}
        combined = None

        for kernel in kernels:
            phi = kernel.feature_map(data)
            T = phi.shape[0]
            dists = np.zeros(T)
            for t in range(T):
                start = max(0, t - rolling_window + 1)
                mu = np.mean(phi[start:t + 1], axis=0)
                dists[t] = np.linalg.norm(phi[t] - mu)

            name = getattr(kernel, 'name', str(kernel))
            layer_distances[name] = dists

            if combined is None:
                combined = dists.copy()
            else:
                combined += dists

        # Compute SNR for each layer
        results = {}
        for name, dists in layer_distances.items():
            snr = SignalDiagnostics.layer_snr(dists, combined)
            results[name] = {
                'distances': dists,
                'snr_vs_combined': snr,
            }

        return results
