"""
Momentum Gate — Slow Kernel Layer
===================================
Traders@SMU — Quantitative Strategies Group
Author: Arjan

Role in the architecture
-------------------------
The slow kernel combination uses an additive + Schur product with a
learnable coefficient (regime gate):

    K_slow_gated[i,j] = gate[i] * gate[j] * K_slow[i,j]

This module computes gate[t] in (0, 1) from three momentum signals:

    1. TSMOM        — trailing cumulative log return (long window)
    2. Vol-scaled   — TSMOM / realised vol  (Sharpe-like momentum)
    3. EMA cross    — fast EMA / slow EMA - 1  (trend confirmation)

A Matern RFF kernel ridge regressor maps these features -> forward
returns. The sigmoid of its output is the gate value.

    gate ~ 1  =>  strong trend regime  =>  pass slow kernel signal
    gate ~ 0  =>  choppy / mean-revert =>  attenuate slow kernel signal

This is NOT a standalone kernel layer. Import and use MomentumGate
inside the slow kernel combination / walk-forward engine:

    from momentum_gate import MomentumGate

    gate = MomentumGate()
    gate.fit(prices, target_returns)
    K_gated = gate.apply(K_slow, prices)   # Schur-gated slow kernel
"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional


# =============================================================================
# Momentum feature extraction
# =============================================================================

def momentum_features(prices: np.ndarray,
                      short_window: int = 20,
                      long_window: int = 60,
                      vol_window: int = 20) -> np.ndarray:
    """
    Build a (M, 3) standardised momentum feature matrix.

    Parameters
    ----------
    prices       : 1-D daily close prices, length N
    short_window : EMA fast period
    long_window  : TSMOM + EMA slow period
    vol_window   : rolling realised vol window

    Returns
    -------
    features : np.ndarray shape (M, 3)
        M = N - 1 - max(long_window, vol_window)
        Rows aligned so features[t] uses data up to and including
        log_returns[t + lookback - 1]  (no lookahead).
    """
    prices = np.asarray(prices, dtype=float)
    log_r = np.diff(np.log(prices))           # length N-1
    n = len(log_r)
    lookback = max(long_window, vol_window)
    M = n - lookback
    if M <= 0:
        raise ValueError(
            f"Price series length {len(prices)} too short for "
            f"lookback {lookback}."
        )

    # EMA helper (causal)
    def ema(r, w):
        a = 2.0 / (w + 1)
        out = np.empty(len(r))
        out[0] = r[0]
        for i in range(1, len(r)):
            out[i] = a * r[i] + (1 - a) * out[i - 1]
        return out

    ema_s = ema(log_r, short_window)
    ema_l = ema(log_r, long_window)

    tsmom     = np.empty(M)
    vol_scaled = np.empty(M)
    ema_cross  = np.empty(M)

    for i in range(M):
        t = i + lookback
        tsmom[i]      = log_r[t - long_window: t].sum()
        rv             = log_r[t - vol_window: t].std()
        vol_scaled[i]  = tsmom[i] / (rv + 1e-8)
        ema_cross[i]   = ema_s[t] / (ema_l[t] + 1e-8) - 1.0

    feats = np.column_stack([tsmom, vol_scaled, ema_cross])
    mu, sigma = feats.mean(axis=0), feats.std(axis=0) + 1e-8
    return (feats - mu) / sigma


# =============================================================================
# Matern RFF (same spectral sampling as sentiment_kernel / LOB)
# =============================================================================

class _MaternRFF:
    def __init__(self, nu=1.5, length_scale=1.0,
                 n_features=256, input_dim=3, random_state=42):
        self.nu = nu
        self.length_scale = length_scale
        self.n_features = n_features
        self.input_dim = input_dim
        self._rng = np.random.RandomState(random_state)
        self._sample()

    def _sample(self):
        D, d = self.n_features, self.input_dim
        Z = self._rng.randn(D, d)
        s = np.sqrt(self._rng.gamma(self.nu, 2.0, D))
        self._omega = (Z / s[:, None]) / self.length_scale
        self._bias  = self._rng.uniform(0, 2 * np.pi, D)

    def update(self, nu, length_scale):
        self.nu, self.length_scale = nu, length_scale
        self._sample()

    def transform(self, X):
        return np.sqrt(2.0 / self.n_features) * np.cos(
            X @ self._omega.T + self._bias
        )


# =============================================================================
# Momentum Gate
# =============================================================================

class MomentumGate:
    """
    Regime gate derived from momentum signals.

    Computes gate[t] in (0, 1) via kernel ridge regression on
    three momentum features.  Apply to a slow kernel Gram matrix
    via a Schur product to implement regime gating.

    Parameters
    ----------
    short_window : int    EMA fast period (default 20)
    long_window  : int    TSMOM + EMA slow period (default 60)
    vol_window   : int    Realised vol window (default 20)
    nu           : float  Matern smoothness
    length_scale : float  Kernel length scale
    n_rff        : int    Random Fourier Features
    reg_lambda   : float  KRR regularisation
    random_state : int
    """

    def __init__(self, short_window=20, long_window=60, vol_window=20,
                 nu=1.5, length_scale=1.0, n_rff=256,
                 reg_lambda=1e-3, random_state=42):
        self.short_window  = short_window
        self.long_window   = long_window
        self.vol_window    = vol_window
        self.reg_lambda    = reg_lambda
        self._rff = _MaternRFF(nu=nu, length_scale=length_scale,
                               n_features=n_rff, input_dim=3,
                               random_state=random_state)
        self._w: Optional[np.ndarray] = None
        self._fitted = False

    def _feats(self, prices):
        return momentum_features(prices, self.short_window,
                                 self.long_window, self.vol_window)

    def fit(self, prices: np.ndarray, target_returns: np.ndarray):
        """
        Fit KRR weights to predict forward returns from momentum features.

        Parameters
        ----------
        prices         : 1-D daily close prices, length N
        target_returns : forward log returns aligned to feature rows,
                         length M = N - 1 - max(long_window, vol_window)
        """
        Phi = self._rff.transform(self._feats(prices))   # (M, D)
        y   = np.asarray(target_returns, dtype=float)
        assert len(y) == len(Phi), (
            f"target_returns length {len(y)} != feature rows {len(Phi)}"
        )
        D = Phi.shape[1]
        self._w = np.linalg.solve(
            Phi.T @ Phi + self.reg_lambda * np.eye(D), Phi.T @ y
        )
        self._fitted = True

    def gate_values(self, prices: np.ndarray) -> np.ndarray:
        """
        Return gate scores in (0, 1) for each feature-aligned bar.

        gate[t] = sigmoid( w^T phi(x_t) )

        Parameters
        ----------
        prices : 1-D daily close prices

        Returns
        -------
        gate : np.ndarray shape (M,)
        """
        assert self._fitted, "Call fit() before gate_values()"
        Phi = self._rff.transform(self._feats(prices))
        return 1.0 / (1.0 + np.exp(-Phi @ self._w))

    def optimize(self, prices: np.ndarray, target_returns: np.ndarray,
                 verbose: bool = True) -> dict:
        """
        Optimise (nu, length_scale, reg_lambda) via Gaussian NLL.

        Returns
        -------
        dict  {"nu": ..., "length_scale": ..., "reg_lambda": ...}
        """
        feats = self._feats(prices)
        y = np.asarray(target_returns, dtype=float)

        def nll(params):
            nu, ell, lam = params
            if nu <= 0 or ell <= 0 or lam <= 0:
                return 1e10
            self._rff.update(nu, ell)
            Phi = self._rff.transform(feats)
            n = len(y)
            K = Phi @ Phi.T + lam * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                return (np.sum(np.log(np.diag(L)))
                        + 0.5 * float(y @ alpha)
                        + 0.5 * n * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                return 1e10

        x0     = [self._rff.nu, self._rff.length_scale, self.reg_lambda]
        bounds = [(0.1, 10.0), (0.001, None), (1e-6, None)]
        res    = minimize(nll, x0, bounds=bounds, method="L-BFGS-B")
        nu, ell, lam = res.x
        self._rff.update(nu, ell)
        self.reg_lambda = lam
        if verbose:
            print(f"MomentumGate optimised: nu={nu:.3f}, "
                  f"ell={ell:.4f}, lambda={lam:.2e}")
        return {"nu": nu, "length_scale": ell, "reg_lambda": lam}

    def apply(self, K_slow: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """
        Apply momentum gate to a slow-layer Gram matrix via Schur product.

            K_gated[i,j] = sqrt(gate[i] * gate[j]) * K_slow[i,j]

        Preserves PSD: Schur product of two PSD matrices is PSD.

        Parameters
        ----------
        K_slow : (M, M) slow kernel Gram matrix
        prices : daily close prices (same series used in fit)

        Returns
        -------
        K_gated : (M, M)
        """
        g = self.gate_values(prices)
        return K_slow * np.outer(np.sqrt(g), np.sqrt(g))


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    rng = np.random.RandomState(7)
    N   = 300
    prices = 100.0 * np.exp(np.cumsum(rng.normal(5e-4, 0.015, N)))

    gate     = MomentumGate(short_window=10, long_window=40,
                            vol_window=20, n_rff=128)
    lookback = max(gate.long_window, gate.vol_window)
    log_r    = np.diff(np.log(prices))
    M        = N - 1 - lookback
    # one-step-ahead forward returns aligned to feature rows
    target   = log_r[lookback: lookback + M]
    p_train  = prices[: lookback + M + 1]

    gate.optimize(p_train, target)
    gate.fit(p_train, target)

    g = gate.gate_values(p_train)
    print(f"Gate shape : {g.shape}  range : [{g.min():.4f}, {g.max():.4f}]  "
          f"mean : {g.mean():.4f}")

    K_slow   = np.corrcoef(np.random.randn(M, M))    # dummy slow kernel
    K_gated  = gate.apply(K_slow, p_train)
    print(f"K_gated shape : {K_gated.shape}")
    print("Smoke test passed.")
