"""
Limit Order Book (LOB) Layer — RKHS Kernel Module
===================================================
Traders@SMU — Quantitative Strategies Group
Layer 3: Limit Order Book (Becky & Connor)

Architecture Overview
---------------------
The full RKHS model sums kernel outputs across three layers:

    K_total(x, y) = α₁ K_GARCH(x, y)        <- Group 1 (Anders, Zabe, Ameen)
                  + α₂ K_Sentiment(x, y)      <- Group 2 (Will, Demetrios, Arjan)
                  + α₃ K_LOB(x, y)            <- Group 3 (Becky, Connor)   *** THIS FILE ***

Within each layer, the layer kernel is itself a weighted sum of 2-3 sub-kernels:

    K_LOB(x, y) = w₁ k₁(x, y) + w₂ k₂(x, y) + w₃ k₃(x, y)

Kernel Implementation
---------------------
All kernels use the Matern kernel via Random Fourier Features (RFF),
matching Group 1's GARCH architecture. This gives us:

    1. Explicit feature maps: phi(x) in R^D  (for Hilbert space composition)
    2. Matern flexibility: smoothness parameter nu + length scale ell
    3. Kernel ridge regression: fit(X, y) for prediction tasks
    4. Hyperparameter optimization: L-BFGS-B on (nu, ell, lambda) via NLL

The kernel is approximated as: K(x, x') ~ phi(x)^T phi(x')
where phi(x) = sqrt(2/D) * cos(X @ omega^T + b) and omega is sampled
from the spectral density of the Matern kernel (Rahimi & Recht 2007).

Legacy RBF mode is available via use_rff=False for backward compatibility.

Event‑Time Sampling (Dollar Bars)
----------------------------------
All LOB snapshots are assumed to be generated in **event time** (dollar bars).
A new snapshot is created whenever a fixed dollar volume threshold is reached.
This stabilises the arrival rate of information and aligns with the
microstructure modelling philosophy described in the design document.

The `timestamp` field in `LOBSnapshot` therefore represents the **bar index**
(an integer) rather than chronological time. For the `TimeScaleKernel`,
the `window` parameter denotes the number of *dollar bars* to average over.

Two Operating Modes
-------------------
Mode 1 -- "aspect" (default, works with single snapshots):
    Each kernel captures a DIFFERENT PROPERTY of the same LOB state:
    1. VolumeProfileKernel  -- similarity of liquidity distribution across levels
    2. BookShapeKernel      -- similarity of price-level spacing / microstructure
    3. DepthImbalanceKernel -- similarity of buy vs sell pressure at each level

Mode 2 -- "timescale" (requires time-series data, i.e., LOBSeries):
    Each kernel captures ALL properties but at a DIFFERENT TIME HORIZON,
    where "time" is measured in dollar bars:
    1. TimeScaleKernel(window=1)   -- instant (current dollar bar)
    2. TimeScaleKernel(window=10)  -- short-term average (10 dollar bars)
    3. TimeScaleKernel(window=50)  -- medium-term average (50 dollar bars)
    (Connor's idea -- activate when streaming LOB data is available)

Use create_lob_layer(mode="aspect") or create_lob_layer(mode="timescale")
to toggle between them. Both produce valid PSD kernels.

References
----------
    - RKHS theory: https://teazrq.github.io/SMLR/reproducing-kernel-hilbert-space.html
    - Random Fourier Features: Rahimi & Recht (2007)
    - Kernel combination: K_sum = w₁K₁ + w₂K₂ + ... (valid PSD by closure)
"""

import numpy as np
from typing import Optional, List
from abc import ABC, abstractmethod
from scipy.special import kv, gamma as gamma_func
from scipy.optimize import minimize


# =============================================================================
# LOB State Representation (Dollar‑Bar Aware)
# =============================================================================

class LOBSnapshot:
    """
    Represents a single snapshot of a Limit Order Book.

    The LOB is captured as the top L levels on each side (bid and ask).

    **Event‑Time Note**: The `timestamp` field should contain the **dollar bar index**
    (an integer) when this snapshot was generated. This ensures that windows in
    `TimeScaleKernel` correspond to a fixed number of dollar bars.

    Parameters
    ----------
    bid_prices : array-like, shape (L,)
        Bid prices from best (highest) to worst (lowest).
    bid_volumes : array-like, shape (L,)
        Volume at each bid price level.
    ask_prices : array-like, shape (L,)
        Ask prices from best (lowest) to worst (highest).
    ask_volumes : array-like, shape (L,)
        Volume at each ask price level.
    timestamp : int, optional
        Dollar bar index (event time). Defaults to None.

    Example
    -------
    >>> snap = LOBSnapshot(
    ...     bid_prices=[100.00, 99.95, 99.90, 99.85, 99.80],
    ...     bid_volumes=[150, 200, 300, 100, 500],
    ...     ask_prices=[100.05, 100.10, 100.15, 100.20, 100.25],
    ...     ask_volumes=[120, 250, 180, 400, 350],
    ...     timestamp=42,  # 42nd dollar bar
    ... )
    """

    def __init__(self, bid_prices, bid_volumes, ask_prices, ask_volumes,
                 timestamp: Optional[int] = None):
        self.bid_prices = np.asarray(bid_prices, dtype=np.float64)
        self.bid_volumes = np.asarray(bid_volumes, dtype=np.float64)
        self.ask_prices = np.asarray(ask_prices, dtype=np.float64)
        self.ask_volumes = np.asarray(ask_volumes, dtype=np.float64)
        self.timestamp = timestamp

        assert len(self.bid_prices) == len(self.bid_volumes), \
            "bid_prices and bid_volumes must have the same length"
        assert len(self.ask_prices) == len(self.ask_volumes), \
            "ask_prices and ask_volumes must have the same length"

    @property
    def n_levels(self):
        """Number of price levels on each side."""
        return len(self.bid_prices)

    @property
    def mid_price(self):
        """Mid-price = average of best bid and best ask."""
        return (self.bid_prices[0] + self.ask_prices[0]) / 2.0

    @property
    def spread(self):
        """Bid-ask spread = best ask - best bid."""
        return self.ask_prices[0] - self.bid_prices[0]

    @property
    def imbalance(self):
        """
        Volume imbalance at the top of book.
        Positive = more bid volume (buying pressure).
        Range: [-1, 1]
        """
        vb = self.bid_volumes[0]
        va = self.ask_volumes[0]
        return (vb - va) / (vb + va) if (vb + va) > 0 else 0.0

    def __repr__(self):
        return (f"LOBSnapshot(mid={self.mid_price:.2f}, "
                f"spread={self.spread:.4f}, "
                f"imbalance={self.imbalance:.3f}, "
                f"levels={self.n_levels}, bar={self.timestamp})")


# =============================================================================
# LOB Time Series Container (for timescale mode, using dollar bars)
# =============================================================================

class LOBSeries:
    """
    A time-ordered sequence of LOB snapshots for a single observation point.

    In timescale mode, a single "data point" isn't one snapshot -- it's a
    WINDOW of recent snapshots. The TimeScaleKernel averages over these
    windows before computing similarity.

    **Event‑Time Note**: The snapshots are ordered by dollar bar index.
    The `window` parameter in `TimeScaleKernel` refers to the number of
    consecutive dollar bars to include.

    Parameters
    ----------
    snapshots : list of LOBSnapshot
        Time-ordered snapshots (oldest first, newest last), each corresponding
        to a distinct dollar bar. Must all have the same n_levels.
    """

    def __init__(self, snapshots: List[LOBSnapshot]):
        assert len(snapshots) > 0, "LOBSeries must contain at least 1 snapshot"
        self.snapshots = snapshots
        self.n_levels = snapshots[0].n_levels

    def window(self, size: int) -> List[LOBSnapshot]:
        """
        Get the most recent `size` snapshots (i.e., the last `size` dollar bars).
        If fewer than `size` exist, returns all available (graceful fallback).
        """
        return self.snapshots[-size:]

    def __len__(self):
        return len(self.snapshots)

    def __repr__(self):
        first_bar = self.snapshots[0].timestamp
        last_bar = self.snapshots[-1].timestamp
        return (f"LOBSeries(length={len(self)}, levels={self.n_levels}, "
                f"bars=[{first_bar}..{last_bar}])")


# =============================================================================
# Matern Random Fourier Features (Explicit Feature Map)
# =============================================================================

class MaternRFF:
    """
    Explicit D-dimensional feature map for the Matern kernel via
    Random Fourier Features (Rahimi & Recht 2007).

    The Matern kernel is shift-invariant, so by Bochner's theorem it has
    a spectral density. For Matern(nu, ell) in 1D, the spectral density
    is a scaled Student-t with df=2*nu, scale=sqrt(2*nu)/ell.

    RFF approximation:
        Draw omega_j ~ p(omega), b_j ~ Uniform(0, 2*pi)
        phi(x) = sqrt(2/D) * [cos(omega_1^T x + b_1), ..., cos(omega_D^T x + b_D)]
        K(x, x') ~ phi(x)^T phi(x')

    Adapted from Group 1's MaternRFF (Anders, Zabe, Ameen).

    Parameters
    ----------
    nu : float
        Matern smoothness parameter. Higher = smoother kernel.
        nu=0.5 -> Laplacian, nu=1.5 -> once differentiable, nu->inf -> RBF
    length_scale : float
        Controls how quickly similarity decays with distance.
    n_features : int
        Number of random Fourier features (D). Higher = better approximation.
    d_input : int
        Dimensionality of the input feature vectors.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, nu=1.5, length_scale=1.0, n_features=500,
                 d_input=1, seed=42):
        self.nu = nu
        self.length_scale = length_scale
        self.D = n_features
        self.d_input = d_input
        self.rng = np.random.RandomState(seed)
        self._sample_frequencies()

    def _sample_frequencies(self):
        """
        Sample frequencies from the spectral measure of Matern.

        In d dimensions the spectral density is:
            p(omega) ~ (2*nu/ell^2 + ||omega||^2)^{-(nu + d/2)}

        In 1D this is Student-t with df=2*nu, scale = sqrt(2*nu)/ell.
        For isotropic Matern in d dimensions, each component is sampled
        independently from 1D Student-t (standard RFF approach).
        """
        df = 2.0 * self.nu
        scale = np.sqrt(2.0 * self.nu) / self.length_scale
        # Each row omega_j is a frequency vector in R^{d_input}
        self.omega = self.rng.standard_t(df=df,
                                          size=(self.D, self.d_input)) * scale
        self.b = self.rng.uniform(0, 2 * np.pi, size=self.D)

    def update_params(self, nu, length_scale):
        """Re-sample frequencies when hyperparameters change."""
        self.nu = nu
        self.length_scale = length_scale
        self._sample_frequencies()

    def transform(self, X):
        """
        Compute the explicit feature map phi(X).

        Parameters
        ----------
        X : np.ndarray, shape (n, d_input) or (d_input,)
            Input feature vectors.

        Returns
        -------
        Phi : np.ndarray, shape (n, D)
            Feature matrix such that Phi @ Phi^T ~ K (the Gram matrix).
        """
        X = np.atleast_2d(X)
        # X @ omega^T : (n, D)
        proj = X @ self.omega.T + self.b[np.newaxis, :]
        return np.sqrt(2.0 / self.D) * np.cos(proj)


# =============================================================================
# Base Kernel Class (shared interface for ALL groups)
# =============================================================================

class BaseKernel(ABC):
    """
    Abstract base class for all RKHS kernels in the Traders@SMU model.

    Every kernel (across all 3 groups) should implement this interface
    so that the layer summation K_total = sum alpha_i K_i works seamlessly.

    Supports two modes:
        use_rff=True  (default) -- Matern kernel via Random Fourier Features
        use_rff=False           -- Legacy RBF kernel (backward compatible)

    Required methods (subclasses must implement):
        - extract_features(snapshot) -> np.ndarray
        - name (property) -> str
        - hyperparameters (property) -> dict
        - _feature_dim (property) -> int

    The evaluate() method is now provided by BaseKernel using RFF.
    """

    @abstractmethod
    def extract_features(self, snapshot: LOBSnapshot) -> np.ndarray:
        """
        Extract the relevant feature vector from a LOB snapshot.
        Each sub-kernel extracts DIFFERENT features from the same snapshot.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable kernel name."""
        pass

    @property
    @abstractmethod
    def hyperparameters(self) -> dict:
        """Return dict of hyperparameter names -> values."""
        pass

    @property
    @abstractmethod
    def _feature_dim(self) -> int:
        """Dimensionality of the raw feature vector from extract_features()."""
        pass

    # --- RFF infrastructure (initialized by subclass __init__) ---

    def _init_rff(self):
        """Lazily initialize the Random Fourier Feature map."""
        if not hasattr(self, 'rff') or self.rff is None:
            seed = getattr(self, '_rff_seed', 42)
            self.rff = MaternRFF(
                nu=self.nu,
                length_scale=self.length_scale,
                n_features=self.n_rff,
                d_input=self._feature_dim,
                seed=seed,
            )

    def _extract_feature_matrix(self, snapshots) -> np.ndarray:
        """Batch-extract raw features into an (n, d_input) matrix."""
        features = []
        for s in snapshots:
            if isinstance(s, (LOBSnapshot, LOBSeries)):
                features.append(self.extract_features(s))
            else:
                features.append(np.asarray(s, dtype=np.float64))
        return np.array(features)

    # --- Core kernel operations ---

    def evaluate(self, x_feat: np.ndarray, y_feat: np.ndarray) -> float:
        """
        Compute kernel value between two feature vectors.

        When use_rff=True:  k(x,y) ~ phi(x)^T phi(y)  (Matern via RFF)
        When use_rff=False: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))  (RBF)
        """
        if getattr(self, 'use_rff', True):
            self._init_rff()
            phi_x = self.rff.transform(np.atleast_2d(x_feat))  # (1, D)
            phi_y = self.rff.transform(np.atleast_2d(y_feat))  # (1, D)
            return float((phi_x @ phi_y.T)[0, 0])
        else:
            return self._rbf(x_feat, y_feat, self.sigma)

    def __call__(self, x, y) -> float:
        """
        Compute kernel value. Accepts LOBSnapshot, LOBSeries, or raw feature vectors.
        """
        if isinstance(x, (LOBSnapshot, LOBSeries)):
            x = self.extract_features(x)
        if isinstance(y, (LOBSnapshot, LOBSeries)):
            y = self.extract_features(y)
        return self.evaluate(np.asarray(x, dtype=np.float64),
                             np.asarray(y, dtype=np.float64))

    def feature_map(self, snapshots) -> np.ndarray:
        """
        Compute explicit feature map phi(X) in R^D.

        This is the key capability that matches Group 1's architecture.
        The feature vectors live in a finite-dimensional approximation
        of the RKHS, and K(x,y) ~ phi(x)^T phi(y).

        Parameters
        ----------
        snapshots : list of LOBSnapshot, LOBSeries, or np.ndarray

        Returns
        -------
        Phi : np.ndarray, shape (n, D)
            Feature matrix where Phi @ Phi^T approximates the Gram matrix.
        """
        self._init_rff()
        X = self._extract_feature_matrix(snapshots)
        return self.rff.transform(X)

    def gram_matrix(self, snapshots: list) -> np.ndarray:
        """
        Compute the Gram matrix K_ij = k(x_i, x_j).

        When use_rff=True:  K = Phi @ Phi^T  (PSD by construction, O(nD))
        When use_rff=False: K_ij = rbf(x_i, x_j) (O(n^2 d) pairwise)

        Returns
        -------
        K : np.ndarray, shape (N, N)
            Symmetric, positive semi-definite matrix.
        """
        if getattr(self, 'use_rff', True):
            Phi = self.feature_map(snapshots)
            return Phi @ Phi.T
        else:
            # Legacy RBF path
            features = []
            for s in snapshots:
                if isinstance(s, (LOBSnapshot, LOBSeries)):
                    features.append(self.extract_features(s))
                else:
                    features.append(np.asarray(s, dtype=np.float64))

            n = len(features)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    val = self.evaluate(features[i], features[j])
                    K[i, j] = val
                    K[j, i] = val
            return K

    def hilbert_distance(self, x, y) -> float:
        """
        Compute ||phi(x) - phi(y)||_H in the RKHS.

        Measures how different two LOB states are in the Hilbert space.
        Requires use_rff=True.
        """
        self._init_rff()
        phi_x = self.feature_map([x])  # (1, D)
        phi_y = self.feature_map([y])  # (1, D)
        return float(np.linalg.norm(phi_x - phi_y))

    def fit(self, snapshots, y):
        """
        Fit kernel ridge regression in primal form.

        w* = (Phi^T Phi + lambda I)^{-1} Phi^T y

        This matches Group 1's GARCH layer pattern exactly.

        Parameters
        ----------
        snapshots : list of LOBSnapshot
            Training data.
        y : np.ndarray, shape (n,)
            Regression targets.
        """
        self._init_rff()
        self.Phi_train = self.feature_map(snapshots)  # (n, D)
        D = self.n_rff
        reg = getattr(self, 'reg_lambda', 1e-3)
        PhiTPhi = self.Phi_train.T @ self.Phi_train  # (D, D)
        PhiTy = self.Phi_train.T @ y                  # (D,)
        self.w = np.linalg.solve(
            PhiTPhi + reg * np.eye(D), PhiTy
        )
        self._fitted = True

    def predict(self, snapshots):
        """
        Predict using fitted weights: f(x) = phi(x) @ w

        Parameters
        ----------
        snapshots : list of LOBSnapshot

        Returns
        -------
        predictions : np.ndarray, shape (n,)
        """
        assert getattr(self, '_fitted', False), "Must call fit() before predict()"
        Phi_new = self.feature_map(snapshots)
        return Phi_new @ self.w

    def is_psd(self, snapshots: list, tol: float = 1e-10):
        """
        Verify this kernel produces a positive semi-definite Gram matrix.

        Returns
        -------
        is_valid : bool
        min_eigenvalue : float
        """
        K = self.gram_matrix(snapshots)
        eigenvalues = np.linalg.eigvalsh(K)
        min_eig = eigenvalues.min()
        return min_eig >= -tol, min_eig

    @staticmethod
    def _rbf(x, y, sigma):
        """
        Gaussian RBF kernel: k(x,y) = exp( -||x-y||^2 / (2*sigma^2) )

        Legacy kernel function. Used when use_rff=False.
        """
        diff = x - y
        return np.exp(-np.dot(diff, diff) / (2.0 * sigma ** 2))


# =============================================================================
# LOB Sub-Kernel 1: Volume Profile Kernel
# =============================================================================

class VolumeProfileKernel(BaseKernel):
    """
    Measures similarity of LIQUIDITY DISTRIBUTION across price levels.

    What it captures:
        "Do these two order books have similar amounts of volume
         at each level?"

    Feature vector: [bid_vol_1/total, ..., bid_vol_L/total,
                     ask_vol_1/total, ..., ask_vol_L/total]
    Dimension: 2L (default 10)

    Parameters
    ----------
    n_levels : int
        Number of price levels on each side (default 5).
    sigma : float
        RBF bandwidth (used when use_rff=False). Default 0.5.
    nu : float
        Matern smoothness. Default 1.5.
    length_scale : float or None
        Matern length scale. None = use sigma.
    n_rff : int
        Number of Random Fourier Features. Default 500.
    reg_lambda : float
        Ridge regression regularization. Default 1e-3.
    use_rff : bool
        True = Matern RFF (default). False = legacy RBF.
    seed : int
        Random seed for RFF frequency sampling.
    """

    def __init__(self, n_levels: int = 5, sigma: float = 0.5,
                 nu: float = 1.5, length_scale: Optional[float] = None,
                 n_rff: int = 500, reg_lambda: float = 1e-3,
                 use_rff: bool = True, seed: int = 42):
        self.n_levels = n_levels
        self.sigma = sigma
        self.nu = nu
        self.length_scale = length_scale if length_scale is not None else sigma
        self.n_rff = n_rff
        self.reg_lambda = reg_lambda
        self.use_rff = use_rff
        self._rff_seed = seed
        self.rff = None
        self.w = None
        self.Phi_train = None
        self._fitted = False

    @property
    def _feature_dim(self):
        return 2 * self.n_levels

    @property
    def name(self):
        return "VolumeProfile"

    @property
    def hyperparameters(self):
        base = {"n_levels": self.n_levels}
        if self.use_rff:
            base.update({"nu": self.nu, "length_scale": self.length_scale,
                         "n_rff": self.n_rff, "mode": "matern_rff"})
        else:
            base.update({"sigma": self.sigma, "mode": "rbf"})
        return base

    def extract_features(self, snapshot: LOBSnapshot) -> np.ndarray:
        """Extract normalized volume distribution from both sides of book."""
        bid_vols = snapshot.bid_volumes[:self.n_levels].copy()
        ask_vols = snapshot.ask_volumes[:self.n_levels].copy()

        # Normalize by total volume -> distribution that sums to 1
        total = bid_vols.sum() + ask_vols.sum()
        if total > 0:
            bid_vols /= total
            ask_vols /= total

        return np.concatenate([bid_vols, ask_vols])

    def __repr__(self):
        if self.use_rff:
            return (f"VolumeProfileKernel(levels={self.n_levels}, "
                    f"nu={self.nu}, ell={self.length_scale:.4f}, D={self.n_rff})")
        return f"VolumeProfileKernel(levels={self.n_levels}, sigma={self.sigma})"


# =============================================================================
# LOB Sub-Kernel 2: Book Shape Kernel
# =============================================================================

class BookShapeKernel(BaseKernel):
    """
    Measures similarity of ORDER BOOK SHAPE / MICROSTRUCTURE.

    What it captures:
        "Do these two order books have similar spacing between
         price levels?"

    Feature vector: price offsets from mid in basis points + spread
        [(bid_1 - mid)/mid * 10000, ..., spread_bps]
    Dimension: 2L + 1 (default 11)

    When use_rff=False and sigma="auto", uses the median heuristic for
    automatic bandwidth selection (legacy mode).

    Parameters
    ----------
    n_levels : int
        Number of price levels on each side (default 5).
    sigma : float or "auto"
        RBF bandwidth (used when use_rff=False). "auto" = median heuristic.
    nu : float
        Matern smoothness. Default 1.5.
    length_scale : float or None
        Matern length scale. None = auto-detect from data.
    n_rff : int
        Number of Random Fourier Features. Default 500.
    reg_lambda : float
        Ridge regression regularization. Default 1e-3.
    use_rff : bool
        True = Matern RFF (default). False = legacy RBF.
    seed : int
        Random seed for RFF frequency sampling.
    """

    def __init__(self, n_levels: int = 5, sigma=1.0,
                 nu: float = 1.5, length_scale: Optional[float] = None,
                 n_rff: int = 500, reg_lambda: float = 1e-3,
                 use_rff: bool = True, seed: int = 43):
        self.n_levels = n_levels
        self._sigma_setting = sigma
        self._calibrated_sigma = None
        self.nu = nu
        self.length_scale = length_scale if length_scale is not None else 1.0
        self.n_rff = n_rff
        self.reg_lambda = reg_lambda
        self.use_rff = use_rff
        self._rff_seed = seed
        self.rff = None
        self.w = None
        self.Phi_train = None
        self._fitted = False

    @property
    def _feature_dim(self):
        return 2 * self.n_levels + 1

    @property
    def sigma(self):
        """RBF sigma (used in legacy mode only)."""
        if self._calibrated_sigma is not None:
            return self._calibrated_sigma
        if self._sigma_setting == "auto":
            return 1.0
        return float(self._sigma_setting)

    @property
    def name(self):
        return "BookShape"

    @property
    def hyperparameters(self):
        base = {"n_levels": self.n_levels}
        if self.use_rff:
            base.update({"nu": self.nu, "length_scale": self.length_scale,
                         "n_rff": self.n_rff, "mode": "matern_rff"})
        else:
            base.update({"sigma": self.sigma,
                         "auto": self._sigma_setting == "auto",
                         "mode": "rbf"})
        return base

    def extract_features(self, snapshot: LOBSnapshot) -> np.ndarray:
        """
        Extract price offsets from mid, normalized by mid-price (basis points).

        Feature vector: [(bid_1 - mid)/mid * 10000, ..., spread_bps]
        """
        mid = snapshot.mid_price if snapshot.mid_price > 0 else 1e-8

        bid_offsets = (snapshot.bid_prices[:self.n_levels] - mid) / mid * 10000
        ask_offsets = (snapshot.ask_prices[:self.n_levels] - mid) / mid * 10000
        spread_bps = snapshot.spread / mid * 10000

        return np.concatenate([bid_offsets, ask_offsets, [spread_bps]])

    def _calibrate_sigma(self, features: list):
        """Median heuristic for RBF bandwidth (legacy mode only)."""
        n = len(features)
        if n < 2:
            self._calibrated_sigma = 1.0
            return

        feats = np.array(features)
        sq_norms = (feats ** 2).sum(axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * feats @ feats.T
        np.fill_diagonal(dist_sq, 0)
        dist_sq = np.maximum(dist_sq, 0)

        upper = dist_sq[np.triu_indices(n, k=1)]
        median_dist = np.sqrt(np.median(upper))
        self._calibrated_sigma = max(median_dist / np.sqrt(2 * np.log(2)), 1e-10)

    def gram_matrix(self, snapshots: list) -> np.ndarray:
        """
        Compute Gram matrix.

        RFF mode: Phi @ Phi^T (efficient, PSD by construction)
        RBF mode: auto-calibrate sigma if "auto", then pairwise computation
        """
        if self.use_rff:
            # Use the efficient RFF path from BaseKernel
            Phi = self.feature_map(snapshots)
            return Phi @ Phi.T

        # Legacy RBF path with median heuristic
        features = []
        for s in snapshots:
            if isinstance(s, (LOBSnapshot, LOBSeries)):
                features.append(self.extract_features(s))
            else:
                features.append(np.asarray(s, dtype=np.float64))

        if self._sigma_setting == "auto":
            self._calibrate_sigma(features)

        n = len(features)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._rbf(features[i], features[j], self.sigma)
                K[j, i] = K[i, j]
        return K

    def __repr__(self):
        if self.use_rff:
            return (f"BookShapeKernel(levels={self.n_levels}, "
                    f"nu={self.nu}, ell={self.length_scale:.4f}, D={self.n_rff})")
        sigma_str = f"auto->{self.sigma:.6f}" if self._sigma_setting == "auto" else f"{self.sigma}"
        return f"BookShapeKernel(levels={self.n_levels}, sigma={sigma_str})"


# =============================================================================
# LOB Sub-Kernel 3: Depth Imbalance Kernel
# =============================================================================

class DepthImbalanceKernel(BaseKernel):
    """
    Measures similarity of BUY vs SELL PRESSURE across the book.

    What it captures:
        "Do these two order books have similar directional pressure
         at each depth level?"

    Feature vector: per-level imbalance
        imbalance_i = (bid_vol_i - ask_vol_i) / (bid_vol_i + ask_vol_i)
        Each value in [-1, 1]
    Dimension: L (default 5)

    Parameters
    ----------
    n_levels : int
        Number of price levels on each side (default 5).
    sigma : float
        RBF bandwidth (used when use_rff=False). Default 0.8.
    nu : float
        Matern smoothness. Default 1.5.
    length_scale : float or None
        Matern length scale. None = use sigma.
    n_rff : int
        Number of Random Fourier Features. Default 500.
    reg_lambda : float
        Ridge regression regularization. Default 1e-3.
    use_rff : bool
        True = Matern RFF (default). False = legacy RBF.
    seed : int
        Random seed for RFF frequency sampling.
    """

    def __init__(self, n_levels: int = 5, sigma: float = 0.8,
                 nu: float = 1.5, length_scale: Optional[float] = None,
                 n_rff: int = 500, reg_lambda: float = 1e-3,
                 use_rff: bool = True, seed: int = 44):
        self.n_levels = n_levels
        self.sigma = sigma
        self.nu = nu
        self.length_scale = length_scale if length_scale is not None else sigma
        self.n_rff = n_rff
        self.reg_lambda = reg_lambda
        self.use_rff = use_rff
        self._rff_seed = seed
        self.rff = None
        self.w = None
        self.Phi_train = None
        self._fitted = False

    @property
    def _feature_dim(self):
        return self.n_levels

    @property
    def name(self):
        return "DepthImbalance"

    @property
    def hyperparameters(self):
        base = {"n_levels": self.n_levels}
        if self.use_rff:
            base.update({"nu": self.nu, "length_scale": self.length_scale,
                         "n_rff": self.n_rff, "mode": "matern_rff"})
        else:
            base.update({"sigma": self.sigma, "mode": "rbf"})
        return base

    def extract_features(self, snapshot: LOBSnapshot) -> np.ndarray:
        """Extract per-level buy/sell imbalance."""
        bid_vols = snapshot.bid_volumes[:self.n_levels]
        ask_vols = snapshot.ask_volumes[:self.n_levels]

        denom = bid_vols + ask_vols
        imbalance = np.where(denom > 0, (bid_vols - ask_vols) / denom, 0.0)

        return imbalance

    def __repr__(self):
        if self.use_rff:
            return (f"DepthImbalanceKernel(levels={self.n_levels}, "
                    f"nu={self.nu}, ell={self.length_scale:.4f}, D={self.n_rff})")
        return f"DepthImbalanceKernel(levels={self.n_levels}, sigma={self.sigma})"


# =============================================================================
# Timescale Kernel (Connor's approach -- Mode 2, using dollar bars)
# =============================================================================

class TimeScaleKernel(BaseKernel):
    """
    Captures LOB similarity at a SPECIFIC TIME HORIZON, where time is measured
    in **dollar bars** (event time).

    Connor's idea: instead of separate kernels for volume/shape/imbalance,
    use separate kernels for different lookback windows. Each timescale
    kernel uses ALL three feature types (volume + shape + imbalance),
    but averaged over a rolling window of the specified number of dollar bars.

    Feature dimension: 5 * n_levels + 1 (default 26)

    Parameters
    ----------
    n_levels : int
        Number of price levels per side (default 5).
    window : int
        Number of **dollar bars** to average over (default 1 = no averaging).
    sigma : float
        RBF bandwidth (used when use_rff=False). Default 1.0.
    nu : float
        Matern smoothness. Default 1.5.
    length_scale : float or None
        Matern length scale. None = use sigma.
    n_rff : int
        Number of Random Fourier Features. Default 500.
    reg_lambda : float
        Ridge regression regularization. Default 1e-3.
    use_rff : bool
        True = Matern RFF (default). False = legacy RBF.
    label : str, optional
        Human-readable name for this timescale.
    seed : int
        Random seed for RFF frequency sampling.
    """

    def __init__(self, n_levels: int = 5, window: int = 1,
                 sigma: float = 1.0, label: Optional[str] = None,
                 nu: float = 1.5, length_scale: Optional[float] = None,
                 n_rff: int = 500, reg_lambda: float = 1e-3,
                 use_rff: bool = True, seed: int = 45):
        self.n_levels = n_levels
        self.window = window
        self.sigma = sigma
        self._label = label or f"w{window}"
        self.nu = nu
        self.length_scale = length_scale if length_scale is not None else sigma
        self.n_rff = n_rff
        self.reg_lambda = reg_lambda
        self.use_rff = use_rff
        self._rff_seed = seed
        self.rff = None
        self.w = None
        self.Phi_train = None
        self._fitted = False

    @property
    def _feature_dim(self):
        return 5 * self.n_levels + 1

    @property
    def name(self):
        return f"TimeScale({self._label})"

    @property
    def hyperparameters(self):
        base = {"n_levels": self.n_levels, "window": self.window}
        if self.use_rff:
            base.update({"nu": self.nu, "length_scale": self.length_scale,
                         "n_rff": self.n_rff, "mode": "matern_rff"})
        else:
            base.update({"sigma": self.sigma, "mode": "rbf"})
        return base

    def _full_features_from_snapshot(self, snapshot: LOBSnapshot) -> np.ndarray:
        """
        Extract ALL feature types from a single snapshot:
            [normalized_volumes (2L), price_offsets_bps (2L), spread_bps (1), imbalances (L)]
        Total dimension: 5 * n_levels + 1
        """
        L = self.n_levels
        mid = snapshot.mid_price if snapshot.mid_price > 0 else 1e-8

        # Volume features (normalized)
        bid_vols = snapshot.bid_volumes[:L].copy()
        ask_vols = snapshot.ask_volumes[:L].copy()
        total_vol = bid_vols.sum() + ask_vols.sum()
        if total_vol > 0:
            bid_vols /= total_vol
            ask_vols /= total_vol

        # Shape features (price offsets from mid in basis points)
        bid_offsets = (snapshot.bid_prices[:L] - mid) / mid * 10000
        ask_offsets = (snapshot.ask_prices[:L] - mid) / mid * 10000
        spread_bps = snapshot.spread / mid * 10000

        # Imbalance features
        denom = snapshot.bid_volumes[:L] + snapshot.ask_volumes[:L]
        imbalance = np.where(
            denom > 0,
            (snapshot.bid_volumes[:L] - snapshot.ask_volumes[:L]) / denom,
            0.0
        )

        return np.concatenate([bid_vols, ask_vols,
                               bid_offsets, ask_offsets,
                               [spread_bps],
                               imbalance])

    def extract_features(self, data) -> np.ndarray:
        """
        Extract time-averaged feature vector over the last `self.window` dollar bars.

        Parameters
        ----------
        data : LOBSnapshot, LOBSeries, or np.ndarray
            - LOBSnapshot: uses it directly (window=1 behavior)
            - LOBSeries: averages over the last `self.window` snapshots
            - np.ndarray: passes through as-is

        Returns
        -------
        features : np.ndarray, shape (5 * n_levels + 1,)
        """
        if isinstance(data, LOBSnapshot):
            return self._full_features_from_snapshot(data)

        if isinstance(data, LOBSeries):
            recent = data.window(self.window)
            feat_vectors = [self._full_features_from_snapshot(s)
                            for s in recent]
            return np.mean(feat_vectors, axis=0)

        return np.asarray(data, dtype=np.float64)

    def __repr__(self):
        if self.use_rff:
            return (f"TimeScaleKernel({self._label}, "
                    f"window={self.window} bars, nu={self.nu}, "
                    f"ell={self.length_scale:.4f}, D={self.n_rff})")
        return (f"TimeScaleKernel({self._label}, "
                f"window={self.window} bars, sigma={self.sigma})")


# =============================================================================
# Hyperparameter Optimization
# =============================================================================

def optimize_kernel_hyperparameters(kernel, snapshots, y=None, n_rff=500,
                                     verbose=True):
    """
    Optimize (nu, ell, lambda) for a single LOB sub-kernel via L-BFGS-B.

    Matches Group 1's train_garch_rkhs() optimization pattern.

    If y is provided: minimizes Gaussian NLL + RKHS penalty (regression mode).
    If y is None: maximizes effective rank of Gram matrix (unsupervised).

    Parameters
    ----------
    kernel : BaseKernel subclass
        The kernel to optimize. Must have use_rff=True.
    snapshots : list of LOBSnapshot
        Training data.
    y : np.ndarray or None
        Regression targets. If None, uses unsupervised objective.
    n_rff : int
        Number of RFF features.
    verbose : bool
        Print optimization progress.

    Returns
    -------
    kernel : BaseKernel
        The kernel with optimized hyperparameters.
    result : scipy.optimize.OptimizeResult
    """
    assert kernel.use_rff, "Optimization requires use_rff=True"

    # Extract features once for initial length_scale estimate
    X = kernel._extract_feature_matrix(snapshots)
    initial_ell = np.std(X[:, 0]) if X.shape[1] > 0 else 1.0
    initial_ell = max(initial_ell, 1e-4)

    if y is not None:
        # Supervised: Gaussian NLL + RKHS penalty
        def objective(params):
            nu, ell, log_lam = params
            if nu <= 0.1 or ell <= 1e-6:
                return 1e10

            reg_lambda = np.exp(log_lam)
            kernel.nu = nu
            kernel.length_scale = ell
            kernel.reg_lambda = reg_lambda
            kernel.rff = None  # Force re-init

            try:
                kernel.fit(snapshots, y)
                h = kernel.predict(snapshots)
                h = np.maximum(np.abs(h), 1e-8)

                # Gaussian NLL: L = 0.5 * sum(log(h) + y^2/h)
                nll = 0.5 * np.sum(np.log(h) + y ** 2 / h)
                rkhs_penalty = reg_lambda * np.dot(kernel.w, kernel.w)
                return nll + rkhs_penalty
            except Exception:
                return 1e10

        x0 = [1.5, initial_ell, np.log(1e-3)]
        bounds = [(0.2, 10.0), (1e-4, None), (-10, 2)]

    else:
        # Unsupervised: maximize effective rank of Gram matrix
        def objective(params):
            nu, ell = params
            if nu <= 0.1 or ell <= 1e-6:
                return 1e10

            kernel.nu = nu
            kernel.length_scale = ell
            kernel.rff = None  # Force re-init

            try:
                Phi = kernel.feature_map(snapshots)
                K = Phi @ Phi.T
                eigs = np.linalg.eigvalsh(K)
                eigs = np.maximum(eigs, 1e-12)
                p = eigs / eigs.sum()
                effective_rank = np.exp(-np.sum(p * np.log(p)))
                return -effective_rank  # Minimize negative
            except Exception:
                return 1e10

        x0 = [1.5, initial_ell]
        bounds = [(0.2, 10.0), (1e-4, None)]

    if verbose:
        print(f"    Optimizing {kernel.name} hyperparameters...")

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-8})

    # Apply best params
    if y is not None:
        best_nu, best_ell, best_log_lam = result.x
        kernel.reg_lambda = np.exp(best_log_lam)
    else:
        best_nu, best_ell = result.x

    kernel.nu = best_nu
    kernel.length_scale = best_ell
    kernel.rff = None  # Force re-init with final params

    if verbose:
        print(f"      nu={best_nu:.4f}, ell={best_ell:.6f}", end="")
        if y is not None:
            print(f", lambda={kernel.reg_lambda:.6f}", end="")
        print(f"  (obj={result.fun:.4f})")

    return kernel, result


# =============================================================================
# LOB Layer: Combines the 3 sub-kernels via weighted sum
# =============================================================================

class RKHSLayer:
    """
    A single RKHS layer that combines multiple kernels via weighted sum.

    This is the generic layer combiner that all three groups use:

        K_layer(x, y) = sum_i w_i k_i(x, y)

    A positive weighted sum of PSD kernels is PSD (RKHS closure property),
    so the layer output is itself a valid kernel.

    With RFF mode, the layer can also produce a composite feature map:

        phi_LOB(x) = [sqrt(w1) * phi_1(x), sqrt(w2) * phi_2(x), ...]

    This preserves: phi_LOB(x)^T phi_LOB(y) = sum_i w_i * k_i(x, y)

    Parameters
    ----------
    kernels : list of BaseKernel
        The sub-kernels to combine.
    weights : list of float
        Weight for each kernel. Must be non-negative for PSD guarantee.
    name : str
        Layer name (e.g., "LOB", "GARCH", "Sentiment").
    """

    def __init__(self, kernels: List[BaseKernel], weights: List[float],
                 name: str = "Unnamed"):
        assert len(kernels) == len(weights), \
            "Must have one weight per kernel"
        assert all(w >= 0 for w in weights), \
            "Weights must be non-negative to preserve PSD property"

        self.kernels = kernels
        self.weights = np.array(weights, dtype=np.float64)
        self.name = name
        self._w_combined = None
        self._fitted = False

    @property
    def _uses_rff(self):
        """Check if all sub-kernels use RFF mode."""
        return all(getattr(k, 'use_rff', False) for k in self.kernels)

    def __call__(self, x, y) -> float:
        """K_layer(x, y) = sum_i w_i k_i(x, y)"""
        total = 0.0
        for kernel, weight in zip(self.kernels, self.weights):
            total += weight * kernel(x, y)
        return total

    def feature_map(self, snapshots) -> np.ndarray:
        """
        Compute the composite feature map:
            phi_LOB(x) = [sqrt(w1) * phi_1(x), sqrt(w2) * phi_2(x), ...]

        This preserves the weighted sum property:
            phi_LOB(x)^T phi_LOB(y) = sum_i w_i * phi_i(x)^T phi_i(y)
                                     = sum_i w_i * k_i(x, y)
                                     = K_LOB(x, y)

        Parameters
        ----------
        snapshots : list of LOBSnapshot

        Returns
        -------
        Phi_combined : np.ndarray, shape (n, sum_i D_i)
        """
        Phis = []
        for kernel, weight in zip(self.kernels, self.weights):
            Phi_k = kernel.feature_map(snapshots)  # (n, D_k)
            Phis.append(np.sqrt(weight) * Phi_k)
        return np.concatenate(Phis, axis=1)  # (n, sum D_k)

    def gram_matrix(self, snapshots: list) -> np.ndarray:
        """
        Compute the layer Gram matrix.

        RFF mode: Phi_combined @ Phi_combined^T (efficient)
        Legacy mode: weighted sum of sub-kernel Gram matrices
        """
        if self._uses_rff:
            Phi = self.feature_map(snapshots)
            return Phi @ Phi.T
        else:
            n = len(snapshots)
            K = np.zeros((n, n))
            for kernel, weight in zip(self.kernels, self.weights):
                K += weight * kernel.gram_matrix(snapshots)
            return K

    def hilbert_distance(self, x, y) -> float:
        """||phi_LOB(x) - phi_LOB(y)||_H"""
        phi_x = self.feature_map([x])
        phi_y = self.feature_map([y])
        return float(np.linalg.norm(phi_x - phi_y))

    def fit(self, snapshots, y, reg_lambda=1e-3):
        """
        Kernel ridge regression using the composite feature map.

        Parameters
        ----------
        snapshots : list of LOBSnapshot
        y : np.ndarray, shape (n,)
        reg_lambda : float
        """
        Phi = self.feature_map(snapshots)
        D = Phi.shape[1]
        self._w_combined = np.linalg.solve(
            Phi.T @ Phi + reg_lambda * np.eye(D), Phi.T @ y
        )
        self._fitted = True

    def predict(self, snapshots):
        """Predict using fitted composite weights."""
        assert self._fitted, "Must call fit() before predict()"
        Phi = self.feature_map(snapshots)
        return Phi @ self._w_combined

    def is_psd(self, snapshots: list, tol: float = 1e-10):
        """Verify the combined layer Gram matrix is PSD."""
        K = self.gram_matrix(snapshots)
        eigenvalues = np.linalg.eigvalsh(K)
        min_eig = eigenvalues.min()
        return min_eig >= -tol, min_eig

    def summary(self):
        """Print a summary of this layer's kernels and weights."""
        lines = [f"Layer: {self.name}",
                 f"  K_{self.name}(x,y) = " +
                 " + ".join(f"{w:.2f}*{k.name}" for k, w in
                            zip(self.kernels, self.weights)),
                 ""]
        for k, w in zip(self.kernels, self.weights):
            lines.append(f"  [{k.name}] weight={w:.2f}, "
                         f"hyperparams={k.hyperparameters}")
        return "\n".join(lines)

    def __repr__(self):
        kernel_str = " + ".join(
            f"{w:.2f}*{k.name}" for k, w in zip(self.kernels, self.weights)
        )
        return f"RKHSLayer({self.name}: {kernel_str})"


# =============================================================================
# Pre-configured LOB Layer (ready to use)
# =============================================================================

def create_lob_layer(mode: str = "aspect",
                     n_levels: int = 5,
                     # --- RFF toggle ---
                     use_rff: bool = False,
                     nu: float = 1.5,
                     n_rff: int = 500,
                     reg_lambda: float = 1e-3,
                     # --- Aspect mode hyperparameters ---
                     sigma_vol: float = 0.5,
                     sigma_shape="auto",
                     sigma_depth: float = 0.8,
                     w_vol: float = 1.0,
                     w_shape: float = 1.0,
                     w_depth: float = 0.8,
                     # --- Timescale mode hyperparameters (dollar‑bar windows) ---
                     windows: Optional[List[int]] = None,
                     sigmas_ts: Optional[List[float]] = None,
                     weights_ts: Optional[List[float]] = None,
                     labels_ts: Optional[List[str]] = None,
                     ) -> RKHSLayer:
    """
    Create the LOB layer. Toggle between two modes:

    mode="aspect" (default)
        3 kernels measuring different PROPERTIES of the same snapshot:
        VolumeProfile + BookShape + DepthImbalance

    mode="timescale"
        3 kernels measuring ALL properties at different TIME HORIZONS,
        where "time" is measured in **dollar bars**:
        - TimeScale(tick)   : window = 1   (current dollar bar)
        - TimeScale(short)  : window = 10  (10 dollar bars)
        - TimeScale(medium) : window = 50  (50 dollar bars)
        Requires LOBSeries input (time-series of snapshots).

    Parameters
    ----------
    mode : str
        "aspect" or "timescale"
    n_levels : int
        Price levels per side (both modes).
    use_rff : bool
        True = Matern RFF kernels. False = legacy RBF (default for compat).
    nu : float
        Matern smoothness (only when use_rff=True).
    n_rff : int
        Number of RFF features (only when use_rff=True).
    reg_lambda : float
        Ridge regression regularization (only when use_rff=True).

    Returns
    -------
    RKHSLayer
    """
    if mode == "aspect":
        k1 = VolumeProfileKernel(n_levels=n_levels, sigma=sigma_vol,
                                  nu=nu, n_rff=n_rff, reg_lambda=reg_lambda,
                                  use_rff=use_rff, seed=42)
        k2 = BookShapeKernel(n_levels=n_levels, sigma=sigma_shape,
                              nu=nu, n_rff=n_rff, reg_lambda=reg_lambda,
                              use_rff=use_rff, seed=43)
        k3 = DepthImbalanceKernel(n_levels=n_levels, sigma=sigma_depth,
                                   nu=nu, n_rff=n_rff, reg_lambda=reg_lambda,
                                   use_rff=use_rff, seed=44)

        return RKHSLayer(
            kernels=[k1, k2, k3],
            weights=[w_vol, w_shape, w_depth],
            name="LOB",
        )

    elif mode == "timescale":
        if windows is None:
            windows = [1, 10, 50]      # number of dollar bars
        if sigmas_ts is None:
            sigmas_ts = [0.8, 1.0, 1.5]
        if weights_ts is None:
            weights_ts = [1.0, 1.0, 0.8]
        if labels_ts is None:
            labels_ts = ["tick", "short", "medium"]

        assert len(windows) == len(sigmas_ts) == len(weights_ts) == len(labels_ts), \
            "windows, sigmas_ts, weights_ts, and labels_ts must all be same length"

        kernels = [
            TimeScaleKernel(n_levels=n_levels, window=w, sigma=s, label=l,
                            nu=nu, n_rff=n_rff, reg_lambda=reg_lambda,
                            use_rff=use_rff, seed=45 + i)
            for i, (w, s, l) in enumerate(zip(windows, sigmas_ts, labels_ts))
        ]

        return RKHSLayer(
            kernels=kernels,
            weights=weights_ts,
            name="LOB",
        )

    else:
        raise ValueError(f"mode must be 'aspect' or 'timescale', got '{mode}'")


# =============================================================================
# Training Pipeline (matches Group 1's train_garch_rkhs pattern)
# =============================================================================

def train_lob_layer(snapshots, y=None, mode="aspect", n_levels=5,
                    n_rff=500, optimize=True, verbose=True):
    """
    Full training pipeline for the LOB RKHS layer.

    1. Creates the LOB layer with Matern RFF kernels
    2. Optionally optimizes (nu, ell, lambda) per sub-kernel
    3. If y is provided, fits kernel ridge regression
    4. Returns the trained layer with explicit feature maps

    Parameters
    ----------
    snapshots : list of LOBSnapshot or LOBSeries
        Training data. Each snapshot corresponds to a **dollar bar**.
    y : np.ndarray or None
        Regression targets (e.g., next-step mid-price move).
    mode : str
        "aspect" or "timescale"
    n_levels : int
        Price levels per side.
    n_rff : int
        Number of RFF features.
    optimize : bool
        Whether to optimize hyperparameters (slow but better).
    verbose : bool
        Print progress.

    Returns
    -------
    layer : RKHSLayer
        Trained LOB layer with Matern RFF kernels.
    """
    if verbose:
        print("  Building LOB layer with Matern RFF kernels...")

    layer = create_lob_layer(mode=mode, n_levels=n_levels,
                              use_rff=True, n_rff=n_rff)

    if optimize:
        if verbose:
            print("  Optimizing kernel hyperparameters...")
        for kernel in layer.kernels:
            optimize_kernel_hyperparameters(
                kernel, snapshots, y=y, n_rff=n_rff, verbose=verbose
            )

    if y is not None:
        if verbose:
            print("  Fitting kernel ridge regression...")
        layer.fit(snapshots, y)

    if verbose:
        print(f"  Done! {layer}")

    return layer


# =============================================================================
# Full Model Combiner (for combining ALL group layers)
# =============================================================================

class RKHSModel:
    """
    Top-level RKHS model that sums across all group layers.

    K_total(x, y) = alpha_1 K_GARCH(x, y) + alpha_2 K_Sentiment(x, y) + alpha_3 K_LOB(x, y)

    This is where all 3 groups' outputs come together.

    Parameters
    ----------
    layers : list of RKHSLayer
        Each group's layer.
    layer_weights : list of float
        Weight for each layer in the final sum. Non-negative.
    """

    def __init__(self, layers: List[RKHSLayer],
                 layer_weights: Optional[List[float]] = None):
        self.layers = layers
        if layer_weights is None:
            self.layer_weights = np.ones(len(layers))
        else:
            assert len(layer_weights) == len(layers)
            self.layer_weights = np.array(layer_weights, dtype=np.float64)

    def __call__(self, x, y) -> float:
        """K_total(x, y) = sum_j alpha_j K_layer_j(x, y)"""
        total = 0.0
        for layer, alpha in zip(self.layers, self.layer_weights):
            total += alpha * layer(x, y)
        return total

    def gram_matrix(self, snapshots: list) -> np.ndarray:
        """Compute the full model Gram matrix."""
        n = len(snapshots)
        K = np.zeros((n, n))
        for layer, alpha in zip(self.layers, self.layer_weights):
            K += alpha * layer.gram_matrix(snapshots)
        return K

    def summary(self):
        lines = ["=" * 50,
                 "RKHS Model -- Traders@SMU",
                 "K_total = " + " + ".join(
                     f"{a:.2f}*K_{l.name}"
                     for l, a in zip(self.layers, self.layer_weights)),
                 "=" * 50]
        for layer, alpha in zip(self.layers, self.layer_weights):
            lines.append(f"\n  Layer weight alpha = {alpha:.2f}")
            lines.append("  " + layer.summary().replace("\n", "\n  "))
        return "\n".join(lines)

    def __repr__(self):
        return "RKHSModel(" + " + ".join(
            f"{a:.2f}*{l.name}" for l, a in
            zip(self.layers, self.layer_weights)) + ")"


# =============================================================================
# Synthetic Data Generator (for testing, now with dollar‑bar indices)
# =============================================================================

def generate_synthetic_lob(n_snapshots: int = 100, n_levels: int = 5,
                           base_price: float = 100.0, tick_size: float = 0.05,
                           seed: int = 42) -> List[LOBSnapshot]:
    """
    Generate synthetic LOB snapshots for testing.

    Creates realistic-ish order book data with:
    - Prices spaced by tick_size from a random-walking mid-price
    - Volumes drawn from a log-normal distribution (heavy-tailed)

    Each snapshot is assigned a sequential integer timestamp representing the
    dollar bar index.
    """
    rng = np.random.default_rng(seed)
    snapshots = []
    mid = base_price

    for t in range(n_snapshots):
        mid += rng.normal(0, tick_size * 0.5)
        half_spread = tick_size * (0.5 + rng.exponential(0.3))

        bid_prices = np.array([mid - half_spread - i * tick_size
                               for i in range(n_levels)])
        ask_prices = np.array([mid + half_spread + i * tick_size
                               for i in range(n_levels)])
        bid_volumes = np.array([
            rng.lognormal(mean=np.log(200) - 0.2 * i, sigma=0.5)
            for i in range(n_levels)])
        ask_volumes = np.array([
            rng.lognormal(mean=np.log(200) - 0.2 * i, sigma=0.5)
            for i in range(n_levels)])

        snapshots.append(LOBSnapshot(
            bid_prices=bid_prices, bid_volumes=bid_volumes,
            ask_prices=ask_prices, ask_volumes=ask_volumes,
            timestamp=t,  # dollar bar index
        ))

    return snapshots


def generate_synthetic_lob_series(n_series: int = 20,
                                  series_length: int = 100,
                                  n_levels: int = 5,
                                  base_price: float = 100.0,
                                  tick_size: float = 0.05,
                                  seed: int = 42) -> List[LOBSeries]:
    """
    Generate synthetic LOBSeries objects for testing timescale mode.

    Creates `n_series` independent LOBSeries, each containing
    `series_length` consecutive dollar‑bar snapshots.
    """
    rng = np.random.default_rng(seed)
    all_series = []

    for s in range(n_series):
        mid = base_price + rng.normal(0, 1.0)
        snapshots = []

        for t in range(series_length):
            mid += rng.normal(0, tick_size * 0.5)
            half_spread = tick_size * (0.5 + rng.exponential(0.3))

            bid_prices = np.array([mid - half_spread - i * tick_size
                                   for i in range(n_levels)])
            ask_prices = np.array([mid + half_spread + i * tick_size
                                   for i in range(n_levels)])
            bid_volumes = np.array([
                rng.lognormal(mean=np.log(200) - 0.2 * i, sigma=0.5)
                for i in range(n_levels)])
            ask_volumes = np.array([
                rng.lognormal(mean=np.log(200) - 0.2 * i, sigma=0.5)
                for i in range(n_levels)])

            snapshots.append(LOBSnapshot(
                bid_prices=bid_prices, bid_volumes=bid_volumes,
                ask_prices=ask_prices, ask_volumes=ask_volumes,
                timestamp=t,  # dollar bar index within this series
            ))

        all_series.append(LOBSeries(snapshots))

    return all_series
