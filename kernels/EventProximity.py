"""
Event Proximity & Calendar Layer — RKHS Kernel Module
======================================================
Traders@SMU — Quantitative Strategies Group
Layer 2b: Event Proximity + Calendar (Will, Demetrios, Arjan)

Motivation
----------
Volatility is not time-homogeneous.  Known, scheduled events —
earnings releases, FOMC decisions, CPI prints, options expiry —
create predictable spikes in implied and realised volatility.
Two dates that are both "3 days before an earnings release" are
more similar (in a volatility sense) than a random Tuesday and Thursday,
even if their squared returns look the same.

This layer encodes WHEN a date sits relative to the event calendar
as a continuous feature vector, then applies a Matérn kernel
(via Random Fourier Features) to produce a valid PSD kernel
K_Event(t_i, t_j) that can be summed with K_GARCH and K_Sentiment.

Architecture
------------
The full RKHS model sums kernel outputs across layers:

    K_total(x, y) = α₁ K_GARCH(x, y)          <- Group 1 (Anders, Zabe, Ameen)
                  + α₂ K_Sentiment(x, y)        <- Group 2 sentiment sub-layer
                  + α₂ K_Event(x, y)            <- Group 2 event/calendar sub-layer  *** THIS FILE ***
                  + α₃ K_LOB(x, y)              <- Group 3 (Becky, Connor)

Within this layer the kernel is a weighted sum of three sub-kernels:

    K_Event(x, y) = w₁ k_proximity(x, y)        -- days-to/from each event type
                  + w₂ k_calendar(x, y)          -- periodic calendar features (DOW, DOM, expiry cycle)
                  + w₃ k_regime(x, y)            -- macro-event regime (pre/during/post)

Why continuous features instead of dummies?
-------------------------------------------
Binary "is earnings day" dummies are discontinuous — they jump from 0 to 1
at a single date, violating the smoothness required for a valid RKHS kernel.

Instead we encode proximity as:

    f_event(t) = exp(-|days_to_event| / tau)

This decays smoothly away from each event, giving a continuous feature
that the Matérn RFF kernel can act on.  The result is a kernel that
assigns high similarity to dates with similar proximity profiles —
mathematically valid and interpretable.

References
----------
    - Matérn kernel: Rasmussen & Williams (2006), §4.2
    - Random Fourier Features: Rahimi & Recht (2007), NeurIPS
    - RKHS closure under weighted sums: Berlinet & Thomas-Agnan (2004)
    - Volatility seasonality: Andersen & Bollerslev (1997), JoF
"""

import numpy as np
from datetime import date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from scipy.optimize import minimize


# =============================================================================
# Event Calendar Representation
# =============================================================================

# Supported event types — extend this list as needed
EVENT_TYPES = [
    "earnings",       # company earnings release
    "fomc",           # Federal Open Market Committee decision
    "cpi",            # Consumer Price Index print
    "ppi",            # Producer Price Index
    "nfp",            # Non-Farm Payrolls
    "options_expiry", # monthly/quarterly options expiry (OpEx)
    "fed_speech",     # major Fed chair speech / Jackson Hole etc.
    "gdp",            # GDP release
]


class CalendarEvent:
    """
    A single dated market event.

    Parameters
    ----------
    event_date : date
        The calendar date of the event.
    event_type : str
        One of EVENT_TYPES (or any custom string).
    label : str, optional
        Human-readable description (e.g. "Apple Q3 2024 Earnings").
    magnitude : float
        Expected impact magnitude [0, 1].  1 = highest impact event
        (e.g. FOMC rate decision), 0.2 = minor speech.  Used to weight
        the proximity decay.  Defaults to 1.0.
    """

    def __init__(self, event_date: date, event_type: str,
                 label: str = "", magnitude: float = 1.0):
        self.event_date = event_date
        self.event_type = event_type.lower()
        self.label = label or f"{event_type} @ {event_date}"
        self.magnitude = float(magnitude)

    def days_from(self, t: date) -> int:
        """Signed days from date t to this event (positive = event is in the future)."""
        return (self.event_date - t).days

    def __repr__(self):
        return f"CalendarEvent({self.event_type!r}, {self.event_date}, mag={self.magnitude:.2f})"


class EventCalendar:
    """
    A collection of CalendarEvents used to build proximity features.

    Parameters
    ----------
    events : list of CalendarEvent
    """

    def __init__(self, events: Optional[List[CalendarEvent]] = None):
        self.events = events or []

    def add(self, event: CalendarEvent):
        self.events.append(event)

    def events_of_type(self, event_type: str) -> List[CalendarEvent]:
        return [e for e in self.events if e.event_type == event_type.lower()]

    def nearest(self, t: date, event_type: Optional[str] = None
                ) -> Optional[Tuple[CalendarEvent, int]]:
        """
        Find the nearest event (past or future) to date t.

        Returns (event, signed_days) or None if no events exist.
        """
        pool = self.events_of_type(event_type) if event_type else self.events
        if not pool:
            return None
        nearest_event = min(pool, key=lambda e: abs(e.days_from(t)))
        return nearest_event, nearest_event.days_from(t)

    def __len__(self):
        return len(self.events)

    def __repr__(self):
        return f"EventCalendar({len(self.events)} events)"


# =============================================================================
# Date Sample
# =============================================================================

class DateSample:
    """
    A single trading date with its pre-computed feature vector.

    Parameters
    ----------
    t : date
        The trading date.
    features : np.ndarray or None
        Feature vector (set by a kernel's extract_features() call).
    """

    def __init__(self, t: date, features: Optional[np.ndarray] = None):
        self.t = t
        self.features = features

    def __repr__(self):
        feat_str = f"feat={self.features.shape}" if self.features is not None else "feat=None"
        return f"DateSample({self.t}, {feat_str})"


# =============================================================================
# Matérn Random Fourier Features (same as sentiment_kernel.py)
# =============================================================================

class MaternRFF:
    """
    Matérn kernel approximated via Random Fourier Features.

    K_Matérn(x, x') ≈ phi(x)^T phi(x')

    Parameters
    ----------
    nu : float            Matérn smoothness (0.5, 1.5, or 2.5 are standard).
    length_scale : float  Kernel length scale.
    n_features : int      Number of random Fourier features D.
    input_dim : int       Dimensionality of input vectors.
    random_state : int    Seed for reproducibility.
    """

    def __init__(self, nu: float = 1.5, length_scale: float = 1.0,
                 n_features: int = 256, input_dim: int = 8,
                 random_state: Optional[int] = 42):
        self.nu = nu
        self.length_scale = length_scale
        self.n_features = n_features
        self.input_dim = input_dim
        self._rng = np.random.RandomState(random_state)
        self._sample_frequencies()

    def _sample_frequencies(self):
        rng = self._rng
        d, D = self.input_dim, self.n_features
        Z = rng.randn(D, d)
        # chi^2(2*nu) via Gamma(nu, 2)
        s_sq = rng.gamma(shape=self.nu, scale=2.0, size=D)
        s = np.sqrt(s_sq)
        self._omega = (Z / s[:, None]) / self.length_scale   # (D, d)
        self._bias = rng.uniform(0, 2 * np.pi, size=D)       # (D,)

    def update_params(self, nu: float, length_scale: float):
        self.nu = nu
        self.length_scale = length_scale
        self._sample_frequencies()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """phi(X) = sqrt(2/D) * cos(X @ omega^T + b),  shape (n, D)"""
        proj = X @ self._omega.T + self._bias   # (n, D)
        return np.sqrt(2.0 / self.n_features) * np.cos(proj)


# =============================================================================
# Abstract Base Kernel
# =============================================================================

class BaseKernel(ABC):
    """Abstract base for all event/calendar sub-kernels."""

    use_rff: bool = True
    _rff: Optional[MaternRFF] = None

    def _init_rff(self, nu: float = 1.5, length_scale: float = 1.0,
                  n_rff: int = 256, random_state: int = 42):
        self._rff = MaternRFF(
            nu=nu, length_scale=length_scale,
            n_features=n_rff, input_dim=self._feature_dim,
            random_state=random_state,
        )

    def _extract_feature_matrix(self, samples: List[DateSample]) -> np.ndarray:
        return np.vstack([self.extract_features(s) for s in samples])

    @abstractmethod
    def extract_features(self, sample: DateSample) -> np.ndarray:
        """Convert a DateSample into a raw feature vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def _feature_dim(self) -> int:
        pass

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        if self._rff is not None:
            return {"nu": self._rff.nu, "length_scale": self._rff.length_scale}
        return {}

    def feature_map(self, samples: List[DateSample]) -> np.ndarray:
        X = self._extract_feature_matrix(samples)
        if self.use_rff and self._rff is not None:
            return self._rff.transform(X)
        return X

    def gram_matrix(self, samples: List[DateSample]) -> np.ndarray:
        Phi = self.feature_map(samples)
        return Phi @ Phi.T

    # FIX 1: Added __call__ so EventLayer.__call__ can invoke k(x, y) without
    # crashing.  K(x, y) = <phi(x), phi(y)> is the standard RKHS inner product.
    def __call__(self, x: DateSample, y: DateSample) -> float:
        phi_x = self.feature_map([x])   # (1, D)
        phi_y = self.feature_map([y])   # (1, D)
        return float(phi_x @ phi_y.T)   # scalar kernel evaluation K(x, y)

    def hilbert_distance(self, x: DateSample, y: DateSample) -> float:
        phi_x = self.feature_map([x])
        phi_y = self.feature_map([y])
        return float(np.linalg.norm(phi_x - phi_y))

    def fit(self, samples: List[DateSample], y: np.ndarray,
            reg_lambda: float = 1e-3):
        Phi = self.feature_map(samples)
        D = Phi.shape[1]
        self._alpha = np.linalg.solve(
            Phi.T @ Phi + reg_lambda * np.eye(D), Phi.T @ y
        )

    def predict(self, samples: List[DateSample]) -> np.ndarray:
        assert hasattr(self, "_alpha"), "Call fit() before predict()"
        return self.feature_map(samples) @ self._alpha

    def is_psd(self, samples: List[DateSample], tol: float = 1e-10):
        K = self.gram_matrix(samples)
        min_eig = np.linalg.eigvalsh(K).min()
        return min_eig >= -tol, min_eig

    def __repr__(self):
        hp = self.hyperparameters
        return f"{self.name}(nu={hp.get('nu','?'):.2f}, ell={hp.get('length_scale','?'):.3f})"


# =============================================================================
# Sub-Kernel 1: Proximity Kernel
# =============================================================================

class ProximityKernel(BaseKernel):
    """
    Captures smooth proximity to each scheduled event type.

    For each event type e and date t we compute:

        f_e(t) = magnitude * exp(-|days_to_nearest_e| / tau_e)

    This is a smooth, bounded, continuous function of time — so it
    lives in a valid RKHS.  The decay rate tau controls how far the
    "influence window" of an event extends.

    Feature vector: one proximity score per event type in EVENT_TYPES,
    giving a vector in [0, 1]^|EVENT_TYPES|.

    Parameters
    ----------
    calendar : EventCalendar
    event_types : list of str
        Subset of EVENT_TYPES to encode.  Defaults to all.
    tau : float
        Decay half-life in calendar days.  tau=5 means proximity
        halves every ~3.5 days (exp(-1/5) ≈ 0.82 per day).
    nu, length_scale, n_rff : Matérn / RFF parameters.
    """

    def __init__(self, calendar: EventCalendar,
                 event_types: Optional[List[str]] = None,
                 tau: float = 5.0,
                 nu: float = 1.5, length_scale: float = 0.5,
                 n_rff: int = 256, random_state: int = 42):
        self.calendar = calendar
        self.event_types = event_types or EVENT_TYPES
        self.tau = tau
        self.use_rff = True
        self._init_rff(nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state)

    @property
    def _feature_dim(self) -> int:
        return len(self.event_types)

    @property
    def name(self) -> str:
        return "ProximityKernel"

    def extract_features(self, sample: DateSample) -> np.ndarray:
        """
        Build the proximity feature vector for date sample.t.

        For each event type, find the nearest event (past or future)
        and compute exp(-|days| / tau) weighted by event magnitude.
        All distances are in calendar days for time-homogeneous treatment.
        """
        t = sample.t
        features = np.zeros(len(self.event_types))
        for i, etype in enumerate(self.event_types):
            pool = self.calendar.events_of_type(etype)
            if not pool:
                features[i] = 0.0
                continue
            # FIX 7: Compute nearest event once; reuse signed distance to avoid
            # calling days_from(t) twice on the same object.
            nearest = min(pool, key=lambda e: abs(e.days_from(t)))
            signed = nearest.days_from(t)
            days = abs(signed)
            features[i] = nearest.magnitude * np.exp(-days / self.tau)
        return features

    def __repr__(self):
        return (f"ProximityKernel(types={self.event_types}, tau={self.tau}, "
                f"nu={self._rff.nu:.2f}, ell={self._rff.length_scale:.3f})")


# =============================================================================
# Sub-Kernel 2: Calendar Kernel
# =============================================================================

class CalendarKernel(BaseKernel):
    """
    Captures periodic calendar seasonality.

    Features encoded (all smooth and continuous in calendar time):
        - Day of week (sine/cosine encoding, period=5 trading days)
        - Day of month (sine/cosine encoding, period=30.5 calendar days)
        - Month of year (sine/cosine encoding, period=12)
        - Week of month [0,1] (position within month, calendar-day based)
        - Quarter position (where in Q1/Q2/Q3/Q4 are we, 0→1, 91-day denominator)
        - OpEx proximity: distance to the 3rd Friday of each month
          (monthly options expiry — always high-vol period)
        - Monday proximity: smooth peak on Monday decaying across the week
          (gap risk from weekend — replaces discontinuous Monday dummy)
        - Friday proximity: smooth peak on Friday decaying back toward Monday
          (position-squaring effect — replaces discontinuous Friday dummy)

    All features are bounded, continuous, and smooth in calendar time.
    No binary indicator functions are used (they are discontinuous and
    violate the smoothness requirement for a well-behaved RKHS).
    Feature vector length: 11 dimensions.

    Parameters
    ----------
    nu, length_scale, n_rff : Matérn / RFF parameters.
    """

    def __init__(self, nu: float = 1.5, length_scale: float = 0.3,
                 n_rff: int = 256, random_state: int = 43):
        self.use_rff = True
        self._init_rff(nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state)

    @property
    def _feature_dim(self) -> int:
        return 11

    @property
    def name(self) -> str:
        return "CalendarKernel"

    @staticmethod
    def _third_friday(year: int, month: int) -> date:
        """Return the 3rd Friday of a given month (standard monthly OpEx date)."""
        d = date(year, month, 1)
        # weekday(): Monday=0, Friday=4
        first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
        return first_friday + timedelta(weeks=2)

    def extract_features(self, sample: DateSample) -> np.ndarray:
        t = sample.t

        # --- Day of week (0=Mon ... 4=Fri), encoded as sin/cos ---
        dow = t.weekday()  # 0-4 for trading days
        dow_sin = np.sin(2 * np.pi * dow / 5)
        dow_cos = np.cos(2 * np.pi * dow / 5)

        # --- Day of month (1-31), encoded with calendar-day period ---
        # FIX 5: Use 30.5 (mean calendar days/month) as the period so that
        # days 22-31 do not alias back onto days 1-10 as they did with /21.
        dom = t.day
        dom_sin = np.sin(2 * np.pi * dom / 30.5)
        dom_cos = np.cos(2 * np.pi * dom / 30.5)

        # --- Month of year (1-12) ---
        moy = t.month
        moy_sin = np.sin(2 * np.pi * moy / 12)
        moy_cos = np.cos(2 * np.pi * moy / 12)

        # --- Week of month [0,1]: calendar-day position within month ---
        # FIX 5 (continued): normalise by 29 calendar days so the feature
        # spans [0, 1] monotonically across any month without clipping early.
        week_of_month = np.clip((dom - 1) / 29.0, 0.0, 1.0)

        # --- Quarter position [0,1]: where in Q are we? ---
        # FIX 6: A calendar quarter is ~91 days.  Using 65 caused the last
        # ~26 days of every quarter to clip to 1.0, destroying information.
        quarter_start_month = ((moy - 1) // 3) * 3 + 1   # 1, 4, 7, or 10
        days_into_quarter = (t - date(t.year, quarter_start_month, 1)).days
        quarter_pos = np.clip(days_into_quarter / 91.0, 0.0, 1.0)

        # --- OpEx proximity: days to 3rd Friday of current month ---
        opex = self._third_friday(t.year, t.month)
        days_to_opex = abs((opex - t).days)
        opex_proximity = np.exp(-days_to_opex / 3.0)   # sharp 3-day decay

        # FIX 4: Replace discontinuous Monday/Friday binary flags with smooth
        # exponential proximity scores.  Both are continuous functions of dow
        # and consistent with the RKHS smoothness requirement stated in the
        # module docstring.
        #
        # monday_prox peaks at dow=0 (Monday) and decays toward Friday.
        # friday_prox peaks at dow=4 (Friday) and decays back toward Monday.
        monday_prox = np.exp(-dow / 2.0)          # 1.0 on Mon, ~0.14 on Fri
        friday_prox = np.exp(-(4 - dow) / 2.0)   # ~0.14 on Mon, 1.0 on Fri

        return np.array([
            dow_sin, dow_cos,
            dom_sin, dom_cos,
            moy_sin, moy_cos,
            week_of_month,
            quarter_pos,
            opex_proximity,
            monday_prox,   # smooth Monday gap-risk signal
            friday_prox,   # smooth Friday position-squaring signal
        ])

    def __repr__(self):
        return (f"CalendarKernel(nu={self._rff.nu:.2f}, "
                f"ell={self._rff.length_scale:.3f})")


# =============================================================================
# Sub-Kernel 3: Regime Kernel
# =============================================================================

class RegimeKernel(BaseKernel):
    """
    Captures the macro-event regime phase: pre-event, during, or post-event.

    For each event type we encode a signed proximity:

        g_e(t) = tanh(days_to_event / tau_e)

    where days_to_event = (event_date - t).days, so:

        g_e(t) ≈ +1  →  well before the event  (anticipation / pre-event phase)
        g_e(t) ≈  0  →  at the event            (realisation phase)
        g_e(t) ≈ -1  →  well after the event    (digestion / post-event phase)

    The tanh encoding preserves directionality (before vs after),
    is continuous, and saturates smoothly — all compatible with RKHS.

    Feature vector: one tanh score per event type.

    Parameters
    ----------
    calendar : EventCalendar
    event_types : list of str
    tau : float
        Scaling factor in calendar days (controls how quickly the regime
        saturates away from the event date).
    """

    def __init__(self, calendar: EventCalendar,
                 event_types: Optional[List[str]] = None,
                 tau: float = 10.0,
                 nu: float = 1.5, length_scale: float = 0.5,
                 n_rff: int = 256, random_state: int = 44):
        self.calendar = calendar
        self.event_types = event_types or EVENT_TYPES
        self.tau = tau
        self.use_rff = True
        self._init_rff(nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state)

    @property
    def _feature_dim(self) -> int:
        return len(self.event_types)

    @property
    def name(self) -> str:
        return "RegimeKernel"

    def extract_features(self, sample: DateSample) -> np.ndarray:
        """
        For each event type, find the nearest event and encode the
        signed calendar-day distance as tanh(days_to_event / tau).

        Sign convention (FIX 3 — now consistent with docstring):
            days_to_event > 0  →  event is in the future  →  tanh > 0  →  pre-event
            days_to_event < 0  →  event is in the past    →  tanh < 0  →  post-event
        """
        t = sample.t
        features = np.zeros(len(self.event_types))
        for i, etype in enumerate(self.event_types):
            pool = self.calendar.events_of_type(etype)
            if not pool:
                features[i] = 0.0
                continue
            nearest = min(pool, key=lambda e: abs(e.days_from(t)))
            # days_from returns (event_date - t).days:
            #   positive  →  event ahead  →  pre-event regime  →  tanh → +1
            #   negative  →  event past   →  post-event regime →  tanh → -1
            signed_days = nearest.days_from(t)
            features[i] = np.tanh(signed_days / self.tau)
        return features

    def __repr__(self):
        return (f"RegimeKernel(types={self.event_types}, tau={self.tau}, "
                f"nu={self._rff.nu:.2f}, ell={self._rff.length_scale:.3f})")


# =============================================================================
# Event Layer (combined kernel)
# =============================================================================

class EventLayer:
    """
    The full Group 2 event/calendar layer: weighted sum of sub-kernels.

    K_Event(x, y) = w₁ k_proximity(x, y)
                  + w₂ k_calendar(x, y)
                  + w₃ k_regime(x, y)

    A non-negative weighted sum of PSD kernels is PSD (RKHS closure property).

    Parameters
    ----------
    kernels : list of BaseKernel
    weights : list of float  (must be non-negative to preserve PSD property)
    name : str

    Notes on weight optimisation (FIX 8)
    -------------------------------------
    optimize_hyperparams() currently tunes (nu, length_scale, lambda) per
    sub-kernel but leaves the layer-level weights fixed.  If you later add
    weight optimisation, the non-negativity constraint (w_i >= 0) is required
    to preserve the PSD property of K_Event.  Use a constrained solver
    (e.g. scipy.optimize with bounds=[(0, None), ...]) and do NOT rely solely
    on the __init__ assertion.
    """

    def __init__(self, kernels: List[BaseKernel], weights: List[float],
                 name: str = "Event"):
        assert len(kernels) == len(weights)
        assert all(w >= 0 for w in weights), \
            "Weights must be non-negative to preserve PSD property"
        self.kernels = kernels
        self.weights = np.array(weights, dtype=np.float64)
        self.name = name
        self._w_combined: Optional[np.ndarray] = None
        self._fitted = False

    # FIX 1: __call__ now works because BaseKernel defines __call__.
    def __call__(self, x: DateSample, y: DateSample) -> float:
        return sum(w * k(x, y) for k, w in zip(self.kernels, self.weights))

    def feature_map(self, samples: List[DateSample]) -> np.ndarray:
        """
        phi_Event(x) = [sqrt(w1)*phi_1(x), sqrt(w2)*phi_2(x), ...]
        Shape: (n, sum_i D_i)

        Scaling each block by sqrt(w_i) ensures that the inner product of
        the concatenated map recovers the weighted sum of sub-kernels exactly:
            <phi_Event(x), phi_Event(y)> = sum_i w_i <phi_i(x), phi_i(y)>
                                         = K_Event(x, y)   ✓
        """
        parts = [np.sqrt(w) * k.feature_map(samples)
                 for k, w in zip(self.kernels, self.weights)]
        return np.concatenate(parts, axis=1)

    def gram_matrix(self, samples: List[DateSample]) -> np.ndarray:
        Phi = self.feature_map(samples)
        return Phi @ Phi.T

    def hilbert_distance(self, x: DateSample, y: DateSample) -> float:
        phi_x = self.feature_map([x])
        phi_y = self.feature_map([y])
        return float(np.linalg.norm(phi_x - phi_y))

    def fit(self, samples: List[DateSample], y: np.ndarray,
            reg_lambda: float = 1e-3):
        """Kernel ridge regression: (Phi^T Phi + lambda I) w = Phi^T y"""
        Phi = self.feature_map(samples)
        D = Phi.shape[1]
        self._w_combined = np.linalg.solve(
            Phi.T @ Phi + reg_lambda * np.eye(D), Phi.T @ y
        )
        self._fitted = True

    def predict(self, samples: List[DateSample]) -> np.ndarray:
        assert self._fitted, "Call fit() before predict()"
        return self.feature_map(samples) @ self._w_combined

    def is_psd(self, samples: List[DateSample], tol: float = 1e-10):
        K = self.gram_matrix(samples)
        min_eig = np.linalg.eigvalsh(K).min()
        return min_eig >= -tol, min_eig

    def optimize_hyperparams(self, samples: List[DateSample],
                             y: np.ndarray, verbose: bool = True) -> dict:
        """
        Optimise (nu, length_scale, lambda) per sub-kernel via Gaussian NLL.

        The log marginal likelihood is:
            log p(y) = -½ y^T K^{-1} y  -  ½ log|K|  -  n/2 log(2π)

        Using the Cholesky factor L of K (so K = L L^T):
            log|K| = log|L^T L| = 2 * sum_i log L_{ii}

        Therefore NLL = log|L| + ½ y^T K^{-1} y + n/2 log(2π)
                      = sum_i log L_{ii} + ½ y^T K^{-1} y + n/2 log(2π)

        FIX 2: The log-determinant term is written explicitly to avoid
        ambiguity.  log|L| = sum log L_{ii} = ½ log|K|, which is the
        correct coefficient in the NLL expression above.
        """
        results = {}
        for kernel in self.kernels:
            if verbose:
                print(f"  Optimising {kernel.name}...")

            def objective(params, k=kernel):
                nu, ell, lam = params
                if nu <= 0 or ell <= 0 or lam <= 0:
                    return 1e10
                k._rff.update_params(nu=nu, length_scale=ell)
                K = k.gram_matrix(samples)
                n = len(samples)
                K_reg = K + lam * np.eye(n)
                try:
                    L = np.linalg.cholesky(K_reg)
                    # log|K_reg| = 2 * sum(log diag(L))  →  ½ log|K_reg| = sum(log diag(L))
                    log_det_term = np.sum(np.log(np.diag(L)))   # = ½ log|K_reg|
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    nll = (log_det_term
                           + 0.5 * y @ alpha
                           + 0.5 * n * np.log(2 * np.pi))
                    return float(nll)
                except np.linalg.LinAlgError:
                    return 1e10

            x0 = [kernel._rff.nu, kernel._rff.length_scale, 1e-3]
            bounds = [(0.1, 10.0), (0.001, None), (1e-6, None)]
            res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
            best_nu, best_ell, best_lam = res.x
            kernel._rff.update_params(nu=best_nu, length_scale=best_ell)
            results[kernel.name] = (best_nu, best_ell, best_lam)
            if verbose:
                print(f"    {kernel.name}: nu={best_nu:.3f}, "
                      f"ell={best_ell:.4f}, lambda={best_lam:.2e}")
        return results

    def summary(self) -> str:
        lines = [
            f"Layer: {self.name}",
            "  K_Event(x,y) = "
            + " + ".join(f"{w:.2f}*{k.name}"
                         for k, w in zip(self.kernels, self.weights)),
            "",
        ]
        for k, w in zip(self.kernels, self.weights):
            lines.append(f"  [{k.name}]  weight={w:.2f},  "
                         f"hyperparams={k.hyperparameters}")
        return "\n".join(lines)

    def __repr__(self):
        s = " + ".join(f"{w:.2f}*{k.name}"
                       for k, w in zip(self.kernels, self.weights))
        return f"EventLayer({self.name}: {s})"


# =============================================================================
# Factory
# =============================================================================

def create_event_layer(calendar: EventCalendar,
                       event_types: Optional[List[str]] = None,
                       tau_proximity: float = 5.0,
                       tau_regime: float = 10.0,
                       nu: float = 1.5,
                       length_scale: float = 0.5,
                       n_rff: int = 256,
                       weights: Optional[List[float]] = None,
                       random_state: int = 42) -> EventLayer:
    """
    Create a ready-to-use EventLayer with three sub-kernels.

    Parameters
    ----------
    calendar : EventCalendar
        Collection of scheduled events.
    event_types : list of str or None
        Event types to encode in Proximity and Regime kernels.
        Defaults to all EVENT_TYPES.
    tau_proximity : float
        Exponential decay half-life in calendar days for ProximityKernel.
    tau_regime : float
        Tanh scaling in calendar days for RegimeKernel.
    nu : float
        Matérn smoothness for all sub-kernels.
    length_scale : float
        Kernel length scale for all sub-kernels.
    n_rff : int
        Number of Random Fourier Features per sub-kernel.
    weights : list of 3 floats or None
        Sub-kernel weights [w_proximity, w_calendar, w_regime].
        Must be non-negative.  Defaults to equal weights [1/3, 1/3, 1/3].
    random_state : int

    Returns
    -------
    EventLayer
    """
    if weights is None:
        weights = [1 / 3, 1 / 3, 1 / 3]

    kernels = [
        ProximityKernel(
            calendar=calendar,
            event_types=event_types,
            tau=tau_proximity,
            nu=nu, length_scale=length_scale,
            n_rff=n_rff, random_state=random_state,
        ),
        CalendarKernel(
            nu=nu, length_scale=length_scale,
            n_rff=n_rff, random_state=random_state + 1,
        ),
        RegimeKernel(
            calendar=calendar,
            event_types=event_types,
            tau=tau_regime,
            nu=nu, length_scale=length_scale,
            n_rff=n_rff, random_state=random_state + 2,
        ),
    ]
    return EventLayer(kernels=kernels, weights=weights, name="Event")


def make_sample_calendar() -> EventCalendar:
    """
    Build a small example EventCalendar for testing / demo.
    Replace with real event dates from an earnings calendar API,
    the Fed website, BLS release schedule, etc.
    """
    cal = EventCalendar()

    # FOMC meetings (8 per year, roughly every 6 weeks)
    fomc_dates = [
        date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
        date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
        date(2024, 11, 7), date(2024, 12, 18),
        date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    ]
    for d in fomc_dates:
        cal.add(CalendarEvent(d, "fomc", magnitude=1.0))

    # CPI releases (monthly, ~2nd week)
    cpi_dates = [
        date(2024, 1, 11), date(2024, 2, 13), date(2024, 3, 12),
        date(2024, 4, 10), date(2024, 5, 15), date(2024, 6, 12),
        date(2024, 7, 11), date(2024, 8, 14), date(2024, 9, 11),
        date(2024, 10, 10), date(2024, 11, 13), date(2024, 12, 11),
        date(2025, 1, 15), date(2025, 2, 12),
    ]
    for d in cpi_dates:
        cal.add(CalendarEvent(d, "cpi", magnitude=0.8))

    # NFP releases (first Friday of each month)
    nfp_dates = [
        date(2024, 1, 5), date(2024, 2, 2), date(2024, 3, 8),
        date(2024, 4, 5), date(2024, 5, 3), date(2024, 6, 7),
        date(2024, 7, 5), date(2024, 8, 2), date(2024, 9, 6),
        date(2024, 10, 4), date(2024, 11, 1), date(2024, 12, 6),
        date(2025, 1, 10), date(2025, 2, 7),
    ]
    for d in nfp_dates:
        cal.add(CalendarEvent(d, "nfp", magnitude=0.7))

    # Sample earnings (major names — extend with real data)
    earnings = [
        (date(2024, 1, 25), "TSLA Q4 2023"),
        (date(2024, 2, 1),  "META Q4 2023"),
        (date(2024, 2, 1),  "AMZN Q4 2023"),
        (date(2024, 2, 2),  "AAPL Q1 FY2024"),
        (date(2024, 4, 25), "MSFT Q3 FY2024"),
        (date(2024, 7, 23), "TSLA Q2 2024"),
        (date(2024, 10, 29), "META Q3 2024"),
        (date(2024, 10, 30), "MSFT Q1 FY2025"),
        (date(2024, 10, 31), "AAPL Q4 FY2024"),
    ]
    for d, label in earnings:
        cal.add(CalendarEvent(d, "earnings", label=label, magnitude=0.9))

    return cal


# =============================================================================
# Demo / smoke test
# =============================================================================

def _demo():
    """
    Smoke test: builds the event layer on a sequence of trading dates
    and verifies PSD property, fit/predict, and Hilbert distances.
    No external dependencies required beyond numpy and scipy.
    """
    print("=" * 60)
    print("EventLayer — smoke test")
    print("=" * 60)

    # Build calendar
    calendar = make_sample_calendar()
    print(f"\nCalendar: {calendar}")

    # Build a sequence of trading dates (calendar days, weekdays only)
    start = date(2024, 1, 2)
    dates = [start + timedelta(days=i)
             for i in range(60)
             if (start + timedelta(days=i)).weekday() < 5]
    samples = [DateSample(d) for d in dates]
    n = len(samples)
    print(f"Dates   : {n} trading days ({dates[0]} to {dates[-1]})")

    # Build layer
    layer = create_event_layer(calendar, n_rff=128)
    print(f"\n{layer.summary()}")

    # Gram matrix
    K = layer.gram_matrix(samples)
    print(f"\nGram matrix shape : {K.shape}")
    print(f"Gram matrix range : [{K.min():.4f}, {K.max():.4f}]")

    # PSD check
    is_psd, min_eig = layer.is_psd(samples)
    print(f"PSD check         : {is_psd}  (min eigenvalue = {min_eig:.2e})")

    # Fit/predict with dummy returns
    rng = np.random.RandomState(1)
    y = rng.randn(n)
    layer.fit(samples, y)
    preds = layer.predict(samples)
    print(f"\nFit/predict       : predictions shape = {preds.shape}")
    print(f"  First 5 preds   : {preds[:5].round(4)}")

    # Hilbert distances: FOMC day vs nearby days
    fomc_day = DateSample(date(2024, 3, 20))    # FOMC meeting
    day_before = DateSample(date(2024, 3, 19))
    week_before = DateSample(date(2024, 3, 13))
    random_day = DateSample(date(2024, 5, 7))

    print(f"\nHilbert distances from FOMC day ({fomc_day.t}):")
    for label, s in [("1 day before", day_before),
                     ("1 week before", week_before),
                     ("random day (May 7)", random_day)]:
        d = layer.hilbert_distance(fomc_day, s)
        print(f"  dist(FOMC, {label:20s}) = {d:.4f}")

    # Verify __call__ works (was broken before FIX 1)
    k_val = layer(fomc_day, day_before)
    print(f"\nK_Event(FOMC day, 1 day before) = {k_val:.6f}  [__call__ check ✓]")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    _demo()
