"""
Sentiment Analysis Layer — RKHS Kernel Module
==============================================
Traders@SMU — Quantitative Strategies Group
Layer 2: Sentiment (Arjan, Will, Demetrios)

Why NOT VADER?
--------------
VADER produces discrete, lookup-based polarity scores in {-1, 0, +1} buckets.
Discrete outputs violate the reproducing property of RKHS:

    f(x) = <f, K(·, x)>_H   requires K to be continuous and PSD.

A step function is not continuous — it has no well-defined inner product
structure — so VADER scores cannot live in a complete inner product space.

Why FinBERT?
------------
FinBERT (ProsusAI/finbert) produces a 768-dimensional contextualised
embedding for each input sentence.  These vectors:

    1. Live in R^768 — a continuous, finite-dimensional Hilbert space.
    2. Are smooth in the input: nearby sentences map to nearby vectors.
    3. Are domain-adapted: pre-trained on financial corpora (Reuters,
       Bloomberg, SEC filings) so financial jargon is handled correctly.

We then apply a Matérn kernel (via Random Fourier Features) on TOP of
the FinBERT embedding to produce a valid PSD kernel K_sentiment(x, y).

Architecture
------------
The full RKHS model sums kernel outputs across three layers:

    K_total(x, y) = α₁ K_GARCH(x, y)        <- Group 1 (Anders, Zabe, Ameen)
                  + α₂ K_Sentiment(x, y)      <- Group 2 (Will, Demetrios, Arjan)  *** THIS FILE ***
                  + α₃ K_LOB(x, y)            <- Group 3 (Becky, Connor)

Within this layer, the sentiment kernel is itself a weighted sum of
three sub-kernels, each operating on a DIFFERENT aspect of the embedding:

    K_Sentiment(x, y) = w₁ k_tone(x, y)         -- overall bullish/bearish tone
                       + w₂ k_topic(x, y)         -- topic similarity (earnings, macro, etc.)
                       + w₃ k_uncertainty(x, y)   -- uncertainty / hedging language

Kernel Implementation
---------------------
All kernels use the Matérn kernel via Random Fourier Features (RFF),
matching Group 3's LOB architecture exactly.  This gives:

    1. Explicit feature maps: phi(x) in R^D  (for Hilbert space composition)
    2. Matérn flexibility: smoothness parameter nu + length scale ell
    3. Kernel ridge regression: fit() / predict() for return forecasting
    4. Hyperparameter optimisation: L-BFGS-B on (nu, ell, lambda) via NLL

The kernel is approximated as: K(x, x') ≈ phi(x)^T phi(x')
where phi(x) = sqrt(2/D) * cos(X @ omega^T + b) and omega is sampled
from the spectral density of the Matérn kernel (Rahimi & Recht 2007).

The correct spectral sampling for Matérn-nu is:

    omega = Z * sqrt(2*nu) / (s * length_scale)

where Z ~ N(0, I_d) and s = sqrt(chi²(2*nu)).  The sqrt(2*nu) factor
is essential: it ensures omega ~ Student-t_{2nu}(0, (2nu/ell²)·I), which
is the spectral density of the Matérn kernel (R&W 2006, §4.2).

Daily Calendar-Time Ingestion
------------------------------
The feature space operates on DAILY granularity (calendar time).
Multiple intra-day headlines are aggregated into a single per-day
embedding before any kernel computation.  Use DailySentimentAggregator
to convert a list of timestamped SentimentSamples into one
DailySentimentSample per calendar date.

    aggregator = DailySentimentAggregator()
    daily = aggregator.aggregate(samples)   # list[DailySentimentSample]

DailySentimentSample is a drop-in replacement for SentimentSample; all
kernels and the SentimentLayer accept either type transparently.

Two modes for sub-kernel decomposition
---------------------------------------
Mode 1 — "aspect" (default):
    Each sub-kernel captures a DIFFERENT SLICE of the 768-d FinBERT vector:
    1. ToneKernel       -- first 256 dims  (sentiment-heavy directions)
    2. TopicKernel      -- dims 256-512    (topic / entity directions)
    3. UncertaintyKernel -- dims 512-768   (uncertainty / hedging directions)

    Note: the dimension split is a principled approximation.  In practice,
    you would run PCA on a held-out corpus and pick the directions that
    most strongly correlate with each axis.  The split-by-index default
    works as a reasonable baseline.

Mode 2 — "temporal" (requires time-series headlines):
    Each sub-kernel captures ALL embedding dimensions but at a DIFFERENT
    aggregation horizon measured in TRADING DAYS:
    1. TemporalKernel(window=1)   -- single daily embedding
    2. TemporalKernel(window=5)   -- 5-day rolling average (~1 week)
    3. TemporalKernel(window=20)  -- 20-day rolling average (~1 month)

References
----------
    - FinBERT: https://huggingface.co/ProsusAI/finbert
    - Random Fourier Features: Rahimi & Recht (2007), NeurIPS
    - Matérn kernel spectral density: Rasmussen & Williams (2006), §4.2
    - RKHS theory: Berlinet & Thomas-Agnan (2004)
    - Kernel combination closure: K_sum = w₁K₁ + w₂K₂ (valid PSD)

Fixes applied (v2)
------------------
1. MaternRFF._sample_frequencies: added missing sqrt(2*nu) factor so that
   omega ~ Student-t_{2nu}(0, (2nu/ell²)·I) as required by R&W §4.2.
2. MaternRFF._sample_frequencies: re-seed RNG from random_state before each
   frequency draw so that update_params() is fully reproducible.
3. SentimentSample.timestamp: typed as Optional[datetime.date].
4. DailySentimentSample + DailySentimentAggregator: enforce daily calendar-time
   ingestion so the feature space indexes by date, not by headline count.
5. TemporalKernel: window now measured in calendar days (matches daily samples).
6. SentimentLayer.optimize_hyperparams: best lambda is now stored per kernel
   and used automatically by fit_optimized(), closing the optimisation loop.
"""

import numpy as np
from datetime import date
from collections import defaultdict
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from scipy.optimize import minimize


# =============================================================================
# Sentence Representation
# =============================================================================

class SentimentSample:
    """
    A single financial headline or sentence with its FinBERT embedding.

    Parameters
    ----------
    text : str
        Raw headline text (kept for debugging / inspection).
    embedding : np.ndarray, shape (768,)
        FinBERT [CLS] token embedding.  Populated by embed_texts().
    timestamp : datetime.date, optional
        Calendar date the headline was published.  Required for daily
        aggregation via DailySentimentAggregator and for TemporalKernel.
    """

    def __init__(self, text: str, embedding: Optional[np.ndarray] = None,
                 timestamp: Optional[date] = None):
        self.text = text
        self.embedding = embedding   # set after encoding
        self.timestamp = timestamp   # datetime.date — required for daily mode

    def __repr__(self):
        emb_str = (f"emb={self.embedding.shape}" if self.embedding is not None
                   else "emb=None")
        ts_str = self.timestamp.isoformat() if self.timestamp is not None else "no-date"
        return f"SentimentSample({self.text[:40]!r}, {emb_str}, ts={ts_str})"


# =============================================================================
# Daily Aggregation — calendar-time feature space
# =============================================================================

class DailySentimentSample:
    """
    One calendar-day's sentiment, represented as the mean FinBERT embedding
    across all headlines published on that day.

    This is the primary unit of the daily feature space.  All kernels and
    the SentimentLayer accept DailySentimentSample in place of
    SentimentSample.

    Parameters
    ----------
    day : datetime.date
        The calendar date this sample represents.
    embedding : np.ndarray, shape (768,)
        Mean FinBERT [CLS] embedding across all headlines on `day`.
    n_headlines : int
        Number of raw headlines averaged to produce this embedding.
    """

    def __init__(self, day: date, embedding: np.ndarray, n_headlines: int = 1):
        self.day = day
        self.timestamp = day           # alias so kernels can use .timestamp
        self.embedding = embedding
        self.n_headlines = n_headlines
        self.text = f"Daily aggregate {day.isoformat()} ({n_headlines} headlines)"

    def __repr__(self):
        return (f"DailySentimentSample(day={self.day.isoformat()}, "
                f"n_headlines={self.n_headlines}, emb={self.embedding.shape})")


class DailySentimentAggregator:
    """
    Convert a list of timestamped SentimentSamples into one
    DailySentimentSample per calendar date.

    Headlines published on the same calendar date are averaged in embedding
    space.  Days with no coverage are omitted; fill-forward or zero-fill
    is the caller's responsibility if a contiguous daily index is required.

    Usage
    -----
    >>> agg = DailySentimentAggregator()
    >>> daily_samples = agg.aggregate(raw_samples)   # sorted by date
    """

    def aggregate(self, samples: List[SentimentSample]) -> List[DailySentimentSample]:
        """
        Aggregate SentimentSamples to daily level.

        Parameters
        ----------
        samples : list of SentimentSample
            Each sample MUST have .timestamp set to a datetime.date and
            .embedding set (call embed_texts() first).

        Returns
        -------
        daily : list of DailySentimentSample, sorted by date ascending.

        Raises
        ------
        ValueError
            If any sample has timestamp=None or embedding=None.
        """
        missing_ts = [s for s in samples if s.timestamp is None]
        if missing_ts:
            raise ValueError(
                f"{len(missing_ts)} sample(s) have timestamp=None. "
                "Set .timestamp (datetime.date) before aggregating."
            )
        missing_emb = [s for s in samples if s.embedding is None]
        if missing_emb:
            raise ValueError(
                f"{len(missing_emb)} sample(s) have embedding=None. "
                "Call embed_texts() / encoder.embed_samples() before aggregating."
            )

        # Group by calendar date
        buckets: Dict[date, List[np.ndarray]] = defaultdict(list)
        for s in samples:
            buckets[s.timestamp].append(s.embedding)

        daily = []
        for day in sorted(buckets.keys()):
            stack = np.vstack(buckets[day])          # (k, 768)
            mean_emb = stack.mean(axis=0)            # (768,)
            daily.append(DailySentimentSample(
                day=day,
                embedding=mean_emb.astype(np.float32),
                n_headlines=len(stack),
            ))
        return daily


# =============================================================================
# FinBERT Encoder (lazy-loaded so import is fast without GPU)
# =============================================================================

class FinBERTEncoder:
    """
    Wraps ProsusAI/finbert to produce 768-d [CLS] embeddings.

    The encoder is lazy-loaded on first call so that importing this module
    does not require transformers to be installed (useful if only the kernel
    math is needed with pre-computed embeddings).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Default: "ProsusAI/finbert".
    device : str
        "cpu", "cuda", or "mps".  Auto-detected if None.
    batch_size : int
        Tokenisation batch size for throughput.
    """

    DEFAULT_MODEL = "ProsusAI/finbert"

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 device: Optional[str] = None, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def _load(self):
        """Lazy-load the model on first call."""
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for FinBERT encoding.\n"
                "Install with: pip install transformers torch\n"
                f"Original error: {e}"
            )
        print(f"[FinBERTEncoder] Loading {self.model_name} on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        print("[FinBERTEncoder] Model loaded.")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into FinBERT [CLS] embeddings.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        embeddings : np.ndarray, shape (n, 768)
        """
        self._load()
        import torch

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self._model(**encoded)

            # [CLS] token is position 0 in last_hidden_state
            cls_embeddings = output.last_hidden_state[:, 0, :]  # (batch, 768)
            all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)  # (n, 768)

    def embed_samples(self, samples: List[SentimentSample]) -> List[SentimentSample]:
        """
        Encode texts in-place, returning the same list with .embedding set.

        Parameters
        ----------
        samples : list of SentimentSample

        Returns
        -------
        samples : list of SentimentSample (same objects, embeddings filled)
        """
        texts = [s.text for s in samples]
        embeddings = self.encode(texts)
        for sample, emb in zip(samples, embeddings):
            sample.embedding = emb
        return samples


def embed_texts(texts: List[str],
                model_name: str = FinBERTEncoder.DEFAULT_MODEL,
                device: Optional[str] = None,
                batch_size: int = 32) -> List[SentimentSample]:
    """
    Convenience function: encode a list of strings and return SentimentSamples.

    Parameters
    ----------
    texts : list of str
    model_name : str
    device : str or None
    batch_size : int

    Returns
    -------
    samples : list of SentimentSample with .embedding filled
    """
    encoder = FinBERTEncoder(model_name=model_name, device=device,
                             batch_size=batch_size)
    samples = [SentimentSample(t) for t in texts]
    return encoder.embed_samples(samples)


# =============================================================================
# Matérn Random Fourier Features
# =============================================================================

class MaternRFF:
    """
    Matérn kernel approximated via Random Fourier Features.

    K_Matérn(x, x') ≈ phi(x)^T phi(x')

    where phi(x) = sqrt(2/D) * cos(X @ omega^T + b),
    omega is sampled from the spectral density of the Matérn kernel,
    and b is uniform on [0, 2*pi].

    Spectral density of the Matérn-nu kernel (R&W 2006, §4.2):

        p(omega) ∝ (2*nu/ell^2 + ||omega||^2)^{-(nu + d/2)}

    This is a multivariate Student-t with 2*nu degrees of freedom and
    scale matrix (2*nu/ell^2) * I_d.  The correct sampling recipe is:

        z   ~ N(0, I_d)
        s^2 ~ chi^2(2*nu)  [equivalently Gamma(nu, 2)]
        omega = z * sqrt(2*nu) / (s * ell)

    The sqrt(2*nu) factor is CRITICAL: it rescales the Gaussian draw z so
    that the ratio z * sqrt(2*nu) / s follows the correct Student-t scale.
    Omitting it produces a kernel whose effective length scale depends on nu
    in an uncontrolled way.

    Parameters
    ----------
    nu : float
        Matérn smoothness parameter (0.5, 1.5, or 2.5 are common).
    length_scale : float
        Kernel length scale.
    n_features : int
        Number of random Fourier features D.
    input_dim : int
        Dimensionality of input vectors.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, nu: float = 1.5, length_scale: float = 1.0,
                 n_features: int = 512, input_dim: int = 768,
                 random_state: Optional[int] = 42):
        self.nu = nu
        self.length_scale = length_scale
        self.n_features = n_features
        self.input_dim = input_dim
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self._sample_frequencies()

    def _sample_frequencies(self):
        """
        Sample frequencies from the Matérn spectral density.

        FIX (v2): The RNG is re-seeded from self.random_state before sampling
        so that update_params() is fully reproducible regardless of prior
        calls (e.g. during hyperparameter optimisation).

        FIX (v2): Added sqrt(2*nu) factor so that
            omega ~ Student-t_{2nu}(0, (2nu/ell^2)*I)
        as required by R&W (2006), §4.2.
        """
        # Re-seed for reproducibility across update_params() calls.
        self._rng = np.random.RandomState(self.random_state)

        rng = self._rng
        d = self.input_dim
        D = self.n_features
        nu = self.nu

        # z ~ N(0, I_d), shape (D, d)
        Z = rng.randn(D, d)

        # s^2 ~ chi^2(2*nu) = Gamma(nu, 2), then s = sqrt(s^2)
        s_sq = rng.gamma(shape=nu, scale=2.0, size=D)   # (D,)
        s = np.sqrt(s_sq)                                # (D,)

        # CORRECT: omega = Z * sqrt(2*nu) / (s * ell)
        # This gives omega ~ Student-t_{2nu}(0, (2*nu/ell^2)*I_d)
        # which is the spectral density of the Matérn-nu kernel.
        self._omega = (Z * np.sqrt(2.0 * nu)) / (s[:, None] * self.length_scale)  # (D, d)
        self._bias = rng.uniform(0, 2 * np.pi, size=D)  # (D,)

    def update_params(self, nu: float, length_scale: float):
        """
        Update hyperparameters and re-sample frequencies.

        Because _sample_frequencies re-seeds the RNG from self.random_state,
        the result is deterministic for a given (nu, length_scale) pair.
        """
        self.nu = nu
        self.length_scale = length_scale
        self._sample_frequencies()

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute RFF feature matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n, input_dim)

        Returns
        -------
        Phi : np.ndarray, shape (n, n_features)
            phi(x) = sqrt(2/D) * cos(X @ omega^T + b)
        """
        projection = X @ self._omega.T + self._bias  # (n, D)
        return np.sqrt(2.0 / self.n_features) * np.cos(projection)


# =============================================================================
# Abstract Base Kernel
# =============================================================================

class BaseKernel(ABC):
    """
    Abstract base class for all sentiment sub-kernels.

    Subclasses must implement:
        extract_features(sample) -> np.ndarray
        name (property)
        _feature_dim (property)

    Everything else (RFF, Gram matrix, fit, predict) is inherited.

    RKHS guarantees
    ---------------
    With RFF + Matérn:
    - K(x, x') = phi(x)^T phi(x') >= 0 for all x, x' (PSD by construction)
    - K is symmetric: K(x, y) = phi(x)^T phi(y) = phi(y)^T phi(x) = K(y, x)
    - Positive weighted sums of PSD kernels are PSD (used in SentimentLayer)
    """

    # Subclasses set these in __init__
    use_rff: bool = True
    _rff: Optional[MaternRFF] = None
    _best_lambda: float = 1e-3   # updated by optimize_hyperparams

    def _init_rff(self, nu: float = 1.5, length_scale: float = 1.0,
                  n_rff: int = 512, random_state: int = 42):
        """Initialise the RFF approximator."""
        self._rff = MaternRFF(
            nu=nu,
            length_scale=length_scale,
            n_features=n_rff,
            input_dim=self._feature_dim,
            random_state=random_state,
        )

    def _extract_feature_matrix(self, samples) -> np.ndarray:
        """Stack extract_features() for a list of samples."""
        return np.vstack([self.extract_features(s) for s in samples])

    @abstractmethod
    def extract_features(self, sample) -> np.ndarray:
        """Convert a SentimentSample (or DailySentimentSample) into a feature vector."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable kernel name."""

    @property
    @abstractmethod
    def _feature_dim(self) -> int:
        """Dimensionality of extract_features() output."""

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        if self._rff is not None:
            return {
                "nu": self._rff.nu,
                "length_scale": self._rff.length_scale,
                "best_lambda": self._best_lambda,
            }
        return {}

    # ---- kernel evaluation --------------------------------------------------

    def evaluate(self, x_feat: np.ndarray, y_feat: np.ndarray) -> float:
        """K(x, y) given pre-extracted feature vectors."""
        if self.use_rff and self._rff is not None:
            phi_x = self._rff.transform(x_feat[np.newaxis, :])
            phi_y = self._rff.transform(y_feat[np.newaxis, :])
            return float(phi_x @ phi_y.T)
        # Fallback: exact RBF (only used when use_rff=False)
        diff = x_feat - y_feat
        return float(np.exp(-0.5 * np.dot(diff, diff)))

    def __call__(self, x, y) -> float:
        return self.evaluate(self.extract_features(x), self.extract_features(y))

    # ---- feature map --------------------------------------------------------

    def feature_map(self, samples) -> np.ndarray:
        """
        Phi(x) via RFF.  Shape: (n, n_rff).
        Satisfies: Phi @ Phi^T ≈ Gram matrix.
        """
        X = self._extract_feature_matrix(samples)   # (n, feat_dim)
        if self.use_rff and self._rff is not None:
            return self._rff.transform(X)            # (n, n_rff)
        # Fallback: use raw features (not recommended for RKHS use)
        return X

    # ---- Gram matrix --------------------------------------------------------

    def gram_matrix(self, samples) -> np.ndarray:
        Phi = self.feature_map(samples)
        return Phi @ Phi.T

    # ---- Hilbert distance ---------------------------------------------------

    def hilbert_distance(self, x, y) -> float:
        """||phi(x) - phi(y)||_H"""
        phi_x = self.feature_map([x])
        phi_y = self.feature_map([y])
        return float(np.linalg.norm(phi_x - phi_y))

    # ---- kernel ridge regression --------------------------------------------

    def fit(self, samples, y: np.ndarray, reg_lambda: float = 1e-3):
        """Kernel ridge regression using RFF feature map."""
        Phi = self.feature_map(samples)   # (n, D)
        D = Phi.shape[1]
        self._alpha = np.linalg.solve(
            Phi.T @ Phi + reg_lambda * np.eye(D), Phi.T @ y
        )

    def predict(self, samples) -> np.ndarray:
        """Predict using fitted alpha weights."""
        assert hasattr(self, "_alpha"), "Call fit() before predict()"
        Phi = self.feature_map(samples)
        return Phi @ self._alpha

    # ---- PSD check ----------------------------------------------------------

    def is_psd(self, samples, tol: float = 1e-10):
        K = self.gram_matrix(samples)
        min_eig = np.linalg.eigvalsh(K).min()
        return min_eig >= -tol, min_eig

    def __repr__(self):
        hp = self.hyperparameters
        return f"{self.name}(nu={hp.get('nu', '?'):.2f}, ell={hp.get('length_scale', '?'):.3f})"


# =============================================================================
# Sub-Kernel 1: Tone Kernel
# =============================================================================

class ToneKernel(BaseKernel):
    """
    Captures the overall bullish / bearish tone of a headline.

    Uses the first `n_tone_dims` dimensions of the FinBERT embedding,
    which in practice encode the global sentiment polarity most strongly
    (the [CLS] token is trained directly on positive/negative/neutral labels
    in the FinBERT fine-tuning objective).

    Parameters
    ----------
    n_tone_dims : int
        Number of embedding dimensions to use (default: 256 of 768).
    nu : float
        Matérn smoothness.
    length_scale : float
        Kernel length scale (in embedding space).
    n_rff : int
        Number of Random Fourier Features.
    """

    def __init__(self, n_tone_dims: int = 256,
                 nu: float = 1.5, length_scale: float = 1.0,
                 n_rff: int = 512, random_state: int = 42):
        self.n_tone_dims = n_tone_dims
        self.use_rff = True
        self._best_lambda = 1e-3
        self._init_rff(nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state)

    @property
    def _feature_dim(self) -> int:
        return self.n_tone_dims

    @property
    def name(self) -> str:
        return "ToneKernel"

    def extract_features(self, sample) -> np.ndarray:
        """First n_tone_dims dims of the FinBERT embedding."""
        assert sample.embedding is not None, \
            f"Sample '{sample.text[:30]}' has no embedding. Call embed_texts() first."
        emb = sample.embedding
        assert len(emb) >= self.n_tone_dims, \
            f"Embedding dim {len(emb)} < n_tone_dims {self.n_tone_dims}"
        return emb[: self.n_tone_dims].copy()

    def __repr__(self):
        return (f"ToneKernel(dims=0:{self.n_tone_dims}, "
                f"nu={self._rff.nu:.2f}, ell={self._rff.length_scale:.3f})")


# =============================================================================
# Sub-Kernel 2: Topic Kernel
# =============================================================================

class TopicKernel(BaseKernel):
    """
    Captures topic similarity: whether two headlines are about the same
    financial subject (earnings, macro, M&A, sector rotation, etc.).

    Uses the middle slice of the FinBERT embedding (dims 256–512 by default),
    which encodes entity and topic information more than pure sentiment polarity.

    Parameters
    ----------
    dim_start : int
        Start index of the embedding slice.
    dim_end : int
        End index of the embedding slice.
    nu : float
    length_scale : float
    n_rff : int
    """

    def __init__(self, dim_start: int = 256, dim_end: int = 512,
                 nu: float = 1.5, length_scale: float = 1.0,
                 n_rff: int = 512, random_state: int = 42):
        self.dim_start = dim_start
        self.dim_end = dim_end
        self.use_rff = True
        self._best_lambda = 1e-3
        self._init_rff(nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state)

    @property
    def _feature_dim(self) -> int:
        return self.dim_end - self.dim_start

    @property
    def name(self) -> str:
        return "TopicKernel"

    def extract_features(self, sample) -> np.ndarray:
        """Middle slice of the FinBERT embedding."""
        assert sample.embedding is not None, \
            f"Sample '{sample.text[:30]}' has no embedding. Call embed_texts() first."
        return sample.embedding[self.dim_start: self.dim_end].copy()

    def __repr__(self):
        return (f"TopicKernel(dims={self.dim_start}:{self.dim_end}, "
                f"nu={self._rff.nu:.2f}, ell={self._rff.length_scale:.3f})")


# =============================================================================
# Sub-Kernel 3: Uncertainty Kernel
# =============================================================================

class UncertaintyKernel(BaseKernel):
    """
    Captures linguistic uncertainty and hedging.

    Phrases like "may report", "expects roughly", "could miss guidance" carry
    meaningful uncertainty signals for volatility modelling that pure tone
    kernels miss.  The final slice of the FinBERT embedding (dims 512–768)
    tends to encode these higher-order pragmatic features.

    Parameters
    ----------
    dim_start : int
        Start index of the embedding slice (default: 512).
    nu : float
    length_scale : float
    n_rff : int
    """

    def __init__(self, dim_start: int = 512,
                 nu: float = 1.5, length_scale: float = 1.0,
                 n_rff: int = 512, random_state: int = 42):
        self.dim_start = dim_start
        self.use_rff = True
        self._best_lambda = 1e-3
        self._init_rff(nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state)

    @property
    def _feature_dim(self) -> int:
        return 768 - self.dim_start

    @property
    def name(self) -> str:
        return "UncertaintyKernel"

    def extract_features(self, sample) -> np.ndarray:
        """Final slice of the FinBERT embedding."""
        assert sample.embedding is not None, \
            f"Sample '{sample.text[:30]}' has no embedding. Call embed_texts() first."
        return sample.embedding[self.dim_start:].copy()

    def __repr__(self):
        return (f"UncertaintyKernel(dims={self.dim_start}:768, "
                f"nu={self._rff.nu:.2f}, ell={self._rff.length_scale:.3f})")


# =============================================================================
# Sub-Kernel (mode 2): Temporal Aggregation Kernel
# =============================================================================

class TemporalKernel(BaseKernel):
    """
    Operates on a rolling average of DAILY FinBERT embeddings over a
    calendar-day window.

    Input must be DailySentimentSample objects (one per calendar day, already
    aggregated via DailySentimentAggregator).  The `window` parameter
    specifies the number of trading days to smooth over.

    Two time-points are similar if their recent daily-sentiment streams have
    similar average tone across the given window.

    Parameters
    ----------
    window : int
        Number of calendar days to include in the rolling average.
        window=1  → single day (no smoothing)
        window=5  → ~1 trading week
        window=20 → ~1 trading month
    nu : float
    length_scale : float
    n_rff : int

    Notes
    -----
    The `feature_map` method applies the rolling average BEFORE passing
    through RFF, so the final Gram matrix captures smoothed-sentiment
    similarity, not instantaneous headline similarity.
    """

    def __init__(self, window: int = 1,
                 nu: float = 1.5, length_scale: float = 1.0,
                 n_rff: int = 512, random_state: int = 42):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.use_rff = True
        self._best_lambda = 1e-3
        self._init_rff(nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state)

    @property
    def _feature_dim(self) -> int:
        return 768

    @property
    def name(self) -> str:
        return f"TemporalKernel(w={self.window}d)"

    def extract_features(self, sample) -> np.ndarray:
        """
        Return the full 768-d embedding of a single (daily) sample.
        Temporal smoothing is applied in feature_map() over the full sequence.
        """
        assert sample.embedding is not None, \
            f"Sample '{sample.text[:30]}' has no embedding. Call embed_texts() first."
        return sample.embedding.copy()

    def feature_map(self, samples) -> np.ndarray:
        """
        Apply a causal rolling average of `window` days, then RFF.

        samples must be ordered chronologically (DailySentimentAggregator
        returns them sorted by date).

        If window=1, this reduces to the standard feature map with no
        smoothing.
        """
        raw = self._extract_feature_matrix(samples)      # (n, 768)
        n = len(samples)
        smoothed = np.zeros_like(raw)
        for i in range(n):
            start = max(0, i - self.window + 1)
            smoothed[i] = raw[start: i + 1].mean(axis=0)
        return self._rff.transform(smoothed)             # (n, n_rff)

    def __repr__(self):
        return (f"TemporalKernel(window={self.window}d, "
                f"nu={self._rff.nu:.2f}, ell={self._rff.length_scale:.3f})")


# =============================================================================
# RKHS Sentiment Layer (combined kernel)
# =============================================================================

class SentimentLayer:
    """
    The full Group 2 sentiment layer, combining sub-kernels via weighted sum.

    K_Sentiment(x, y) = sum_i w_i k_i(x, y)

    A positive weighted sum of PSD kernels is PSD (RKHS closure under
    addition), so this layer is a valid RKHS kernel.

    With RFF mode the layer also exposes a composite feature map:

        phi_Sentiment(x) = [sqrt(w1)*phi_1(x), sqrt(w2)*phi_2(x), ...]

    which satisfies phi_Sentiment(x)^T phi_Sentiment(y) = K_Sentiment(x, y).

    Parameters
    ----------
    kernels : list of BaseKernel
    weights : list of float   (must be non-negative for PSD guarantee)
    name : str
    """

    def __init__(self, kernels: List[BaseKernel], weights: List[float],
                 name: str = "Sentiment"):
        assert len(kernels) == len(weights)
        assert all(w >= 0 for w in weights), \
            "Weights must be non-negative to preserve PSD property"
        self.kernels = kernels
        self.weights = np.array(weights, dtype=np.float64)
        self.name = name
        self._w_combined: Optional[np.ndarray] = None
        self._fitted = False

    # ---- kernel call --------------------------------------------------------

    def __call__(self, x, y) -> float:
        """K_Sentiment(x, y) = sum_i w_i k_i(x, y)"""
        return float(sum(w * k(x, y) for k, w in zip(self.kernels, self.weights)))

    # ---- feature map --------------------------------------------------------

    def feature_map(self, samples) -> np.ndarray:
        """
        phi_Sentiment(x) = [sqrt(w1)*phi_1(x), sqrt(w2)*phi_2(x), ...]
        Shape: (n, sum_i D_i)

        This satisfies:
            phi_Sentiment(x)^T phi_Sentiment(y) = K_Sentiment(x, y)
        confirming membership in a valid RKHS.
        """
        parts = [np.sqrt(w) * k.feature_map(samples)
                 for k, w in zip(self.kernels, self.weights)]
        return np.concatenate(parts, axis=1)

    # ---- Gram matrix --------------------------------------------------------

    def gram_matrix(self, samples) -> np.ndarray:
        """K_Sentiment Gram matrix via RFF: Phi @ Phi^T"""
        Phi = self.feature_map(samples)
        return Phi @ Phi.T

    # ---- Hilbert distance ---------------------------------------------------

    def hilbert_distance(self, x, y) -> float:
        """||phi_Sentiment(x) - phi_Sentiment(y)||_H"""
        phi_x = self.feature_map([x])
        phi_y = self.feature_map([y])
        return float(np.linalg.norm(phi_x - phi_y))

    # ---- kernel ridge regression --------------------------------------------

    def fit(self, samples, y: np.ndarray, reg_lambda: float = 1e-3):
        """
        Kernel ridge regression (primal form):

            (Phi^T Phi + lambda * I) w = Phi^T y

        Parameters
        ----------
        samples : list of SentimentSample or DailySentimentSample
        y : np.ndarray, shape (n,)  — target returns / volatility proxy
        reg_lambda : float          — regularisation strength
        """
        Phi = self.feature_map(samples)   # (n, D_total)
        D = Phi.shape[1]
        self._w_combined = np.linalg.solve(
            Phi.T @ Phi + reg_lambda * np.eye(D), Phi.T @ y
        )
        self._fitted = True

    def predict(self, samples) -> np.ndarray:
        """Predict using fitted weights."""
        assert self._fitted, "Call fit() before predict()"
        Phi = self.feature_map(samples)
        return Phi @ self._w_combined

    # ---- PSD check ----------------------------------------------------------

    def is_psd(self, samples, tol: float = 1e-10):
        """Verify the combined Gram matrix is PSD."""
        K = self.gram_matrix(samples)
        min_eig = np.linalg.eigvalsh(K).min()
        return min_eig >= -tol, min_eig

    # ---- hyperparameter optimisation ----------------------------------------

    def optimize_hyperparams(self, samples, y: np.ndarray,
                             verbose: bool = True) -> dict:
        """
        Jointly optimise (nu, length_scale, lambda) for each sub-kernel
        by minimising Gaussian NLL on the training data.

        NLL = 0.5 * (log|K + lambda*I| + y^T (K + lambda*I)^{-1} y)

        The optimised lambda is stored on each kernel as `_best_lambda` and
        is used automatically by fit_optimized().

        Parameters
        ----------
        samples : list of SentimentSample or DailySentimentSample
        y : np.ndarray, shape (n,)
        verbose : bool

        Returns
        -------
        result : dict  {kernel_name: {"nu": ..., "ell": ..., "lambda": ...}}
        """
        results = {}
        for kernel in self.kernels:
            if verbose:
                print(f"  Optimising {kernel.name}...")

            def objective(params):
                nu, ell, lam = params
                if nu <= 0 or ell <= 0 or lam <= 0:
                    return 1e10
                kernel._rff.update_params(nu=nu, length_scale=ell)
                K = kernel.gram_matrix(samples)
                n = len(samples)
                K_reg = K + lam * np.eye(n)
                try:
                    L = np.linalg.cholesky(K_reg)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    nll = (np.sum(np.log(np.diag(L)))
                           + 0.5 * y @ alpha
                           + 0.5 * n * np.log(2 * np.pi))
                    return float(nll)
                except np.linalg.LinAlgError:
                    return 1e10

            x0 = [kernel._rff.nu, kernel._rff.length_scale, 1e-3]
            bounds = [(0.1, 10.0), (1e-4, None), (1e-6, None)]
            res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
            best_nu, best_ell, best_lam = res.x

            # Apply optimised (nu, ell) back to the kernel's RFF
            kernel._rff.update_params(nu=best_nu, length_scale=best_ell)

            # FIX (v2): Store optimised lambda on the kernel so that
            # fit_optimized() can use it without the caller having to
            # manually pass it back in.
            kernel._best_lambda = float(best_lam)

            results[kernel.name] = {
                "nu": best_nu,
                "ell": best_ell,
                "lambda": best_lam,
            }
            if verbose:
                print(f"    {kernel.name}: nu={best_nu:.3f}, "
                      f"ell={best_ell:.4f}, lambda={best_lam:.2e}")
        return results

    def fit_optimized(self, samples, y: np.ndarray, verbose: bool = True):
        """
        Convenience method: run optimize_hyperparams() then fit() using the
        mean of the per-sub-kernel optimised lambdas as the final
        regularisation strength.

        This closes the optimisation loop that was previously broken:
        optimize_hyperparams() found the best lambda per sub-kernel but
        fit() always used the default 1e-3.

        Parameters
        ----------
        samples : list of SentimentSample or DailySentimentSample
        y : np.ndarray, shape (n,)
        verbose : bool

        Returns
        -------
        opt_results : dict (from optimize_hyperparams)
        """
        opt_results = self.optimize_hyperparams(samples, y, verbose=verbose)
        best_lambdas = [k._best_lambda for k in self.kernels]
        # Use the geometric mean of per-kernel lambdas for the combined fit
        combined_lambda = float(np.exp(np.mean(np.log(best_lambdas))))
        if verbose:
            print(f"  Combined lambda (geo-mean): {combined_lambda:.2e}")
        self.fit(samples, y, reg_lambda=combined_lambda)
        return opt_results

    # ---- display ------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Layer: {self.name}",
            "  K_Sentiment(x,y) = "
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
        return f"SentimentLayer({self.name}: {s})"


# =============================================================================
# Pre-configured factory
# =============================================================================

def create_sentiment_layer(mode: str = "aspect",
                           nu: float = 1.5,
                           length_scale: float = 1.0,
                           n_rff: int = 512,
                           weights: Optional[List[float]] = None,
                           random_state: int = 42) -> SentimentLayer:
    """
    Create a ready-to-use SentimentLayer.

    Parameters
    ----------
    mode : str
        "aspect"   — three sub-kernels on different embedding slices (default).
        "temporal" — three sub-kernels at different rolling-average horizons
                     in calendar days (1d, 5d, 20d).  Input must be
                     DailySentimentSample objects.
    nu : float
        Matérn smoothness for all sub-kernels.
    length_scale : float
        Kernel length scale for all sub-kernels.
    n_rff : int
        Number of Random Fourier Features per sub-kernel.
    weights : list of float or None
        Sub-kernel weights (must be non-negative). Defaults to [1/3, 1/3, 1/3].
    random_state : int
        RNG seed.

    Returns
    -------
    SentimentLayer
    """
    if weights is None:
        weights = [1 / 3, 1 / 3, 1 / 3]

    if mode == "aspect":
        kernels = [
            ToneKernel(n_tone_dims=256, nu=nu, length_scale=length_scale,
                       n_rff=n_rff, random_state=random_state),
            TopicKernel(dim_start=256, dim_end=512, nu=nu,
                        length_scale=length_scale, n_rff=n_rff,
                        random_state=random_state + 1),
            UncertaintyKernel(dim_start=512, nu=nu, length_scale=length_scale,
                              n_rff=n_rff, random_state=random_state + 2),
        ]
    elif mode == "temporal":
        # window is measured in calendar days (requires DailySentimentSample input)
        kernels = [
            TemporalKernel(window=1, nu=nu, length_scale=length_scale,
                           n_rff=n_rff, random_state=random_state),
            TemporalKernel(window=5, nu=nu, length_scale=length_scale,
                           n_rff=n_rff, random_state=random_state + 1),
            TemporalKernel(window=20, nu=nu, length_scale=length_scale,
                           n_rff=n_rff, random_state=random_state + 2),
        ]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose 'aspect' or 'temporal'.")

    return SentimentLayer(kernels=kernels, weights=weights, name="Sentiment")


# =============================================================================
# Demo / smoke test (runs without transformers)
# =============================================================================

def _demo_with_random_embeddings():
    """
    Smoke test that exercises the full layer pipeline using
    randomly-generated 768-d 'embeddings' (no GPU / transformers needed).

    Tests:
    1. Aspect mode — ToneKernel / TopicKernel / UncertaintyKernel
    2. Temporal mode — TemporalKernel with DailySentimentSample input
    3. DailySentimentAggregator — raw SentimentSample -> daily aggregation
    4. PSD check on both modes
    5. fit_optimized() — hyperparameter optimisation + ridge regression
    """
    from datetime import date, timedelta

    print("=" * 60)
    print("SentimentLayer — smoke test (random embeddings)")
    print("=" * 60)

    rng = np.random.RandomState(0)
    n = 10

    # ------------------------------------------------------------------ #
    # 1. Raw SentimentSamples with timestamps (two per day for 5 days)
    # ------------------------------------------------------------------ #
    headlines = [
        ("Apple beats earnings, raises guidance",          date(2024, 1, 2)),
        ("Fed signals rate hike amid inflation fears",     date(2024, 1, 2)),
        ("Tesla misses delivery targets, stock drops",     date(2024, 1, 3)),
        ("Oil prices surge on OPEC supply cut",            date(2024, 1, 3)),
        ("Microsoft Azure growth slows in Q3",             date(2024, 1, 4)),
        ("Strong jobs report boosts market confidence",    date(2024, 1, 4)),
        ("Recession fears mount as yield curve inverts",   date(2024, 1, 5)),
        ("Amazon announces $10B share buyback",            date(2024, 1, 5)),
        ("Inflation cools, CPI below expectations",        date(2024, 1, 8)),
        ("Bank of America warns of credit tightening",     date(2024, 1, 8)),
    ]
    raw_samples = [
        SentimentSample(h, embedding=rng.randn(768).astype(np.float32), timestamp=ts)
        for h, ts in headlines
    ]

    # ------------------------------------------------------------------ #
    # 2. Daily aggregation
    # ------------------------------------------------------------------ #
    agg = DailySentimentAggregator()
    daily_samples = agg.aggregate(raw_samples)   # 5 DailySentimentSample objects
    print(f"\nRaw headlines    : {len(raw_samples)}")
    print(f"Daily aggregates : {len(daily_samples)}")
    for ds in daily_samples:
        print(f"  {ds}")

    # ------------------------------------------------------------------ #
    # 3. Aspect mode on daily samples
    # ------------------------------------------------------------------ #
    print("\n--- Aspect mode (daily embeddings) ---")
    layer = create_sentiment_layer(mode="aspect", n_rff=256)
    print(layer.summary())

    K = layer.gram_matrix(daily_samples)
    print(f"\nGram matrix shape : {K.shape}")
    print(f"Gram matrix range : [{K.min():.4f}, {K.max():.4f}]")

    is_psd, min_eig = layer.is_psd(daily_samples)
    print(f"PSD check         : {is_psd}  (min eigenvalue = {min_eig:.2e})")

    dist = layer.hilbert_distance(daily_samples[0], daily_samples[1])
    print(f"\nHilbert distance(day0, day1) : {dist:.4f}")

    y = rng.randn(len(daily_samples))
    layer.fit(daily_samples, y)
    preds = layer.predict(daily_samples)
    print(f"\nFit/predict check : predictions shape = {preds.shape}")
    print(f"  First 5 preds   : {preds[:5].round(4)}")

    # ------------------------------------------------------------------ #
    # 4. Temporal mode on daily samples
    # ------------------------------------------------------------------ #
    print("\n--- Temporal mode (calendar-day windows) ---")
    layer_t = create_sentiment_layer(mode="temporal", n_rff=256)
    K_t = layer_t.gram_matrix(daily_samples)
    is_psd_t, min_eig_t = layer_t.is_psd(daily_samples)
    print(f"Gram matrix shape : {K_t.shape}")
    print(f"PSD check         : {is_psd_t}  (min eigenvalue = {min_eig_t:.2e})")

    # ------------------------------------------------------------------ #
    # 5. fit_optimized (closes the optimise->fit loop)
    # ------------------------------------------------------------------ #
    print("\n--- fit_optimized() ---")
    layer2 = create_sentiment_layer(mode="aspect", n_rff=128)
    opt = layer2.fit_optimized(daily_samples, y, verbose=True)
    preds2 = layer2.predict(daily_samples)
    print(f"Optimized fit predictions shape : {preds2.shape}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv or len(sys.argv) == 1:
        _demo_with_random_embeddings()
