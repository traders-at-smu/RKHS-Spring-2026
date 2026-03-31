"""
vanna_charm_layer.py
=====================
VannaCharm RKHS Layer — dollar-bar-time feature space.

Captures the activity-triggered hedging flows of market makers measured
in DOLLAR-BAR TIME (intrinsic / market time):

    - Vanna un-hedging    (dealer delta changes from IV moves)
    - Charm decay         (theta / OPEX pinning in activity time)

Why dollar bars?
────────────────
Vanna and Charm hedging flows are TRIGGERED BY ACTIVITY, not by the
clock.  A dealer re-hedges delta when enough premium has traded to make
the adjustment economical — not because a minute or a day has passed.

In calendar time the autocorrelation of these flows is non-stationary:
  • High-vol days:  many large trades → charm re-hedged frequently
  • Low-vol days:   sparse trades     → charm bleed mostly unrealised

A dollar bar closes once per $D of underlying notional traded
(∑ price·volume ≥ D).  This means:
  • Each bar represents the same economic "work done" regardless of
    clock time.
  • The resulting process has stable, near-stationary autocorrelation
    structure in bar time even when the calendar-time process is
    strongly heteroskedastic.
  • Outlier bars (news events) naturally get more bars (finer
    resolution) precisely when dealers are most active.

Kernel:
    K_dbar(x, x') = k_ν(x₂, x₂') + k_τ^$(x₃, x₃')

    k_ν    : ScaleKernel(MaternKernel(ν=1.5))                  on dim 0
    k_τ^$  : ScaleKernel(PeriodicKernel × MaternKernel(ν=2.5)) on dim 1

Feature vector (2-D):
    x_dbar = [Net Vanna_b,  Net Charm_b]

    Net Vanna_b = Σ_K [ ν_call·OI_call − ν_put·OI_put ]  (per bar b)
    Net Charm_b = Σ_K [ τ_call·OI_call − τ_put·OI_put ]  (per bar b)

Z-score normalisation is applied in BAR TIME (rolling window measured
in bars, not calendar days).

Dollar-bar construction
───────────────────────
A new bar opens after the cumulative underlying dollar notional resets.
Per-bar aggregation correctly deduplicates the long-format (one row per
strike/type) input so that only one dollar-notional reading per intraday
timestamp is accumulated.

OPEX period in bars
───────────────────
The calendar OPEX cycle (~30 days) is re-expressed as a bar count:
    opex_bars ≈ opex_calendar_days × avg_bars_per_calendar_day
This is used as the median of the LogNormal prior on the PeriodicKernel
period_length parameter.  The prior scale is widened (0.25 vs the
calendar-space 0.15) because the bars-per-day ratio varies with the
dollar threshold and prevailing volume regime.

Fixes carried from v2
─────────────────────
  FIX-1  forward() calls sub-kernel via __call__ (no __call__ override)
  FIX-3  _rolling_zscore guard len(window) < 2  (std=0 blowup)
  FIX-4  LogNormalPrior on period_length (positive domain)
  FIX-5  Vectorized groupby[col].sum() — no lambda apply
  FIX-6  predict() saves/restores train/eval state

Authors: Refactored from dual-space design (March 2026)
Requirements: gpytorch >= 1.11, torch >= 2.0, pandas, numpy, matplotlib
"""

from __future__ import annotations

import math
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, PeriodicKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ExactGP
from gpytorch.priors import LogNormalPrior
from torch import Tensor


# ══════════════════════════════════════════════════════════════════════════════
# §0  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

OPEX_CALENDAR_DAYS:        float = 30.0
AVG_BARS_PER_CALENDAR_DAY: float = 5.0
DEFAULT_OPEX_BARS:         float = OPEX_CALENDAR_DAYS * AVG_BARS_PER_CALENDAR_DAY  # 150

REQUIRED_COLUMNS: frozenset = frozenset({
    "timestamp", "strike", "type",
    "vanna", "charm", "open_interest",
    "price",   # underlying mid/last — used to accumulate dollar notional
    "volume",  # underlying shares/contracts per intraday snapshot
})

# Feature vector layout
#   dim 0 : net_vanna
#   dim 1 : net_charm


# ══════════════════════════════════════════════════════════════════════════════
# §1  SUB-KERNELS
# ══════════════════════════════════════════════════════════════════════════════

class VannaKernel(ScaleKernel):
    """k_ν : ScaleKernel(MaternKernel(ν=1.5)) on active_dims=[0].

    Matérn-3/2 is C¹ — once differentiable — capturing rapid but
    continuous dealer delta adjustments driven by implied-vol changes.
    Vanna flows arrive with each increment of option premium traded,
    making this a natural dollar-bar-space feature.

    k(r) = σ² (1 + √3 r/ℓ) exp(−√3 r/ℓ)
    """

    def __init__(self, **kwargs) -> None:
        base_kernel = MaternKernel(nu=1.5, active_dims=torch.tensor([0]))
        super().__init__(base_kernel=base_kernel, **kwargs)

    @property
    def matern_kernel(self) -> MaternKernel:
        return self.base_kernel

    @property
    def lengthscale(self) -> Tensor:
        return self.base_kernel.lengthscale


class DollarBarCharmKernel(ScaleKernel):
    """k_τ^$ : ScaleKernel(PeriodicKernel × MaternKernel(ν=2.5)) on active_dims=[1].

    Calendar-time Charm adapted for DOLLAR-BAR time.

    Two jointly-necessary properties of theta-decay pressure:

        1.  Periodicity (PeriodicKernel):
            The OPEX calendar rhythm persists but is measured in BARS.
            Prior median is anchored at opex_bars; the kernel learns the
            actual bar-count period from data.

        2.  Smooth intraday theta bleed (Matérn ν=2.5, C²):
            Each $D bar advances the theta clock by a small, differentiable
            amount.  Matérn-2.5 correctly captures this C² smoothness in
            bar time.

    Why dollar bars improve Charm stationarity:
        In calendar time the charm autocorrelation depends on session
        length and vol regime — heteroskedastic, non-stationary.  Dollar
        bars sample once per fixed notional, so the "theta clock" ticks
        at a constant rate in bar time regardless of wall-clock speed.

    FIX-4: LogNormalPrior on period_length keeps the prior over ℝ⁺.
        NormalPrior assigned non-zero mass to negative values; gradient
        descent could push the raw parameter to a region where softplus
        produces near-zero period before the constraint stabilises.
        LogNormal(loc=log(opex_bars), scale=0.25) anchors the median at
        the OPEX cycle while allowing ≈25% relative log-uncertainty
        (wider than calendar prior of 0.15, reflecting uncertainty in
        the bars-per-day conversion).

    k(r) = σ² · exp(−2sin²(πr/p)/ℓ²) · (1+√5r/ℓ+5r²/(3ℓ²)) exp(−√5r/ℓ)
    where r is bar-distance, p is OPEX period in bars.
    """

    def __init__(
        self,
        opex_bars: float = DEFAULT_OPEX_BARS,
        period_prior_log_std: float = 0.25,
        **kwargs,
    ) -> None:
        # FIX-4: LogNormalPrior — positive domain, median at opex_bars
        periodic = PeriodicKernel(
            active_dims=torch.tensor([1]),
            period_length_prior=LogNormalPrior(
                loc=torch.tensor(math.log(opex_bars)),
                scale=torch.tensor(period_prior_log_std),
            ),
        )
        periodic.period_length = torch.tensor([[opex_bars]])

        matern25 = MaternKernel(nu=2.5, active_dims=torch.tensor([1]))

        # Product: joint necessity of periodicity AND smooth decay
        product_kernel = periodic * matern25
        super().__init__(base_kernel=product_kernel, **kwargs)
        self._periodic_kernel = periodic
        self._matern_kernel   = matern25

    @property
    def periodic_kernel(self) -> PeriodicKernel:
        return self._periodic_kernel

    @property
    def matern_kernel(self) -> MaternKernel:
        return self._matern_kernel

    @property
    def period_length(self) -> Tensor:
        return self._periodic_kernel.period_length

    @property
    def matern_lengthscale(self) -> Tensor:
        return self._matern_kernel.lengthscale


# ══════════════════════════════════════════════════════════════════════════════
# §2  COMPOSITE LAYER KERNEL
# ══════════════════════════════════════════════════════════════════════════════

class VannaCharmLayerKernel(Kernel):
    """K_dbar = k_ν(x₀,x₀') + k_τ^$(x₁,x₁')

    Additive composition over two dollar-bar-time channels.
    Input must be 2-D: x = [net_vanna, net_charm].

    FIX-1: forward() calls self._additive_kernel(x1, x2, ...) via __call__,
    not .forward(), preserving GPyTorch's lazy-eval pipeline and caching.
    No __call__ override.
    """

    def __init__(
        self,
        opex_bars: float = DEFAULT_OPEX_BARS,
        period_prior_log_std: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vanna_kernel = VannaKernel()
        self.charm_kernel = DollarBarCharmKernel(
            opex_bars=opex_bars,
            period_prior_log_std=period_prior_log_std,
        )
        # AdditiveKernel via GPyTorch '+' operator
        self._additive_kernel = self.vanna_kernel + self.charm_kernel

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params,
    ) -> Tensor:
        # FIX-1: via __call__, not .forward()
        return self._additive_kernel(x1, x2, diag=diag, **params)

    def get_channel_contributions(
        self,
        x1: Tensor,
        x2: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Return individual kernel matrices for variance decomposition."""
        if x2 is None:
            x2 = x1
        return {
            "vanna": self.vanna_kernel(x1, x2).to_dense(),
            "charm": self.charm_kernel(x1, x2).to_dense(),
        }

    def get_param_summary(self) -> Dict[str, float]:
        def _s(t: Tensor) -> float:
            return t.squeeze().item()
        return {
            "vanna_outputscale":   _s(self.vanna_kernel.outputscale),
            "vanna_lengthscale":   _s(self.vanna_kernel.matern_kernel.lengthscale),
            "charm_outputscale":   _s(self.charm_kernel.outputscale),
            "charm_period_bars":   _s(self.charm_kernel.period_length),
            "charm_matern_ls":     _s(self.charm_kernel.matern_lengthscale),
        }


# ══════════════════════════════════════════════════════════════════════════════
# §3  FEATURE CONSTRUCTOR
# ══════════════════════════════════════════════════════════════════════════════

class VannaCharmFeatureConstructor:
    """Intraday options + underlying flow → normalised 2-D [net_vanna, net_charm].

    Dollar-bar construction
    ───────────────────────
    A new bar closes when cumulative underlying dollar notional
    (∑ price·volume) reaches the threshold D.  This implements
    "intrinsic time":

      • High-activity periods → many bars per calendar day
      • Low-activity periods  → few bars per calendar day
      • Each bar = the same economic work regardless of wall-clock time

    Deduplication step:
        The input is long-format (one row per strike/type per timestamp).
        Dollar notional is accumulated using ONE reading per intraday
        timestamp (deduplicated before accumulation) to prevent bar-count
        inflation from the multi-row structure.

    Aggregation per bar b:
        Net Vanna_b = Σ_K [ ν_call·OI_call − ν_put·OI_put ]
        Net Charm_b = Σ_K [ τ_call·OI_call − τ_put·OI_put ]

    Z-score normalisation is applied in BAR TIME (rolling_window_bars),
    not calendar days, preserving the stationarity properties of dollar bars.

    FIX-3: rolling z-score guard len(window) < 2.
    FIX-5: vectorised groupby[col].sum() throughout.

    Args:
        dollar_bar_threshold: $ notional per bar (D).
            Calibrate so that AVG_BARS_PER_CALENDAR_DAY bars close per day
            on average (used to anchor the OPEX prior in bar units).
        rolling_window_bars: Past bars for rolling z-score (≥ 2).
        zscore_eps:          Stability constant added to rolling std.
        device:              Output tensor device.
        dtype:               Output tensor dtype.
    """

    CHANNEL_NAMES = ("net_vanna", "net_charm")

    def __init__(
        self,
        dollar_bar_threshold: float = 1_000_000.0,
        rolling_window_bars: int = 315,    # ≈ 63 days × 5 bars/day
        zscore_eps: float = 1e-8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if rolling_window_bars < 2:
            raise ValueError("rolling_window_bars must be >= 2")
        self.dollar_bar_threshold = dollar_bar_threshold
        self.rolling_window_bars  = rolling_window_bars
        self.zscore_eps           = zscore_eps
        self.device               = device or torch.device("cpu")
        self.dtype                = dtype
        self._bar_df:    Optional[pd.DataFrame] = None
        self._normalised: Optional[pd.DataFrame] = None

    def transform(self, flow_df: pd.DataFrame) -> Tuple[pd.DataFrame, Tensor]:
        """Convert intraday options + underlying data to normalised Vanna/Charm tensor.

        Args:
            flow_df: DataFrame with all REQUIRED_COLUMNS.
                'price' and 'volume' are the underlying mid-price and
                volume for each intraday snapshot row (the same value is
                repeated across all strike/type rows at the same timestamp).

        Returns:
            bar_df:  Raw (pre-normalisation) bar DataFrame with columns
                         [bar_id, bar_close_time, net_vanna, net_charm].
            tensor:  torch.Tensor shape (n_bars, 2), z-score normalised
                     in bar time.
        """
        self._validate(flow_df)
        bar_ids          = self._assign_bar_ids(flow_df)
        bar_df           = self._aggregate_bars(flow_df, bar_ids)
        self._bar_df     = bar_df.copy()
        normed           = self._rolling_zscore_bars(bar_df)
        self._normalised = normed.copy()
        values = normed[list(self.CHANNEL_NAMES)].to_numpy(dtype=np.float32)
        tensor = torch.tensor(values, dtype=self.dtype, device=self.device)
        return bar_df, tensor

    def get_bar_df(self) -> Optional[pd.DataFrame]:
        """Raw (pre-normalisation) bar DataFrame (after transform())."""
        return self._bar_df

    def get_normalised(self) -> Optional[pd.DataFrame]:
        """Post-normalisation bar DataFrame (after transform())."""
        return self._normalised

    # ── Bar construction ──────────────────────────────────────────────────────

    def _assign_bar_ids(self, df: pd.DataFrame) -> pd.Series:
        """Assign integer bar_id to each row via cumulative dollar notional.

        Deduplicates to one (timestamp, dollar_notional) record before
        accumulating — prevents bar ID inflation from the long-format
        multi-row structure.

        Returns:
            pd.Series aligned with df.index containing integer bar_ids.
        """
        ts_notional = (
            df.assign(_dn=df["price"] * df["volume"])
            .drop_duplicates(subset=["timestamp"])
            [["timestamp", "_dn"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        bar_ids_ts = np.empty(len(ts_notional), dtype=int)
        cumvol = 0.0
        bar_id = 0
        for i, dv in enumerate(ts_notional["_dn"].to_numpy()):
            bar_ids_ts[i] = bar_id
            cumvol += dv
            if cumvol >= self.dollar_bar_threshold:
                cumvol = 0.0
                bar_id += 1

        ts_to_bar = dict(
            zip(ts_notional["timestamp"].tolist(), bar_ids_ts.tolist())
        )
        return df["timestamp"].map(ts_to_bar).rename("bar_id")

    def _aggregate_bars(
        self, df: pd.DataFrame, bar_ids: pd.Series
    ) -> pd.DataFrame:
        """Aggregate long-format option rows into per-bar net exposures.

        FIX-5: vectorised groupby[col].sum(), no lambda apply.
        """
        df = df.copy()
        df["_bar_id"] = bar_ids.values
        df["type"]    = df["type"].str.lower().str.strip()

        for greek in ("vanna", "charm"):
            df[f"_w_{greek}"] = df[greek] * df["open_interest"]

        calls = df[df["type"] == "call"]
        puts  = df[df["type"] == "put"]

        def _net(col: str) -> pd.Series:
            # FIX-5: vectorised groupby.sum()
            c = calls.groupby("_bar_id")[col].sum()
            p = puts.groupby("_bar_id")[col].sum()
            all_bars = c.index.union(p.index)
            return (
                c.reindex(all_bars, fill_value=0.0)
                - p.reindex(all_bars, fill_value=0.0)
            )

        bar_close = df.groupby("_bar_id")["timestamp"].max().rename("bar_close_time")

        result = pd.DataFrame({
            "net_vanna": _net("_w_vanna"),
            "net_charm": _net("_w_charm"),
        })
        result.index.name = "bar_id"
        result = result.join(bar_close).fillna(0.0)
        result.sort_index(inplace=True)
        return result

    # ── Rolling z-score in bar time ───────────────────────────────────────────

    def _rolling_zscore_bars(self, bar_df: pd.DataFrame) -> pd.DataFrame:
        """Causal rolling z-score per channel in BAR time.

        Window is measured in bars, not calendar days.  A window of W bars
        always covers the same economic activity regardless of clock hours.

        FIX-3: guard len(window) < 2 prevents std=0 blowup at bar 1.
        """
        n      = len(bar_df)
        normed = bar_df.copy()
        for col in self.CHANNEL_NAMES:
            series = bar_df[col].to_numpy(dtype=np.float64)
            z = np.empty_like(series)
            for b in range(n):
                window = series[max(0, b - self.rolling_window_bars):b]
                if len(window) < 2:   # FIX-3
                    z[b] = 0.0
                else:
                    z[b] = (series[b] - window.mean()) / (
                        window.std(ddof=0) + self.zscore_eps
                    )
            normed[col] = z
        return normed

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        unknown = set(df["type"].str.lower().str.strip().unique()) - {"call", "put"}
        if unknown:
            raise ValueError(f"Column 'type' contains unknown values: {unknown}")
        if (df["volume"] < 0).any():
            raise ValueError("Column 'volume' must be non-negative.")

    # ── Synthetic data ────────────────────────────────────────────────────────

    @staticmethod
    def make_synthetic_dataframe(
        n_trading_days: int = 60,
        ticks_per_day: int = 78,           # ≈ 5-minute bars, 6.5hr session
        n_strikes: int = 8,
        dollar_bar_threshold: float = 1_000_000.0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate intraday per-strike options data for testing.

        Underlying price follows GBM; volume follows a log-normal with
        intraday U-shaped seasonality.
        """
        rng            = np.random.default_rng(seed)
        session_start  = pd.Timestamp("09:30")
        session_end    = pd.Timestamp("16:00")
        session_mins   = int((session_end - session_start).total_seconds() / 60)
        step_mins      = max(1, session_mins // ticks_per_day)
        base_dates     = pd.bdate_range("2024-01-02", periods=n_trading_days)

        timestamps: list = []
        for d in base_dates:
            for m in range(0, session_mins, step_mins):
                timestamps.append(d + pd.Timedelta(minutes=390 + m))

        n_ts     = len(timestamps)
        strikes  = np.linspace(100, 200, n_strikes)

        # Underlying GBM
        price_series = 150.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n_ts)))

        # U-shaped intraday volume
        within_day = np.tile(np.linspace(0, 1, ticks_per_day), n_trading_days)[:n_ts]
        vol_shape  = 1.5 - np.sin(within_day * np.pi) * 0.7
        volume_series = np.abs(
            rng.lognormal(mean=10.0, sigma=0.6, size=n_ts)
        ) * vol_shape

        rows = []
        for i, ts in enumerate(timestamps):
            p, v = price_series[i], volume_series[i]
            for strike in strikes:
                for opt_type in ("call", "put"):
                    sign = 1.0 if opt_type == "call" else -1.0
                    rows.append({
                        "timestamp":     ts,
                        "price":         p,
                        "volume":        v,
                        "strike":        strike,
                        "type":          opt_type,
                        "vanna":         sign * rng.uniform(0.01, 0.5),
                        "charm":         sign * rng.uniform(-0.02, 0.02),
                        "open_interest": rng.uniform(100, 5000),
                    })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# §4  EXACT GP MODEL
# ══════════════════════════════════════════════════════════════════════════════

class VannaCharmGP(ExactGP):
    """ExactGP for the dollar-bar-time VannaCharmLayerKernel.

    Args:
        train_x:     (n_bars, 2) — [net_vanna, net_charm], z-score normalised.
        train_y:     (n_bars,)   — per-bar price return or target.
        likelihood:  GaussianLikelihood; created if not provided.
        mean_type:   "constant" (default) or "zero".
        opex_bars:   OPEX cycle length for DollarBarCharmKernel prior anchor.
        **kernel_kwargs: Forwarded to VannaCharmLayerKernel.
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: Optional[GaussianLikelihood] = None,
        mean_type: str = "constant",
        opex_bars: float = DEFAULT_OPEX_BARS,
        **kernel_kwargs,
    ) -> None:
        if likelihood is None:
            likelihood = GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module  = ConstantMean() if mean_type == "constant" else ZeroMean()
        self.covar_module = VannaCharmLayerKernel(opex_bars=opex_bars, **kernel_kwargs)

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def predict(
        self,
        test_x: Tensor,
        full_cov: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Posterior predictive mean + variance or full covariance.

        FIX-6: saves and restores train/eval state so mid-loop diagnostic
        calls do not permanently flip the model to eval mode.
        """
        was_train     = self.training
        lik_was_train = self.likelihood.training
        self.eval(); self.likelihood.eval()
        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = self.likelihood(self(test_x))
            mean = preds.mean.detach()
            out  = (mean, preds.covariance_matrix.detach()) if full_cov \
                   else (mean, preds.variance.detach())
        finally:
            self.train(was_train)               # FIX-6
            self.likelihood.train(lik_was_train)
        return out

    def get_kernel_summary(self) -> Dict[str, float]:
        return self.covar_module.get_param_summary()


# ══════════════════════════════════════════════════════════════════════════════
# §5  VERIFICATION SUITE
# ══════════════════════════════════════════════════════════════════════════════

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def check_psd(kernel: VannaCharmLayerKernel, x: Tensor, tol: float = -1e-6) -> Dict:
    print(_bold("\n[Check 1] Positive semi-definiteness — K_dbar"))
    print("─" * 60)
    with torch.no_grad(), gpytorch.settings.fast_computations(False):
        K = kernel(x, x).to_dense()
    K_np    = K.cpu().double().numpy()
    sym_err = float(np.abs(K_np - K_np.T).max())
    eigvals = np.linalg.eigvalsh(K_np)
    min_eig = float(eigvals.min())
    print(f"  Symmetry max-error : {sym_err:.3e}")
    print(f"  Min eigenvalue     : {min_eig:.6f}")
    passed = (sym_err < 1e-4) and (min_eig >= tol)
    print(f"  Result             : {_bold('✓ PASSED' if passed else '✗ FAILED')}")
    return {"passed": passed, "min_eigenvalue": min_eig, "symmetry_error": sym_err}


def check_channel_decomposition(kernel: VannaCharmLayerKernel, x: Tensor) -> Dict:
    print(_bold("\n[Check 2] Channel decomposition (Frobenius norms)"))
    print("─" * 60)
    with torch.no_grad(), gpytorch.settings.fast_computations(False):
        contribs = kernel.get_channel_contributions(x)
        K_total  = kernel(x, x).to_dense()
    total_frob = float(torch.norm(K_total, p="fro").item())
    result: Dict[str, float] = {}
    for name, K_i in contribs.items():
        frob = float(torch.norm(K_i, p="fro").item())
        rel  = frob / (total_frob + 1e-12)
        result[name] = rel
        print(f"  {name:<10} {frob:>12.4f}  ({rel:.1%})")
    print(f"  {'total':<10} {total_frob:>12.4f}")
    dominated = max(result.values()) > 0.95
    result["balanced"] = not dominated
    print(f"  {_bold('✗ WARNING: one channel dominates' if dominated else '✓ Channels balanced')}")
    return result


def check_smoothness(
    kernel: VannaCharmLayerKernel,
    save_path: str = "vanna_charm_smoothness.png",
) -> None:
    print(_bold("\n[Check 3] Smoothness visualisation"))
    print("─" * 60)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N      = 300
    x_ref  = torch.zeros(1, 2)
    x_vary = torch.linspace(-5.0, 5.0, N)
    specs  = [
        (0, kernel.vanna_kernel,
         "Vanna — Matérn ν=1.5 (C¹)\n(dollar-bar time)", "tab:blue"),
        (1, kernel.charm_kernel,
         "Charm — Periodic×Matérn ν=2.5\n(dollar-bar time)", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("VannaCharm Layer — Kernel Decay by Channel", fontweight="bold")
    with torch.no_grad(), gpytorch.settings.fast_computations(False):
        for ax, (dim, sk, title, color) in zip(axes, specs):
            xp = torch.zeros(N, 2)
            xp[:, dim] = x_vary
            K = sk(x_ref, xp).to_dense().squeeze().cpu().numpy()
            ax.plot(x_vary.numpy(), K, color=color, lw=2.5)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Δ feature value (bar units)")
            ax.set_ylabel("k(x_ref, x)")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#f0f4ff")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def check_feature_pipeline(
    n_trading_days: int = 40,
    dollar_bar_threshold: float = 1_000_000.0,
    seed: int = 7,
) -> Dict:
    print(_bold("\n[Check 4] Dollar-bar feature construction pipeline"))
    print("─" * 60)
    df    = VannaCharmFeatureConstructor.make_synthetic_dataframe(
        n_trading_days=n_trading_days,
        dollar_bar_threshold=dollar_bar_threshold,
        seed=seed,
    )
    ctor      = VannaCharmFeatureConstructor(dollar_bar_threshold=dollar_bar_threshold)
    bar_df, tensor = ctor.transform(df)
    n, d      = tensor.shape
    no_nan    = not tensor.isnan().any().item()
    correct_d = (d == 2)
    print(f"  Input rows     : {len(df):,}")
    print(f"  Dollar bars    : {n}")
    print(f"  Shape          : {tuple(tensor.shape)}")
    print(f"  Means          : {tensor.mean(0).tolist()}")
    print(f"  Stds           : {tensor.std(0).tolist()}")
    print(f"  Bar close times: {bar_df['bar_close_time'].iloc[[0,-1]].tolist()}")
    passed = no_nan and correct_d and (n > 0)
    print(f"  Result         : {_bold('✓ PASSED' if passed else '✗ FAILED')}")
    return {"passed": passed, "n_bars": n, "shape": tuple(tensor.shape)}


def run_all_checks(
    n_points: int = 200,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    save_plot: str = "vanna_charm_smoothness.png",
) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    print("=" * 60)
    print(_bold("VannaCharmLayerKernel — Verification Suite"))
    print("=" * 60)

    kernel = VannaCharmLayerKernel().to(device)
    kernel.eval()
    torch.manual_seed(seed)
    x = torch.randn(n_points, 2, device=device)

    results: Dict = {}
    results["psd"]      = check_psd(kernel, x)
    results["decomp"]   = check_channel_decomposition(kernel, x)
    check_smoothness(kernel, save_path=save_plot)
    results["pipeline"] = check_feature_pipeline()

    all_passed = (
        results["psd"]["passed"]
        and results["decomp"]["balanced"]
        and results["pipeline"]["passed"]
    )
    print("\n" + "=" * 60)
    print(_bold("ALL CHECKS PASSED ✓" if all_passed else "SOME CHECKS FAILED ✗"))
    print("=" * 60 + "\n")
    results["all_passed"] = all_passed
    return results


if __name__ == "__main__":
    import os
    results = run_all_checks(save_plot=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "vanna_charm_smoothness.png"
    ))
    sys.exit(0 if results["all_passed"] else 1)
