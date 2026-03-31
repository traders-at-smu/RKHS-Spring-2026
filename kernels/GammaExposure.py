"""
gamma_exposure_layer.py
========================
Gamma Exposure RKHS Layer — calendar-time feature space.

Captures the structural, strike-anchored hedging pressure of market makers
measured in CALENDAR TIME (daily bars):

    - Gamma squeezes    (GEX)
    - Open Interest pinning

Why calendar time?
──────────────────
GEX and OI are end-of-day inventory concepts.  Open interest changes once
per day at settlement.  The strike grid is fixed to listed expiries, which
expire on calendar dates (monthly/weekly OPEX).  The relevant distance
metric between two observations is calendar days — not how much premium
traded between them.  A Monday and the following Friday are always 4 days
apart regardless of volume.

Kernel:
    K_cal(x, x') = k_Γ(x₀, x₀') + k_OI(x₁, x₁')

    k_Γ  : ScaleKernel(MaternKernel(ν=0.5)) on dim 0  — GEX
    k_OI : ScaleKernel(MaternKernel(ν=1.5)) on dim 1  — Put/Call OI ratio

Feature vector (2-D):
    x_cal = [Net GEX_t,  Put/Call OI Ratio_t]

    Net GEX_t   = Σ_K [ Γ_call·OI_call − Γ_put·OI_put ]
    PC OI Ratio = Σ OI_put / Σ OI_call

All channels z-score normalised using a causal rolling window.

Fixes carried from v2
─────────────────────
  FIX-1  forward() calls sub-kernel via __call__ (no __call__ override)
  FIX-3  _rolling_zscore guard len(window) < 2  (single-element std=0 blowup)
  FIX-5  Vectorized groupby[col].sum() — no lambda apply
  FIX-6  predict() saves/restores train/eval state

Authors: Refactored from dual-space design (March 2026)
Requirements: gpytorch >= 1.11, torch >= 2.0, pandas, numpy, matplotlib
"""

from __future__ import annotations

import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior
from torch import Tensor


# ══════════════════════════════════════════════════════════════════════════════
# §0  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS: frozenset = frozenset({
    "timestamp", "strike", "type",
    "gamma", "open_interest",
})

# Feature vector layout
#   dim 0 : net_gex
#   dim 1 : pc_oi_ratio


# ══════════════════════════════════════════════════════════════════════════════
# §1  SUB-KERNELS
# ══════════════════════════════════════════════════════════════════════════════

class GammaExposureKernel(ScaleKernel):
    """k_Γ : ScaleKernel(MaternKernel(ν=0.5)) on active_dims=[0].

    Matérn-1/2 (Ornstein-Uhlenbeck) is C⁰ — continuous but nowhere
    differentiable — matching the sharp, angular regime transitions of
    gamma squeezes at discrete strike clusters.

    GammaPrior(α=2, β=10) biases the length-scale toward short values
    (mode = (α−1)/β = 0.1) in daily-normalised feature space.

    k(r) = σ² exp(−r / ℓ)
    """

    def __init__(
        self,
        lengthscale_prior_concentration: float = 2.0,
        lengthscale_prior_rate: float = 10.0,
        **kwargs,
    ) -> None:
        base_kernel = MaternKernel(
            nu=0.5,
            active_dims=torch.tensor([0]),
            lengthscale_prior=GammaPrior(
                concentration=lengthscale_prior_concentration,
                rate=lengthscale_prior_rate,
            ),
        )
        super().__init__(base_kernel=base_kernel, **kwargs)

    @property
    def matern_kernel(self) -> MaternKernel:
        return self.base_kernel

    @property
    def lengthscale(self) -> Tensor:
        return self.base_kernel.lengthscale


class OpenInterestKernel(ScaleKernel):
    """k_OI : ScaleKernel(MaternKernel(ν=1.5)) on active_dims=[1].

    Put/Call OI walls create structural support/resistance: dealers pin
    prices toward high-OI strikes with a C¹ pressure field.  OI changes
    are end-of-day events, making this a calendar-time feature.

    k(r) = σ² (1 + √3 r/ℓ) exp(−√3 r/ℓ)
    """

    def __init__(self, **kwargs) -> None:
        base_kernel = MaternKernel(nu=1.5, active_dims=torch.tensor([1]))
        super().__init__(base_kernel=base_kernel, **kwargs)

    @property
    def matern_kernel(self) -> MaternKernel:
        return self.base_kernel

    @property
    def lengthscale(self) -> Tensor:
        return self.base_kernel.lengthscale


# ══════════════════════════════════════════════════════════════════════════════
# §2  COMPOSITE LAYER KERNEL
# ══════════════════════════════════════════════════════════════════════════════

class GammaExposureLayerKernel(Kernel):
    """K_cal = k_Γ(x₀,x₀') + k_OI(x₁,x₁')

    Additive composition over two calendar-time channels.
    Input must be 2-D: x = [net_gex, pc_oi_ratio].

    FIX-1: forward() calls self._additive_kernel(x1, x2, ...) via __call__,
    not .forward(), preserving GPyTorch's lazy-eval pipeline and caching.
    No __call__ override.
    """

    def __init__(
        self,
        gamma_ls_concentration: float = 2.0,
        gamma_ls_rate: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gamma_kernel = GammaExposureKernel(
            lengthscale_prior_concentration=gamma_ls_concentration,
            lengthscale_prior_rate=gamma_ls_rate,
        )
        self.oi_kernel = OpenInterestKernel()
        # AdditiveKernel via GPyTorch '+' operator
        self._additive_kernel = self.gamma_kernel + self.oi_kernel

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
            "gamma": self.gamma_kernel(x1, x2).to_dense(),
            "oi":    self.oi_kernel(x1, x2).to_dense(),
        }

    def get_param_summary(self) -> Dict[str, float]:
        def _s(t: Tensor) -> float:
            return t.squeeze().item()
        return {
            "gamma_outputscale": _s(self.gamma_kernel.outputscale),
            "gamma_lengthscale": _s(self.gamma_kernel.matern_kernel.lengthscale),
            "oi_outputscale":    _s(self.oi_kernel.outputscale),
            "oi_lengthscale":    _s(self.oi_kernel.matern_kernel.lengthscale),
        }


# ══════════════════════════════════════════════════════════════════════════════
# §3  FEATURE CONSTRUCTOR
# ══════════════════════════════════════════════════════════════════════════════

class GammaExposureFeatureConstructor:
    """Daily options chain DataFrame → normalised 2-D [net_gex, pc_oi_ratio].

    Aggregation per calendar day:
        Net GEX_t   = Σ_K [ Γ_call·OI_call − Γ_put·OI_put ]
        PC OI Ratio = Σ OI_put / Σ OI_call

    Both channels are z-score normalised using a causal rolling window.

    FIX-3: rolling z-score guard is len(window) < 2. A single-element
    window produces std=0 → z ≈ Δ/ε ≈ 1e8, causing catastrophic feature
    values on the second timestep. The first two observations are zeroed
    as statistically insufficient — negligible cost versus typical history.

    FIX-5: vectorised groupby[col].sum() throughout — no lambda apply.

    Args:
        rolling_window: Calendar days for rolling z-score look-back (≥ 2).
        zscore_eps:     Stability constant added to rolling std.
        device:         Output tensor device.
        dtype:          Output tensor dtype.
    """

    CHANNEL_NAMES = ("net_gex", "pc_oi_ratio")

    def __init__(
        self,
        rolling_window: int = 63,
        zscore_eps: float = 1e-8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if rolling_window < 2:
            raise ValueError("rolling_window must be >= 2")
        self.rolling_window = rolling_window
        self.zscore_eps     = zscore_eps
        self.device         = device or torch.device("cpu")
        self.dtype          = dtype
        self._raw:        Optional[pd.DataFrame] = None
        self._normalised: Optional[pd.DataFrame] = None

    def transform(self, options_df: pd.DataFrame) -> Tensor:
        """Convert daily options chain to normalised 2-D tensor.

        Args:
            options_df: DataFrame with columns in REQUIRED_COLUMNS.
                Multiple rows per day (one per strike/type).

        Returns:
            Tensor of shape (n_days, 2), z-score normalised, no look-ahead.
        """
        self._validate(options_df)
        raw = self._aggregate(options_df)
        self._raw = raw.copy()
        normed = self._rolling_zscore(raw)
        self._normalised = normed.copy()
        values = normed[list(self.CHANNEL_NAMES)].to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=self.dtype, device=self.device)

    def get_raw(self) -> Optional[pd.DataFrame]:
        """Pre-normalisation aggregate features (after transform())."""
        return self._raw

    def get_normalised(self) -> Optional[pd.DataFrame]:
        """Post-normalisation features as DataFrame (after transform())."""
        return self._normalised

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["type"]     = df["type"].str.lower().str.strip()
        df["_date"]    = pd.to_datetime(df["timestamp"]).dt.normalize()
        df["_w_gamma"] = df["gamma"] * df["open_interest"]

        calls = df[df["type"] == "call"]
        puts  = df[df["type"] == "put"]

        # FIX-5: vectorised groupby.sum()
        c_gex = calls.groupby("_date")["_w_gamma"].sum()
        p_gex = puts.groupby("_date")["_w_gamma"].sum()
        c_oi  = calls.groupby("_date")["open_interest"].sum()
        p_oi  = puts.groupby("_date")["open_interest"].sum()

        all_dates = c_gex.index.union(p_gex.index)
        c_gex = c_gex.reindex(all_dates, fill_value=0.0)
        p_gex = p_gex.reindex(all_dates, fill_value=0.0)
        c_oi  = c_oi.reindex(all_dates, fill_value=0.0)
        p_oi  = p_oi.reindex(all_dates, fill_value=0.0)

        result = pd.DataFrame({
            "net_gex":     c_gex - p_gex,
            "pc_oi_ratio": p_oi / c_oi.clip(lower=self.zscore_eps),
        }).fillna(0.0)
        result.index.name = "date"
        result.sort_index(inplace=True)
        return result

    # ── Rolling z-score (causal, no look-ahead) ───────────────────────────────

    def _rolling_zscore(self, raw: pd.DataFrame) -> pd.DataFrame:
        n      = len(raw)
        normed = raw.copy()
        for col in self.CHANNEL_NAMES:
            series = raw[col].to_numpy(dtype=np.float64)
            z = np.empty_like(series)
            for t in range(n):
                window = series[max(0, t - self.rolling_window):t]
                if len(window) < 2:   # FIX-3: single-element std=0 blowup
                    z[t] = 0.0
                else:
                    z[t] = (series[t] - window.mean()) / (
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

    # ── Synthetic data ────────────────────────────────────────────────────────

    @staticmethod
    def make_synthetic_dataframe(
        n_days: int = 126,
        n_strikes: int = 10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic daily per-strike options data for testing."""
        rng     = np.random.default_rng(seed)
        dates   = pd.date_range("2024-01-02", periods=n_days, freq="B")
        strikes = np.linspace(100, 200, n_strikes)
        rows = []
        for ts in dates:
            for strike in strikes:
                for opt_type in ("call", "put"):
                    rows.append({
                        "timestamp":     ts,
                        "strike":        strike,
                        "type":          opt_type,
                        "gamma":         rng.uniform(0.001, 0.05),
                        "open_interest": rng.uniform(100, 5000),
                    })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# §4  EXACT GP MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GammaExposureGP(ExactGP):
    """ExactGP for the calendar-time GammaExposureLayerKernel.

    Args:
        train_x:     (n, 2) — [net_gex, pc_oi_ratio], z-score normalised.
        train_y:     (n,)   — daily price return or target.
        likelihood:  GaussianLikelihood; created if not provided.
        mean_type:   "constant" (default) or "zero".
        **kernel_kwargs: Forwarded to GammaExposureLayerKernel.
    """

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: Optional[GaussianLikelihood] = None,
        mean_type: str = "constant",
        **kernel_kwargs,
    ) -> None:
        if likelihood is None:
            likelihood = GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module  = ConstantMean() if mean_type == "constant" else ZeroMean()
        self.covar_module = GammaExposureLayerKernel(**kernel_kwargs)

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
        was_train      = self.training
        lik_was_train  = self.likelihood.training
        self.eval(); self.likelihood.eval()
        try:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = self.likelihood(self(test_x))
            mean = preds.mean.detach()
            out  = (mean, preds.covariance_matrix.detach()) if full_cov \
                   else (mean, preds.variance.detach())
        finally:
            self.train(was_train)              # FIX-6
            self.likelihood.train(lik_was_train)
        return out

    def get_kernel_summary(self) -> Dict[str, float]:
        return self.covar_module.get_param_summary()


# ══════════════════════════════════════════════════════════════════════════════
# §5  VERIFICATION SUITE
# ══════════════════════════════════════════════════════════════════════════════

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def check_psd(kernel: GammaExposureLayerKernel, x: Tensor, tol: float = -1e-6) -> Dict:
    print(_bold("\n[Check 1] Positive semi-definiteness — K_cal"))
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


def check_channel_decomposition(kernel: GammaExposureLayerKernel, x: Tensor) -> Dict:
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
    kernel: GammaExposureLayerKernel,
    save_path: str = "gamma_exposure_smoothness.png",
) -> None:
    print(_bold("\n[Check 3] Smoothness visualisation"))
    print("─" * 60)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N      = 300
    x_ref  = torch.zeros(1, 2)
    x_vary = torch.linspace(-5.0, 5.0, N)
    specs  = [
        (0, kernel.gamma_kernel, "GEX — Matérn ν=0.5 (C⁰)\n(calendar time)", "tab:red"),
        (1, kernel.oi_kernel,    "OI Ratio — Matérn ν=1.5 (C¹)\n(calendar time)", "tab:orange"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Gamma Exposure Layer — Kernel Decay by Channel", fontweight="bold")
    with torch.no_grad(), gpytorch.settings.fast_computations(False):
        for ax, (dim, sk, title, color) in zip(axes, specs):
            xp = torch.zeros(N, 2)
            xp[:, dim] = x_vary
            K = sk(x_ref, xp).to_dense().squeeze().cpu().numpy()
            ax.plot(x_vary.numpy(), K, color=color, lw=2.5)
            ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Δ feature value")
            ax.set_ylabel("k(x_ref, x)")
            ax.grid(True, alpha=0.3)
            ax.set_facecolor("#fff8f0")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def check_feature_pipeline(n_days: int = 80, seed: int = 7) -> Dict:
    print(_bold("\n[Check 4] Feature construction pipeline"))
    print("─" * 60)
    df         = GammaExposureFeatureConstructor.make_synthetic_dataframe(n_days=n_days, seed=seed)
    ctor       = GammaExposureFeatureConstructor()
    tensor     = ctor.transform(df)
    n, d       = tensor.shape
    no_nan     = not tensor.isnan().any().item()
    correct_d  = (d == 2)
    print(f"  Input rows  : {len(df):,}")
    print(f"  Days        : {n}")
    print(f"  Shape       : {tuple(tensor.shape)}")
    print(f"  Means       : {tensor.mean(0).tolist()}")
    print(f"  Stds        : {tensor.std(0).tolist()}")
    passed = no_nan and correct_d and (n > 0)
    print(f"  Result      : {_bold('✓ PASSED' if passed else '✗ FAILED')}")
    return {"passed": passed, "n_days": n, "shape": tuple(tensor.shape)}


def run_all_checks(
    n_points: int = 200,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    save_plot: str = "gamma_exposure_smoothness.png",
) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    print("=" * 60)
    print(_bold("GammaExposureLayerKernel — Verification Suite"))
    print("=" * 60)

    kernel = GammaExposureLayerKernel().to(device)
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
        os.path.dirname(os.path.abspath(__file__)), "gamma_exposure_smoothness.png"
    ))
    sys.exit(0 if results["all_passed"] else 1)
