"""
Backtesting Framework — RKHS Kernel Model
==========================================
Traders@SMU — Quantitative Strategies Group

Modules:
    walk_forward        Walk-forward engine, kernel combiner, CPCV, CKA gate
    signal_definition   OU signal generation and diagnostics
    metrics             DSR, Sortino, CKA, effective dimensionality, OOS NLL
    hyperparameter_cv   Nested CV, MKL optimizer, purged k-fold
    visualize_backtest  15-figure publication-quality dashboard
"""

from .walk_forward import (
    PurgedWalkForward,
    TwoLevelKernelCombiner,
    MultiResolutionAligner,
    CPCVEvaluator,
    CKARedundancyGate,
    BacktestResult,
    WalkForwardResult,
    exponential_decay_weights,
)
from .signal_definition import (
    OUSignalGenerator,
    estimate_ou_params,
    generate_positions,
    SignalDiagnostics,
)
from .metrics import (
    deflated_sharpe_ratio,
    sortino_ratio,
    centered_kernel_alignment,
    effective_dimensionality,
    oos_negative_log_likelihood,
)
from .hyperparameter_cv import (
    NestedCV,
    MKLOptimizer,
    PurgedKFold,
    ScalerWrapper,
    inner_cv_grid_search,
)
