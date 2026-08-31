# Pristine-RKHS — CL Futures RKHS Backtest

Traders@SMU Quantitative Strategies Group. Walk-forward backtest of a
multi-kernel RKHS strategy on CL (crude oil) futures, Jul 2023 – Dec 2024
(471 trading days).

> **Read `POSTMORTEM.md` before quoting any result.** Runs 1–25 were
> produced by a pipeline with four time-alignment bugs and are invalid.
> The published "Sharpe 1.56" came from an out-of-sample target that was
> identically zero. On the fixed pipeline the same Run 25 config gives
> Sharpe 1.10 / Sortino 1.63 / Max DD 2.6% — but with OOS R² of −0.0011,
> near-zero kernel weights and only 40 active days, that number is not
> evidence of a directional edge either.

## Setup

Requires Python 3.14 (3.12+ should work).

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Data (required — NOT in git)

The `data/` directory is gitignored (multi-GB) and shared via Google Drive.
Ask Rebecca for access, then place it at the repo root so it looks like:

```
data/
├── trades_CL_full.parquet            533 MB  trade ticks
├── ohlcv1m_CL_full.parquet             7 MB  1-min bars
├── definition_CL_FUT_full.parquet     14 MB  contract defs
├── statistics_LO_OPT_full.parquet    2.2 GB  options stats
├── greeks_daily.parquet
├── sentiment_daily.parquet
├── vanna_charm_bars.parquet
├── raw_articles_gdelt.pkl
├── LOB/                              precomputed LOB features/snapshots/gram matrices
└── external/                         VIX, SPY, XLE, put/call, event calendar
```

`data/external/` can be regenerated with `python fetch_external_data.py`
if missing.

## Running the backtest

Everything is configured via environment variables (no CLI args).
`run_configs.txt` is the authoritative list of every run configuration and
its results — start there. `pipeline.txt` documents the architecture and
file layout.

Best configuration (Run 25 — v2 hybrid; Sharpe 1.10 post-fix, see
`POSTMORTEM.md`):

```bash
ARCH_VERSION=v2 SIGNAL_MODE=kalman KALMAN_Q=aggressive TUNE_KERNELS=0 WARMUP_DAYS=60 VOL_SIZING=1 ENTRY_CHOPPY=1.50 EXIT_TRENDING=0.10 EXIT_CHOPPY=0.35 .venv/bin/python run_backtest.py
```

Plain `python run_backtest.py` runs the v1 default — note the defaults
(`WARMUP_DAYS=120`, `TUNE_KERNELS=1`, `POSITIVE_WEIGHTS=1`) differ from the
original Run 13/14 configs, so always set env vars explicitly when
reproducing a numbered run.

Results are saved to `backtest_results_<config>.npz` (gitignored,
regenerable). Post-processing:

```bash
python generate_charts.py      # Midnight Ocean charts → figures/
python stress_test_run.py      # 6 stress scenarios (COVID, GFC, Volmageddon, ...)
python generate_report.py      # full text report → reports/
```

Tests:

```bash
.venv/bin/python test_feature_alignment.py   # 15 tests, no market data
```

`backtesting/test_pipeline.py` is currently broken — it imports a module
`lob_kernel` that does not exist in this repo, and has been failing at
import since before the alignment fix.

## Repo map

- `run_backtest.py` — main entry point; full walk-forward pipeline end-to-end
- `feature_alignment.py` — master dollar-bar schedule and the wall-clock
  as-of joins that put every kernel on one time axis. Everything time-
  related goes through here; never align features by row number.
- `data_loader.py` — data → kernel-input bridge
- `kernels/` — 10 RKHS kernel layers. Active: LOB, VPIN, Kyle's Lambda,
  Hawkes (fast); VRP, MacroMotion (slow). Gates: momentum_gate,
  EventProximity. Shelved: VannaCharm, GammaExposure (need real Greeks).
  Blocked: SentimentAnalysis (needs news API).
- `backtesting/` — PurgedWalkForward engine, Kalman/OU signal extraction,
  nested CV, metrics (DSR, Sortino, CKA)
- `analytics/` — bootstrap MC, stress tests, trade simulator, Kelly allocation
- `reporting/` — Midnight Ocean chart theme (navy/teal/coral/gold)
- `results_table.txt` — comparison table across all runs

## Architecture (one paragraph)

Two-level kernel combiner: `K_total = K_fast + β · K_fast · K_slow_gated`.
Stage 1 fits per-kernel KRR predictions; Stage 2 combines them with a
walk-forward ElasticNet. The combined prediction is Kalman-filtered into an
s-score with regime-conditional entry/exit thresholds, gated by momentum
regime and event proximity. Evaluation is daily, 8 expanding walk-forward
folds with a 100-bar embargo, 1-tick ($10) transaction cost per trade.

All fast kernels are resampled onto a single master $1M dollar-bar
schedule by wall clock (`feature_alignment.py`), and all slow kernels onto
a single daily date axis, before anything is combined. The forward-return
target comes from dollar-bar closes and is checked for degeneracy on every
fold. The old claim that the kernels act as a noise filter — "null test
gives Sharpe -0.76 vs +1.55 with kernels" — was measured on the buggy
pipeline and has not been re-run; the Stage-2 OOS R² on the fixed pipeline
is −0.0011, so treat the kernels as unproven on returns.
