# Pristine-RKHS — CL Futures RKHS Backtest

Traders@SMU Quantitative Strategies Group. Walk-forward backtest of a
multi-kernel RKHS strategy on CL (crude oil) futures, Jul 2023 – Dec 2024
(471 trading days).

> **Read `POSTMORTEM.md` before quoting any result.** Runs 1–25 were
> produced by a pipeline with four time-alignment bugs plus a lookahead
> leak, and are invalid. The published "Sharpe 1.56" came from an
> out-of-sample target that was identically zero.
>
> **The reported Sharpe carries no information about predictive skill.**
> Across seeds the fixed pipeline scores 0.28 ± 1.18 — while a control
> whose target has been randomly shuffled, so that nothing can predict it
> by construction, scores 1.15 ± 1.08. The two are not separable. Any
> single-seed Sharpe from this pipeline, 1.56 included, is a draw from
> that distribution.
>
> Stage-2 out-of-sample correlation is ≈ −0.014 with real features and
> ≈ −0.015 with pure noise. The OrderFlow kernel gets an ElasticNet weight
> of exactly 0. Do not use Sharpe to choose between configurations here —
> run `validate_pipeline.py` and read `POSTMORTEM.md`.

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

Run 25 configuration (the historical "best"; see `POSTMORTEM.md` before
quoting any number it produces):

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
.venv/bin/python test_feature_alignment.py   # 21 tests, no market data
.venv/bin/python test_results_io.py          #  8 tests, no market data
.venv/bin/python validate_pipeline.py        # falsification suite, ~25 min
```

`validate_pipeline.py` re-runs the pipeline with the information it claims
to use destroyed (noise features, features shuffled in time, shuffled
target) across several seeds, and fails if out-of-sample correlation
survives a shuffled target. Run it after any change to feature
construction, alignment, or the walk-forward engine.

`backtesting/test_pipeline.py` was removed: it imported a module
`lob_kernel` that does not exist in this repo, so the "11 synthetic
end-to-end tests" it was credited with had not run in a long time. The
demo block at the bottom of `backtesting/visualize_backtest.py` imports
the same missing module and is likewise dead.

`figures/` and `reports/` are no longer tracked — every chart in them was
generated from the pre-fix pipeline. Regenerate them with the commands
above after a run.

## Repo map

- `run_backtest.py` — main entry point; full walk-forward pipeline end-to-end
- `feature_alignment.py` — master dollar-bar schedule and the wall-clock
  as-of joins that put every kernel on one time axis, plus the causal
  `expanding_zscore`. Everything time-related goes through here; never
  align features by row number.
- `validate_pipeline.py` — falsification suite (noise / shuffled-time /
  shuffled-target controls across seeds)
- `results_io.py` — results-file discovery that refuses to hand a
  falsification control run to a reporting tool
- `HANDOFF.md` — state of play and the multi-asset project brief
- `TODO.md` — prioritised work list, with the reasoning for each item
- `FALL_2026_TASKS.md` — the Fall 2026 execution list: concrete tasks,
  acceptance criteria, sector owners, and the live-trading build
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
pipeline. It has now been re-run properly: with every prediction feature
replaced by iid noise the pipeline scores 1.18 ± 1.25 against the real
0.28 ± 1.18, so the kernels are not distinguishable from random features
on CL.

All feature normalisation is causal (`expanding_zscore`): row i is
standardised against rows 0..i only. Do not reintroduce a full-sample
`features.mean(axis=0)` / `.std(axis=0)` — that was bug 5.
