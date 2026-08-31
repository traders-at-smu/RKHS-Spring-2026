# Pristine-RKHS — Reproducing Kernel Hilbert Space on CL Futures

**Traders@SMU · Quantitative Strategies · Spring 2026**

Archive of the Quant Strategies sector project for Spring 2026: a
walk-forward backtest of a multi-kernel RKHS strategy on CL (crude oil)
futures, July 2023 – December 2024, built on 36.7M Databento trade ticks
plus LOB snapshots and options statistics.

Includes an August 2026 remediation pass. Read the next section before
using any number from this repository.

---

## Read this first

**The headline result this project reported — Sharpe 1.56 — was not real,
and neither is any number in the Runs 1–25 history.**

Five bugs were found and fixed after the semester ended. Four were time
alignment: features from different months were being lined up by row
number, and the out-of-sample target was identically zero, so the model
was scoring random drift as signal. The fifth leaked the future into every
training bar through full-sample normalisation.

Fixing them did not rescue the result. It exposed a second, larger
problem: **the performance metric itself does not work.** We tested it by
shuffling the answer key — permuting the target so nothing could predict
it by construction — and that scored *better* than the real strategy
(1.15 vs 0.28 across three random seeds). Replacing every feature with
random numbers also scored better (1.18).

So the honest state of this project is: the pipeline is correct, and the
strategy has no demonstrated edge on CL. Those are separate claims and
this repository is careful to keep them apart.

| Document | What's in it |
|----------|--------------|
| [`POSTMORTEM.md`](POSTMORTEM.md) | All five bugs, how each was found, and exactly what they invalidated |
| [`HANDOFF.md`](HANDOFF.md) | State of play, the invariants that must not be broken, the multi-asset brief |
| [`TODO.md`](TODO.md) | Prioritised work list in plain language, with the reasoning for each item |
| [`CLAUDE.md`](CLAUDE.md) | Setup, data layout, architecture, repo map |

---

## Quick start

Requires Python 3.14 (3.12+ should work). `data/` is gitignored — it is
multi-GB and shared separately. Ask Rebecca.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# unit tests — no market data needed, run in seconds
.venv/bin/python test_feature_alignment.py
.venv/bin/python test_results_io.py

# the full backtest (Run 25 configuration)
ARCH_VERSION=v2 SIGNAL_MODE=kalman KALMAN_Q=aggressive TUNE_KERNELS=0 \
WARMUP_DAYS=60 VOL_SIZING=1 ENTRY_CHOPPY=1.50 EXIT_TRENDING=0.10 \
EXIT_CHOPPY=0.35 .venv/bin/python run_backtest.py

# falsification suite — needs data, ~25 minutes
.venv/bin/python validate_pipeline.py
```

Run `validate_pipeline.py` after any change to feature construction,
alignment, or the walk-forward engine. It re-runs the pipeline with the
information it claims to use destroyed and checks that the results
collapse. It is the thing that would have caught the original bugs.

---

## Layout

```
run_backtest.py           Main entry point — full walk-forward pipeline
feature_alignment.py      Master dollar-bar schedule, wall-clock as-of joins,
                          causal expanding_zscore. All time handling goes here.
validate_pipeline.py      Falsification suite (noise / shuffled-time /
                          shuffled-target controls, across seeds)
results_io.py             Results discovery that refuses to hand a control
                          run to a reporting tool
data_loader.py            Data → kernel-input bridge
kernels/                  10 RKHS kernel layers
backtesting/              Purged walk-forward engine, Kalman/OU signal
                          extraction, nested CV, metrics
analytics/                Bootstrap MC, stress tests, trade simulator
reporting/                Chart theme
```

`results_table.txt`, `next_steps.txt`, `run_configs.txt` and `pipeline.txt`
hold the original semester's run history. **Every number in them is
invalid** and each carries a banner saying so. They are kept because the
reasoning is worth reading even where the results are wrong.

---

## Deviations from `AI.md`

This is Spring 2026 work, written before the org's `AI.md` conventions
existed. It is archived as-is for provenance rather than retrofitted.
Where it diverges, and why:

| Convention | This repo | Note |
|------------|-----------|------|
| Approved packages only | Uses `scikit-learn`, `matplotlib`, `pyarrow`, `tqdm`, `torch`, `gpytorch`, `transformers`, `yfinance` | None are on `approved-packages.md`. `scikit-learn` is load-bearing (ElasticNet, PCA); `pyarrow` reads the Parquet data. Would need dplynn's approval to carry forward. |
| UV + `pyproject.toml` | `venv` + `requirements.txt` | `requirements.txt` is a `pip freeze`, so it also pins packages nothing imports. |
| Settings in `config.json` | Environment variables | Every knob is an env var; `CLAUDE.md` and `HANDOFF.md` document them. |
| No docstrings | Docstrings throughout the new modules | Deliberate: the docstrings in `feature_alignment.py` and `results_io.py` explain which bug each function exists to prevent. That context is the point. |
| TDD, ≥1 test per feature | 29 tests, covering the alignment and results-selection modules only | The pipeline itself is largely untested. See `TODO.md`. |
| Feature branch + PR, never commit to main | Followed for the remediation work | Five commits, branched and reviewed before merge. |

Anyone continuing this work in a new repo should start from the
conventions, not from this layout.
