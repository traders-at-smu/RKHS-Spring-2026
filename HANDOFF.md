# Handoff — Pristine-RKHS

For the Traders@SMU Quantitative Strategies Group taking this over.

Read `POSTMORTEM.md` first. The short version: the published Sharpe 1.56
was an artifact of five bugs, all now fixed. What you are inheriting is a
pipeline whose *mechanics* are verified, running a strategy whose *edge*
is not established. Those are different claims and it matters that you
keep them apart.

## State of play

**Working and checked.** Every kernel now sits on one time axis, the
out-of-sample target is real, feature normalisation is causal, and there
is a falsification suite that tries to break the harness on demand. The
bugs that made the old numbers meaningless cannot silently return —
positional trimming raises, a short daily grid raises, and a degenerate
target raises, per fold.

**Not established.** That the strategy predicts CL direction — and, more
awkwardly, that the metric everyone has been quoting means anything.
Across seeds the fixed pipeline scores Sharpe 0.28 ± 1.18. A control run
with the target randomly shuffled — unpredictable by construction —
scores 1.15 ± 1.08. Those are not separable, so the Sharpe is not
measuring skill; it is measuring which random Fourier features got drawn.

The direct measurements say the same thing more quietly: Stage-2
out-of-sample correlation is ≈ −0.014 with real features and ≈ −0.015
with pure noise, R² is negative in both cases, the OrderFlow kernel — the
whole consolidated LOB + VPIN + Kyle + Hawkes stack — gets an ElasticNet
weight of exactly zero, and the P&L comes from ~29 active days out of 172.

Practical consequence: **do not tune against Sharpe.** Runs 1–25 did, and
that alone would have made the ranking meaningless even without the five
bugs.

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`data/` is gitignored and shared separately — ask Rebecca. See `CLAUDE.md`
for the expected layout.

```bash
# the Run 25 configuration
ARCH_VERSION=v2 SIGNAL_MODE=kalman KALMAN_Q=aggressive TUNE_KERNELS=0 \
WARMUP_DAYS=60 VOL_SIZING=1 ENTRY_CHOPPY=1.50 EXIT_TRENDING=0.10 \
EXIT_CHOPPY=0.35 .venv/bin/python run_backtest.py

# unit tests — no market data needed, run in seconds
.venv/bin/python test_feature_alignment.py
.venv/bin/python test_results_io.py

# falsification suite — needs data, takes ~25 minutes
.venv/bin/python validate_pipeline.py
```

Run `validate_pipeline.py` after any change to feature construction,
alignment, or the walk-forward engine.

It gates on one condition only: out-of-sample correlation must not survive
a shuffled target, which would mean the harness leaks. Everything else it
prints is `[INFO]` — a suite that fails whenever the strategy is bad is a
suite people learn to ignore. Read the `[INFO]` lines anyway; they are
currently where the interesting news is.

### Controls worth knowing

| Env var | Purpose |
|---------|---------|
| `SEED=n` | Offsets every RNG (RFF draws, PCA, CKA subsample). A result that only holds at one seed is a result about one draw. |
| `NULL_MODE=noise` | Replaces all prediction features with iid noise — the "no kernels" null. |
| `NULL_MODE=shuffle_time` | Permutes feature rows in time. Catches alignment bugs: code that ignores time cannot tell this from the real thing. |
| `NULL_MODE=shuffle_target` | Permutes the target. Any surviving correlation is leakage. |
| `POSITION_GATE=0` | Disables the VPIN/Hawkes position-sizing gate. |
| `VOL_SIZING=0` | Disables vol-targeted sizing. |

Together the last two strip the machinery that generates P&L independently
of prediction — useful when you want to know what the kernels alone do.

## Invariants — please do not break these

1. **Never align features by row number.** Everything time-related goes
   through `feature_alignment.py`. `arr[:n]` to make two kernels the same
   length is the original bug; `_assert_on_master` now raises on it.
2. **Never normalise with full-sample statistics.** `features.mean(axis=0)`
   over the whole array leaks the future into every training bar. Use
   `expanding_zscore`.
3. **Any new data source joins on the date, not the row.** VIX/SPY/XLE
   cover 378 equity sessions where CL trades 468 days; joining those
   positionally is what silently disabled the VRP kernel.
4. **A new fast kernel must return timestamps** and be registered in the
   `_named_fast` dict so it is resampled onto the master schedule.
5. **Tools that auto-discover a results file go through `results_io`.**
   `validate_pipeline.py` leaves control runs in the working directory
   whose features are noise or whose target is shuffled; anything that
   grabs "the newest .npz" will otherwise report on one of them.

## The semester project: multi-asset

This is the direction Rebecca wants picked up, and it is the main open
question.

RKHS methods of this kind are built to compare and rank *across* many
instruments — the kernel measures similarity between market states, which
is most useful when there are many states to rank. Running the whole
apparatus on one contract asks it to do the one thing it is worst at:
predict the sign of a single noisy series. The expectation is that CL-only
performance stays poor no matter how the thresholds are tuned, and that
the interesting signal only appears once the kernels can express
interactions across a book of futures.

The April 2026 bug report reached the same conclusion independently, and
sketched the architecture: a direction layer that is *not* RKHS (plain
trend-following per market, which is a commodity), and a sizing layer that
*is* — the kernels rank markets by predicted vol regime and microstructure
quality, and allocation follows that ranking. The per-market edge is
small; the thesis is that it compounds across uncorrelated markets.

**What that needs:**

- **Tick data for more contracts.** The kernels need trades, LOB snapshots
  and options statistics to compute VPIN, Kyle's Lambda, Hawkes, LOB depth
  and VannaCharm. Right now that exists only for CL (Databento GLBX.MDP3).
  The bug report prioritises ES and NQ. Daily bars are too coarse — the
  microstructure kernels have nothing to compute from them.
- **A per-asset pipeline boundary.** `run_backtest.py` is currently one
  long `main()` with CL assumptions threaded through it. Extracting a
  "build features for one symbol" function is the first refactor, and the
  master-schedule design already supports it: each symbol gets its own
  dollar-bar schedule, and cross-asset comparison happens on the daily
  axis where the calendars actually line up.
- **A cross-sectional target.** Forward return of one contract is the
  wrong thing to predict. Relative ranking across the book is the natural
  RKHS target and is what the vol-regime sizing thesis actually needs.
- **Keep the falsification suite green throughout.** Multi-asset work
  multiplies the number of places a time axis can go wrong, and every one
  of the five bugs here was an alignment bug that nothing detected.

**Before any of that, one smaller job is worth doing:** find a measure of
the kernels that does not route through ~29 days of position P&L. The
current Sharpe has been shown not to separate the real pipeline from a
shuffled-target control, so it cannot referee whether a change helped —
which makes it useless as an instrument for a semester of changes.
Per-kernel out-of-sample information coefficient against forward returns
and forward vol, reported per fold with seed dispersion, would be a start;
`diagnostic_predictiveness.py` is a rough first pass and reads the saved
`.npz`. Whatever you pick, validate it the same way: it must distinguish
the real pipeline from `NULL_MODE=shuffle_target`. If it cannot, it is not
a metric.

## Known dead code

- The demo block at the bottom of `backtesting/visualize_backtest.py`
  imports `lob_kernel`, which does not exist in this repo. It has not run
  in a long time. `backtesting/test_pipeline.py` had the same import and
  was removed.
- `figures/` and `reports/` are no longer tracked; every chart in them
  came from the pre-fix pipeline. Regenerate with `generate_charts.py` and
  `generate_report.py` after a run.
