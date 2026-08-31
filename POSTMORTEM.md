# Post-mortem: the bugs behind Sharpe 1.56

**Status: every result numbered Run 1–25 was produced by a pipeline with
time-alignment bugs and should not be quoted.** The bugs are fixed as of
this commit. This document records what was wrong, how it was found, and
what the corrected pipeline actually produces.

## Summary

Five bugs compounded. Two of them (1 and 2) were identified in the April
2026 bug report; bugs 3 and 4 surfaced only once the first two were fixed,
because the original code structurally could not reach the data that
exposed them. Bug 5 is a lookahead leak, separate from the alignment work.

| # | Bug | Effect |
|---|-----|--------|
| 1 | Fast kernels aligned by row number, not time | Order-flow PCA mixed features from different months |
| 2 | Daily grid ended before the bars did | OOS forward-return target identically zero |
| 3 | 451 non-finite rows in the LOB features | Never seen, because only 6 days of LOB were ever read |
| 4 | Slow kernels stacked by row position | VRP day 30 aligned with Macro day 60; VIX/SPY/XLE joined positionally |
| 5 | Features z-scored over the whole sample | Every training bar saw the future distribution |

## Bug 1 — fast features aligned by row number

Every fast kernel is sampled on its own clock:

| Kernel | Rows | Sampling |
|--------|------|----------|
| LOB | 372,893 | one per LOB snapshot, 14 to a few thousand per day |
| Hawkes | 157,824 | one per fixed 300-second window |
| VPIN | 5,546 | one per $1M dollar bar |
| VannaCharm | 5,598 | one per $1M dollar bar |
| Kyle | 4,980 | one per fixed trade-count bin |

The pipeline aligned them by truncating to the shortest:

```python
n_fast = min(a.shape[0] for a in fast_arrays)
fast_trimmed = {name: arr[:n_fast] for name, arr in zip(names, fast_arrays)}
```

At `n_fast = 4,980`, LOB row 4,000 was July 2023 while Kyle row 4,000 was
late 2024. The v2 OrderFlow PCA was concatenating features from completely
different time periods and reporting 97% explained variance on the result.

**Fix.** `feature_alignment.py` builds one master $1M dollar-bar schedule
from the 36.7M-trade tape (5,598 bars) and resamples every fast kernel onto
it with a backward as-of join — the last observation at or before each bar
close, so nothing looks forward. Every builder now returns wall-clock
timestamps alongside its features. Measured staleness after the fix:

```
LOB           372,893 rows → 5,598 aligned | median    14.4s
VPIN            5,546 rows → 5,547 aligned | median  1319.8s
Kyle            4,980 rows → 5,577 aligned | median  1298.6s
Hawkes        157,824 rows → 5,598 aligned | median   141.2s
VannaCharm      5,598 rows → 5,598 aligned | median     0.0s
```

Positional trimming is now a hard error (`_assert_on_master`), not a
silent repair.

## Bug 2 — degenerate out-of-sample target

The bar→day map was built from a synthetic grid:

```python
bars_per_day = n_fast // len(lob_dates_used)
bar_timestamps = np.arange(n_fast, dtype=float)
daily_timestamps = np.arange(0, n_fast, bars_per_day, dtype=float)[:n_slow]
```

The `[:n_slow]` truncation made the daily grid span only
`n_slow * bars_per_day` bars. Every bar past that point clipped to the
final day, so across the whole OOS window `bar_prices` was constant and
the forward-return target was identically zero. The ElasticNet collapsed
to an intercept, the Kalman filter read fold-boundary intercept shifts as
"regime transitions", and those happened to line up with CL's trend often
enough to produce a Sharpe of 1.56. Nothing in the pipeline complained.

**Fix.** Real timestamps on both axes, the target computed from actual
dollar-bar closes rather than daily closes mapped down onto bars, and a
guard (`check_target_not_degenerate`) that fails the run if the target is
constant or mostly zeros — checked on the full sample and on every fold's
test window:

```
fold 1 OOS: std=4.708e-03, zeros=0.9%, n=340 — OK
...
fold 8 OOS: std=5.195e-03, zeros=1.2%, n=340 — OK
```

The same `np.linspace` fake mapping also appeared in the MKL slow→fast
expansion, the per-fold slow-kernel expansion, the warm-up filter, and —
most consequentially — the daily collapse used for final scoring, where it
spread the OOS bars evenly across *all* days so out-of-sample positions
were scored against in-sample dates. All now use the real aligner.

## Bug 3 — unreachable NaNs in the LOB features

The LOB feature files contain 451 non-finite rows spread over 19 days,
all after 2023-07-20. Positional trimming only ever read the first ~6 days
of LOB, so they were never touched. Once alignment covered the full range
they crashed the PCA. Non-finite rows are not observations, so the as-of
join now skips them and carries the previous clean row forward.

## Bug 4 — the slow layer was misaligned against itself

`build_vrp_features` trims 30 warm-up days, `build_macro_features` trims
60, and the sentiment file carries its own 503-row date axis against CL's
468 trading days. All three were stacked with `[:n_slow]`, lining up VRP
day 30 with Macro day 60 with sentiment day 0.

Worse, VIX/SPY/XLE cover 378 US equity sessions over a span where CL
trades 468 days, and they were joined by row position:

- `build_vrp_features` guards on `len(vix_df) >= n`. With 378 < 468 this
  was always false, so VIX silently fell back to a placeholder — making
  the `vrp` column (`iv_30 - rv_30`) identically zero.
- `build_macro_features` padded SPY/XLE returns with 90 zeros and
  correlated CL against the padding.

This is also the mechanism behind bug 2: with `n_slow = 408`, the daily
grid ended 2024-10-22 while bars ran to 2024-12-31, orphaning the last
2.5 months.

**Fix.** `reindex_daily()` joins the external series onto the CL calendar
by date, and all slow kernels are resampled onto one common daily axis by
date before use.

## Bug 5 — full-sample standardisation

Seven feature builders finished with

```python
mu, sigma = features.mean(axis=0), features.std(axis=0) + 1e-8
return (features - mu) / sigma
```

`mu` and `sigma` are computed over every row, including rows that had not
happened yet at bar i, so each training bar was standardised against the
sample's future distribution. VPIN also divided volume by the whole
sample's mean volume, and Kyle z-scored its price-impact residual the same
way.

**Fix.** `expanding_zscore()` standardises row i against rows 0..i only,
implemented with running sums so it stays O(n) — LOB carries 372,893 rows
and Hawkes 157,824, which rules out a Python loop. Warm-up rows return
zero rather than a ratio of two near-zero numbers.

## Corrected results

Run 25 configuration, unchanged, on the fixed pipeline:

```bash
ARCH_VERSION=v2 SIGNAL_MODE=kalman KALMAN_Q=aggressive TUNE_KERNELS=0 \
WARMUP_DAYS=60 VOL_SIZING=1 ENTRY_CHOPPY=1.50 EXIT_TRENDING=0.10 \
EXIT_CHOPPY=0.35 .venv/bin/python run_backtest.py
```

| Metric | Published (buggy) | Bugs 1–4 fixed | Bug 5 also fixed |
|--------|-------------------|----------------|------------------|
| Sharpe | 1.56 | 1.10 | **1.41** |
| Sortino | 2.85 | 1.63 | 2.32 |
| Deflated SR | 0.97 | 0.76 | 0.84 |
| Max DD | 3.6% | 2.6% | 2.3% |
| Hit rate | 55.0% | 65.0% | 63.6% |
| Trades | 20 | 32 | 28 |
| Days in position | — | 40 of 172 | 44 of 172 |
| Stage-2 OOS correlation | — | +0.0052 | **−0.0059** |
| Stage-2 OOS R² | — | −0.0011 | **−0.0033** |

**Do not read 1.41 as evidence of edge.** The middle and right columns are
the clearest result in this document: removing a lookahead leak — strictly
*less* information available to the model — moved the Sharpe from 1.10 to
1.41 while the out-of-sample prediction got measurably worse. A Sharpe
that improves when you take information away is not measuring predictive
skill. It is measuring something else.

The rest of the diagnostics agree:

- Stage-2 OOS correlation is **−0.0059** with R² **−0.0033** — worse than
  predicting the mean, and now negative.
- ElasticNet weights are ~0 on everything, and OrderFlow — the entire
  consolidated microstructure stack, LOB + VPIN + Kyle + Hawkes — gets
  **exactly 0**. Only Sentiment (0.047) is meaningfully non-zero.
- The P&L comes from 44 active days out of 172. That is not a sample you
  can draw a conclusion from.

The same intercept-drift mechanism that produced 1.56 is what produces
1.41; the bugs are gone, but the strategy has not been shown to predict CL
direction, and the headline number moves largely independently of whether
the model got better or worse. This matches the April 2026 bug report's
conclusion that the kernels are vol-regime detectors, not return
predictors. That report's post-fix hit rate (16.7%) came from a different
fix that did not include the slow-layer corrections here, so the numbers
are not directly comparable.

## What is still open

- **The headline metric is not diagnostic.** Before tuning anything else,
  the pipeline needs a measure of the kernels that does not route through
  44 days of position P&L.
- **Removed as dead or unsafe:** `backtesting/test_pipeline.py` (imported
  a nonexistent `lob_kernel` module, so the "11 synthetic end-to-end
  tests" it was credited with had not run in a long time) and
  `run_mkl_optimize.py` (still contained bugs 1 and 2 verbatim and would
  have reproduced the fake result). The demo block at the bottom of
  `backtesting/visualize_backtest.py` imports the same missing
  `lob_kernel` and is likewise dead — left in place, but do not trust it.
  `test_feature_alignment.py` (21 tests) is what currently passes.
- **`figures/` and `reports/` are no longer tracked.** Every chart in them
  came from the pre-fix pipeline. Regenerate after a run.
- **The null test is invalid.** The "no kernels → Sharpe −0.76 vs +1.55
  with kernels" claim was measured on the buggy pipeline and has not been
  re-run.
- Every number in `results_table.txt` and `next_steps.txt` for Runs 1–25
  predates this fix.
