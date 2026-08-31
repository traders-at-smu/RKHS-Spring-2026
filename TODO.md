# TODO

Ordered roughly by what blocks what. Each item says *why* it matters, not
just what to do — if a reason stops being true, drop the item.

Background for all of it is in `POSTMORTEM.md` (what was broken) and
`HANDOFF.md` (where this is going).

---

## 1. Build a measurement you can trust

**Do this before anything else.**

Right now the number everyone quotes — the Sharpe ratio — does not work.
We tested it by scrambling the answer key: we shuffled the thing the model
is supposed to predict, so that *by construction* nothing could predict it.
That scrambled version scored **better** than the real strategy (1.15 vs
0.28, averaged over three runs).

Think of it like grading an exam where the answer key has been shredded and
students still score 80%. The problem isn't the students — it's that the
grading tells you nothing.

Why this blocks everything: if you change the model and the score goes up,
you cannot tell whether you improved anything or just got lucky. Every
tuning decision after this point is a coin flip until there's a real
measurement. This is also why the old Runs 1–25 rankings are worthless —
not only because of the bugs, but because they were ranked on noise.

**What a working metric looks like:** something measured directly on the
model's predictions, not on trading profit. Per-kernel correlation with
what actually happened next, reported fold by fold, averaged over several
random seeds. Profit is a bad measuring instrument here because it passes
through position sizing, entry/exit thresholds and only 44 active days —
too many places for luck to enter.

**How to know you succeeded:** your new metric must score the real pipeline
differently from `NULL_MODE=shuffle_target`. If it can't tell those apart,
it isn't a metric. Run `validate_pipeline.py` to check.

---

## 2. Decide whether you're predicting hours or days

The model predicts the return of the **next dollar bar**. There are about
12 bars per trading day, so it is forecasting roughly the next two hours.
But profit is measured on **end-of-day positions held overnight**.

So we train the model to answer one question and grade it on a different
one. Neither is necessarily wrong, but they should match. Pick one:

- predict and trade at bar resolution, or
- predict and trade daily.

This is cheap to change and probably affects results more than most of the
parameter tuning in the old run history.

---

## 3. Get tick data for more contracts

**This is what actually blocks the multi-asset project.**

The kernels are built from trade-by-trade and order-book data — who traded,
how much, how fast, how deep the book was. Daily closing prices don't
contain any of that, so the microstructure kernels have literally nothing
to compute from them.

We have this data for CL (crude oil) only, from Databento's GLBX.MDP3 feed.
Until there's a second market, "expand to many futures" cannot start.

Priority order from the April bug report: **ES** (S&P E-mini, the most
liquid futures contract in the world) and **NQ** (Nasdaq E-mini). Two
markets is enough to build and test the cross-market machinery; you don't
need all 26 to begin.

Worth pricing out a Databento subscription early — this has a lead time
and everything downstream waits on it.

---

## 4. Make the pipeline handle more than one symbol

`run_backtest.py` is currently one very long function with "CL" assumptions
threaded all the way through. To run ten markets you need to be able to say
"build the features for symbol X" and call it ten times.

The good news: the hard part is already done. Every kernel is now placed on
a master schedule built from that market's own trades (`feature_alignment.py`),
so each symbol naturally gets its own schedule. Comparison across markets
then happens on the daily calendar, where the dates actually line up.

Extract a `build_features_for_symbol(symbol)` function first. Everything
else follows from that.

**Please keep `validate_pipeline.py` passing throughout.** Multi-asset work
multiplies the number of places a date can be misaligned, and all five bugs
we just fixed were misalignment bugs that nothing detected for months.

---

## 5. Change the question from "will CL go up" to "which market looks best"

This is the actual research idea, and Rebecca's reason for expecting the
current results to stay poor no matter how much they're tuned.

An RKHS measures how *similar* two market states are. That's most useful
when you have many states to compare and rank. Asking it to call the
direction of one contract is the single thing it's worst at — and the
measurements agree: replacing every feature with random numbers performs
just as well as the real ones on CL.

The architecture the April report proposed, and the one worth building:

- **Direction** comes from something simple and non-RKHS — a trend
  signal per market. This is a commodity; nobody has an edge in it.
- **Size** comes from the kernels — they rank markets by predicted
  volatility regime and how clean the order flow looks. Markets that look
  calm and orderly get a full position; markets that look toxic get scaled
  down or skipped.
- **Diversification** does the rest. Any single market's edge is tiny; the
  bet is that it compounds across many uncorrelated ones.

The existing VPIN/Hawkes position-sizing gate is already a small version of
this idea on one market, and it was the one component that behaved sensibly
in testing.

---

## 6. Smaller things, whenever

- **Run the falsification suite with more seeds.** We used three, which is
  enough to show the metric is broken but not enough to measure precisely
  how broken. `validate_pipeline.py --seeds 10`, about 35 minutes.

- **Eighteen months of data is not much.** The whole backtest is July 2023
  to December 2024, which collapses to 172 evaluation days and 44 days
  actually holding a position. Even a perfect strategy would be hard to
  prove on that. More history would help; so would more markets (see above),
  since ten markets over the same window give ten times the observations.

- **GammaExposure is still not wired in.** It needs gamma data that was
  never delivered. Either chase it or delete the stub so nobody assumes
  it's running — right now the code sets it to `None` unconditionally.

- **Dead code in `backtesting/visualize_backtest.py`.** The demo block at
  the bottom imports a module (`lob_kernel`) that doesn't exist in this
  repo, so it cannot run. Fix it or delete it.

- **The old run history is kept but invalid.** `results_table.txt`,
  `next_steps.txt`, `run_configs.txt` and `pipeline.txt` still contain the
  Runs 1–25 numbers, behind warning banners. That's on purpose — the
  reasoning is worth reading even though the numbers are wrong. But if you
  find people quoting them anyway, delete them.

- **Check the transaction cost assumption.** Costs are modelled as a flat
  0.015% per trade, roughly one tick on CL at $70/barrel. That's a
  reasonable guess for CL and probably wrong for other contracts. Revisit
  when the book expands.

---

## Rules that shouldn't be broken along the way

These are in `HANDOFF.md` too, but they're the ones that caused every bug
we just fixed:

1. Never line up two datasets by row number. Use timestamps.
2. Never normalise a feature using statistics from the whole sample — that
   tells the model about the future.
3. Join any new data source on the date, not the row.
4. Any new fast kernel must return timestamps.
5. Tools that go looking for a results file must use `results_io.py`, or
   they'll happily report on one of the deliberately-broken control runs.
