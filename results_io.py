"""
Picking the right results file.

Several tools auto-discover a `backtest_results_*.npz` when not given one.
That was harmless when the directory held only real runs. It is not
harmless now: `validate_pipeline.py` writes control runs whose features
are pure noise or whose target has been shuffled, and a tool that grabs
"the most recent .npz" will happily build a report out of one.

`select_results_file()` refuses to return a control run, and refuses to
return the abbreviated baseline stub that `run_backtest.py` also writes
(it holds only a handful of arrays and is missing most keys).
"""

from __future__ import annotations

import glob
import os

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))

CONTROL_MARKER = "_null-"
STUB_MARKER = "_baseline"


def is_control_run(path: str) -> bool:
    """True if this file came from a NULL_MODE falsification run."""
    if CONTROL_MARKER in os.path.basename(path):
        return True
    try:
        d = np.load(path, allow_pickle=True)
        return "null_mode" in d.files and str(d["null_mode"]) != "none"
    except Exception:
        return False


def select_results_file(
    explicit: str | None = None,
    require: tuple[str, ...] = (),
    root: str = _ROOT,
) -> str:
    """
    Return the newest real results file, or `explicit` if given.

    `explicit` is honoured as-is apart from an existence check — if you
    deliberately point a tool at a control run, that is your call, but you
    get told.

    `require` names arrays the caller needs; files missing them are
    skipped, which is what keeps the baseline stub out of the way.
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(explicit)
        if is_control_run(explicit):
            print(f"  NOTE: {os.path.basename(explicit)} is a falsification "
                  f"control run — its features or target were destroyed on "
                  f"purpose.")
        return explicit

    candidates = sorted(
        glob.glob(os.path.join(root, "backtest_results_*.npz")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No backtest_results_*.npz found — run run_backtest.py first."
        )

    skipped_control = 0
    for path in candidates:
        if is_control_run(path):
            skipped_control += 1
            continue
        if STUB_MARKER in os.path.basename(path):
            continue
        if require:
            try:
                files = set(np.load(path, allow_pickle=True).files)
            except Exception:
                continue
            if not set(require).issubset(files):
                continue
        if skipped_control:
            print(f"  (skipped {skipped_control} falsification control run(s))")
        return path

    raise FileNotFoundError(
        f"No usable results file among {len(candidates)} candidates. "
        f"{skipped_control} were falsification control runs; the rest were "
        f"stubs or missing required arrays {list(require)}. "
        f"Run run_backtest.py to produce a real one."
    )
