"""Quick-mode hooks (release addition): run the real pipeline on a tiny subset.

All hooks are inert unless the corresponding environment variable is set, so the default
behaviour of every script is exactly the paper's.

  TT_QUICK=1          booked-number reproduction guards (which compare a re-computed number
                      with the value booked for the FULL dataset) print a warning instead of
                      aborting -- on a user subset they cannot match by construction.
  TT_MAX_STEPS=N      cap on optimizer steps of every tower trainer.
  TT_MAX_EPOCHS=N     cap on SASRec epochs.

`scripts/reproduce.sh --quick` sets these and runs every step inside a scratch workspace
(`outputs/quick/<dataset>/`) whose `data/` holds the first N users of each sequence file,
so nothing under `results/` is ever touched.
"""
from __future__ import annotations

import os


def is_quick() -> bool:
    return os.environ.get("TT_QUICK") == "1"


def _cap(var: str, default: int) -> int:
    v = os.environ.get(var)
    return min(int(default), int(v)) if v else int(default)


def max_steps(default: int) -> int:
    """`default`, or min(default, $TT_MAX_STEPS) when set."""
    return _cap("TT_MAX_STEPS", default)


def max_epochs(default: int) -> int:
    """`default`, or min(default, $TT_MAX_EPOCHS) when set."""
    return _cap("TT_MAX_EPOCHS", default)


def guard(ok: bool, msg: str) -> bool:
    """Booked-number guard: raise unless TT_QUICK=1, in which case warn and continue."""
    if ok:
        return True
    if is_quick():
        print(f"[quick] guard relaxed (expected on a subset): {msg}", flush=True)
        return False
    raise AssertionError(msg)
