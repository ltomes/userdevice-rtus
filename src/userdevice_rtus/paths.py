"""Where the data lives.

Every tool in this package used to hardcode `/workspace/...`. That works
inside the container and nowhere else, which made the scripts effectively
container-only without ever saying so.

One environment variable now decides:

    RTUS_DATA_ROOT      default "/workspace"

The default keeps the container behaviour identical, so nothing that worked
before stops working; a contributor running on a laptop sets the variable
instead of editing source.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Root under which datasets, pretrained weights and outputs live.
DATA_ROOT = Path(os.environ.get("RTUS_DATA_ROOT", "/workspace"))

DATASETS = DATA_ROOT / "datasets"
PRETRAINED = DATA_ROOT / "pretrained"
EXPERIMENTS = DATA_ROOT / "experiments"
RESULTS = DATA_ROOT / "results"
ONNX_OUT = DATA_ROOT / "onnx_out"

#: The teacher, under the exact stem the tools expect.
TEACHER = PRETRAINED / "teacher_4xNomosWebPhoto_RealPLKSR.safetensors"

#: SCRFD detector used by the (blocking) face gate.
SCRFD = PRETRAINED / "scrfd_2.5g_bnkps.onnx"

# --------------------------------------------------------------------------
# Corpus names.
#
# RTUS_DATA_ROOT made the ROOT configurable but left every tool hardcoding the
# corpus NAME, which is the half of the problem that actually bit us
# (talos-cluster-1a20). The package refactor renamed the corpus to rtus2x*;
# the data on odinson is still named greyduck2x*. So RTUS_DATA_ROOT=/ar
# resolves to /ar/datasets/rtus2x_v2, which does not exist -- and a tool then
# either fails or, worse, silently scores zero frames.
#
# These are the SAME CORPUS under two names, so the fix is to SEARCH for both
# rather than to pick one. Order is preference, not exclusivity:
#
#     RTUS_CORPUS_V2   default "rtus2x_v2,greyduck2x_v2"
#     RTUS_CORPUS_V1   default "rtus2x,greyduck2x"
#
# Nothing that worked before stops working: the rtus2x names stay first, so a
# tree holding both resolves exactly as it did.
# --------------------------------------------------------------------------


def _names(var: str, default: str) -> list[str]:
    return [n.strip() for n in os.environ.get(var, default).split(",") if n.strip()]


CORPUS_V2_NAMES = _names("RTUS_CORPUS_V2", "rtus2x_v2,greyduck2x_v2")
CORPUS_V1_NAMES = _names("RTUS_CORPUS_V1", "rtus2x,greyduck2x")

#: Name that new data is WRITTEN under -- always the first V2 candidate, so a
#: generator and the readers that follow it agree even on a bare tree.
PRIMARY_CORPUS = CORPUS_V2_NAMES[0]


def corpus_dirs(names: list[str] | None = None, existing: bool = True) -> list[Path]:
    """Dataset directories for `names`, most-preferred first.

    With `existing=True` (the default) only directories actually on disk come
    back, because an absent alias must never shadow a present one. An empty
    result is meaningful -- the corpus is not on this machine -- and callers
    should say so rather than quietly looping over nothing.
    """
    cand = [DATASETS / n for n in (names if names is not None else
                                   CORPUS_V2_NAMES + CORPUS_V1_NAMES)]
    return [d for d in cand if d.is_dir()] if existing else cand


def val_dirs() -> list[Path]:
    """`val` subdirectories of every corpus present, V2 first."""
    return [d / "val" for d in corpus_dirs() if (d / "val").is_dir()]


def v2_dir() -> Path:
    """The V2 corpus, under whichever name it carries on this machine.

    Distinct from `VAL_SETS[0]`, which falls through to V1 when V2 is absent.
    A tool pinned to V2 must FAIL when V2 is missing rather than quietly score
    a different corpus and report the number as if it were comparable.
    """
    found = corpus_dirs(CORPUS_V2_NAMES)
    return found[0] if found else DATASETS / CORPUS_V2_NAMES[0]


def v1_dir() -> Path:
    """The V1 corpus, under whichever name it carries on this machine.

    A few tools are pinned to V1 on purpose -- the teacher ablation scores a
    fixed image set, and moving it to V2 would silently change what a
    published measurement was measured on. They get the V1 name resolved, not
    the V1 preference relaxed. Falls back to the nominal path so `require()`
    can name something real.
    """
    found = corpus_dirs(CORPUS_V1_NAMES)
    return found[0] if found else DATASETS / CORPUS_V1_NAMES[0]


#: Validation image sets, in the order tools should search them.
#:
#: Falls back to the NOMINAL paths when nothing resolves, so a missing corpus
#: still produces `require()`'s actionable message naming a real path instead
#: of an empty list and a zero-length loop that looks like success.
VAL_SETS = val_dirs() or [d / "val" for d in corpus_dirs(existing=False)]


def require(path: Path, what: str, how: str) -> Path:
    """Fail with an actionable message instead of a bare FileNotFoundError.

    A missing asset here means a fetch step was skipped, so say which one.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n{how}\n"
            f"(RTUS_DATA_ROOT is currently {DATA_ROOT})"
        )
    return path
