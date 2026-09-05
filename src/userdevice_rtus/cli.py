"""Console entry points.

The tools in `userdevice_rtus.tools` are scripts: they do their work at
import time and read `sys.argv` directly. That is deliberate -- they are
research tools and rewriting them into libraries would risk changing
measured behaviour for no benefit.

So each entry point below runs its module the way `python -m` would, which
preserves `__name__ == "__main__"` semantics and argv handling exactly, while
still giving contributors and Kubernetes jobs a stable command to call
instead of a path into the source tree.
"""

from __future__ import annotations

import runpy
import sys


def _run(module: str) -> None:
    sys.exit(runpy.run_module(module, run_name="__main__") and 0)


def train() -> None:
    """Train a config. Requires traiNNer, and fails loudly without it."""
    from userdevice_rtus import require_real_registry

    # Importing the package registers the architectures; this check turns a
    # silent "unknown architecture" much later into an immediate, explained
    # failure here.
    require_real_registry()
    runpy.run_module("traiNNer.train", run_name="__main__")


def evaluate() -> None:
    _run("userdevice_rtus.tools.eval_compare")


def export() -> None:
    _run("userdevice_rtus.tools.export_student")


def facegate() -> None:
    _run("userdevice_rtus.tools.facegate.face_gate")


def probe() -> None:
    _run("userdevice_rtus.tools.hallucination_probe")


def gen_pairs() -> None:
    _run("userdevice_rtus.tools.gen_pairs_v2")


def teacher_targets() -> None:
    _run("userdevice_rtus.tools.gen_teacher_targets")


def blend_targets() -> None:
    _run("userdevice_rtus.tools.gen_blend_targets")


def info() -> None:
    """Print what this install can actually do.

    First thing to run after installing, and the first thing to ask for in a
    bug report: it distinguishes "the model is wrong" from "the environment is
    incomplete", which are otherwise easy to confuse.
    """
    import userdevice_rtus as rtus
    from userdevice_rtus import paths

    print(f"userdevice-rtus     tiers: {', '.join(sorted(rtus.TIERS))}")
    print(f"traiNNer registry   {'YES' if rtus.USING_REAL_REGISTRY else 'NO'}"
          f"  ({'training available' if rtus.USING_REAL_REGISTRY else 'EVAL/EXPORT ONLY — cannot train'})")
    print(f"RTUS_DATA_ROOT      {paths.DATA_ROOT}")
    for label, p in [
        ("datasets", paths.DATASETS),
        ("pretrained", paths.PRETRAINED),
        ("teacher", paths.TEACHER),
        ("scrfd (face gate)", paths.SCRFD),
    ]:
        print(f"  {label:<18} {'present' if p.exists() else 'MISSING':<8} {p}")

    try:
        import torch

        print(f"torch               {torch.__version__} "
              f"(cuda: {torch.cuda.is_available()})")
    except ImportError:
        print("torch               MISSING")
