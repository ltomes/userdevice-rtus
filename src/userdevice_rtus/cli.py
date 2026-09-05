"""Console entry points.

Deliberately only two. Everything in `userdevice_rtus.tools` is already
runnable as `python -m userdevice_rtus.tools.<name>`, so wrapping each one in
a console script would add documentation and maintenance surface without
adding a capability.

These two exist because they do something `python -m` cannot:

`train` has real work to do. traiNNer is an application, not a library: its
entry point is a train.py at its repository root, which expects to be run
from that root with the root importable. Without this wrapper a Kubernetes
Job would carry `bash -c "cd ... && python train.py ..."`, and inline shell
in a Job spec is precisely what this layout removes.

`info` reports what an environment can actually do, which is the difference
between "the model is wrong" and "the environment is incomplete".
"""

from __future__ import annotations

import runpy
import sys


def train() -> None:
    """Train a config. Requires traiNNer, and fails loudly without it."""
    import os
    from pathlib import Path

    from userdevice_rtus import require_real_registry

    # Importing the package registers the architectures; this check turns a
    # silent "unknown architecture" much later into an immediate, explained
    # failure here.
    require_real_registry()

    # traiNNer's entry point is train.py at the REPO ROOT, outside the
    # `traiNNer` package -- so there is no `traiNNer.train` module to run.
    # Locate it relative to the installed package instead of hardcoding a
    # path, so this works for any checkout location.
    import traiNNer

    root = Path(traiNNer.__file__).resolve().parent.parent
    script = root / "train.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"traiNNer's train.py not found at {script}. Expected it beside "
            f"the traiNNer package (its repository root)."
        )

    # Upstream is an application: train.py resolves experiment directories
    # relative to the working directory, and traiNNer imports its own
    # top-level `scripts` package. Both only work when you are standing in the
    # repo root with it importable, which is what running it as an application
    # normally gives you.
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # traiNNer writes checkpoints to a RELATIVE experiments/ under its own
    # tree, but the configs and paths.EXPERIMENTS both name
    # $RTUS_DATA_ROOT/experiments -- which is the directory that actually gets
    # mounted. Without this link the two diverge silently and a run's output
    # lands inside the container, to be lost when it exits.
    from userdevice_rtus.paths import EXPERIMENTS

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    link = root / "experiments"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(EXPERIMENTS, target_is_directory=True)

    runpy.run_path(str(script), run_name="__main__")


def info() -> None:
    """Print what this install can actually do.

    First thing to run after installing, and the first thing to ask for in a
    bug report: it distinguishes "the model is wrong" from "the environment is
    incomplete", which are otherwise easy to confuse.
    """
    import userdevice_rtus as rtus
    from userdevice_rtus import paths

    print(f"userdevice-rtus     tiers: {', '.join(sorted(rtus.TIERS))}")
    registry = "YES" if rtus.USING_REAL_REGISTRY else "NO"
    capability = (
        "training available"
        if rtus.USING_REAL_REGISTRY
        else "EVAL/EXPORT ONLY — cannot train"
    )
    print(f"traiNNer registry   {registry}  ({capability})")
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
