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

#: Validation image sets, in the order tools should search them.
VAL_SETS = [
    DATASETS / "rtus2x_v2" / "val",
    DATASETS / "rtus2x" / "val",
]


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
