"""Blended stage-B targets: 0.7*GT + 0.3*colorfix(teacher).

Rationale (log.md 2026-08-16): pure teacher targets FAILED the sanity gate
(-0.48 dB vs bicubic against GT; GAN fidelity trade + ~-3.4/255 global tone
bias). GT-dominant blend keeps fidelity, injects teacher texture; per-image
mean-matching (colorfix) removes the tone bias so B doesn't fight A on color.

Parallel across CPU cores (WORKERS env, default cpu_count) — the job is
embarrassingly parallel and was measured CPU-bound at ~48% of one core
(2026-08-22), i.e. single-thread PNG decode/encode was the bottleneck.
Writes are atomic (tmp + rename) so an interrupted run never leaves a
truncated PNG that the skip-if-exists resume would then treat as done.

Usage: gen_blend_targets.py <dataset> [alpha_gt]
(expects <dataset>/ and <dataset>_teacher/ to exist; writes
<dataset>_blend/<split>/hr at the default alpha 0.7, or
<dataset>_blend<teacher-pct>/<split>/hr when alpha is given —
e.g. alpha 0.55 -> <dataset>_blend45. Stage C uses 0.55.)
"""
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

BASE = "/workspace/datasets"
DS = sys.argv[1]
ALPHA_GT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
SUFFIX = "_blend" if len(sys.argv) <= 2 else \
    f"_blend{round((1 - ALPHA_GT) * 100)}"
WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 4))


def blend_one(args):
    """Returns 1 if a target was written, 0 if skipped."""
    gt_p, t_dir, out_dir = args
    base = os.path.basename(gt_p)
    out_p = f"{out_dir}/{base}"
    t_p = f"{t_dir}/{base}"
    if os.path.exists(out_p) or not os.path.exists(t_p):
        return 0
    gt = np.asarray(Image.open(gt_p).convert("RGB")).astype(np.float64)
    t = np.asarray(Image.open(t_p).convert("RGB")).astype(np.float64)
    t += (gt.mean(axis=(0, 1)) - t.mean(axis=(0, 1)))  # colorfix
    blend = ALPHA_GT * gt + (1 - ALPHA_GT) * t
    # atomic: a killed worker leaves a .tmp, never a half-written target
    tmp_p = f"{out_p}.{os.getpid()}.tmp"
    img = Image.fromarray(blend.clip(0, 255).round().astype(np.uint8))
    img.save(tmp_p, format="PNG")
    os.replace(tmp_p, out_p)
    return 1


def main():
    for split in ("train", "val"):
        gt_dir = f"{BASE}/{DS}/{split}/hr"
        t_dir = f"{BASE}/{DS}_teacher/{split}/hr"
        out_dir = f"{BASE}/{DS}{SUFFIX}/{split}/hr"
        os.makedirs(out_dir, exist_ok=True)
        jobs = [(g, t_dir, out_dir)
                for g in sorted(glob.glob(f"{gt_dir}/*.png"))]
        with Pool(WORKERS) as pool:
            n = sum(pool.imap_unordered(blend_one, jobs, chunksize=16))
        print(f"[{split}] {n} blended targets ({WORKERS} workers)", flush=True)


if __name__ == "__main__":
    main()
