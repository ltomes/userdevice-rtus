"""Precompute teacher targets for stage-B distillation.

teacher(LR) with 4xNomosWebPhoto_RealPLKSR (loaded via spandrel), output
downscaled 2x (Lanczos) to the student's GT size. The downscale itself
attenuates GAN hallucination (measured 2026-08-16). Teacher was
trained WITH noise injection so it denoises — stage B2 finishes on real GT
to restore grain (ECO-style staged annealing).

Runs in the container image; shares the GPU with stage-A (student
training uses ~6%). Idempotent — skips existing outputs.
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

from userdevice_rtus.paths import DATA_ROOT as _RTUS_ROOT

BASE = f"{_RTUS_ROOT}"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "rtus2x"
TEACHER = os.path.join(BASE, "pretrained/teacher_4xNomosWebPhoto_RealPLKSR.safetensors")

model = ModelLoader().load_from_file(TEACHER).model.eval().cuda().half()

for split in ("train", "val"):
    src = os.path.join(BASE, f"datasets/{DATASET}/{split}/lr")
    dst = os.path.join(BASE, f"datasets/{DATASET}_teacher/{split}/hr")
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.png")))
    done = 0
    for i, p in enumerate(files):
        out_p = os.path.join(dst, os.path.basename(p))
        if os.path.exists(out_p):
            continue
        lr = Image.open(p).convert("RGB")
        x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0
                             ).permute(2, 0, 1)[None].cuda().half()
        with torch.no_grad():
            y = model(x)
        y = y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray((y * 255).round().astype(np.uint8))
        img = img.resize((lr.width * 2, lr.height * 2), Image.LANCZOS)
        img.save(out_p)
        done += 1
        if done % 250 == 0:
            print(f"[{split}] {done} generated ({i+1}/{len(files)} seen)", flush=True)
    print(f"[{split}] complete: {len(glob.glob(os.path.join(dst, '*.png')))} targets", flush=True)
