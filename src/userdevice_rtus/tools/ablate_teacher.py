"""Teacher mechanism ablation (2026-08-20): which parts of
4xNomosWebPhoto_RealPLKSR produce the attributes we want to mimic?

Method: run the intact teacher on N val LR images (baseline), then re-run
with one mechanism ablated at a time; report PSNR(ablated, baseline) — how
much of the teacher's own transform survives — plus PSNR(bicubic, baseline)
as the floor ("no mechanism at all"). High survival = mechanism is not
where the look lives; low = it is.

Ablations:
  A. kernel support: zero outer rings of every 17x17 lk conv -> effective
     13/9/5/3/1 (receptive-field contribution).
  B. per-block skip: bypass block i (blocks are residual -> clean).
  C. depth truncation: keep first k blocks only.
  D. EA attention -> identity everywhere (instance modulation contribution).

Runs in the container image, ~2 GB VRAM, light IO (N images).
"""
import copy
import glob
import json

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

N_IMAGES = 24
from userdevice_rtus.paths import DATA_ROOT as _RTUS_ROOT

TEACHER = f"{_RTUS_ROOT}/pretrained/teacher_4xNomosWebPhoto_RealPLKSR.safetensors"
VAL = f"{_RTUS_ROOT}/datasets/rtus2x/val/lr"
OUT = f"{_RTUS_ROOT}/teacher_ablation.json"

base_model = ModelLoader().load_from_file(TEACHER).model.eval().cuda().half()

files = sorted(glob.glob(f"{VAL}/*.png"))
files = files[:: max(1, len(files) // N_IMAGES)][:N_IMAGES]
imgs = []
for p in files:
    lr = Image.open(p).convert("RGB")
    imgs.append(torch.from_numpy(
        np.asarray(lr).astype(np.float32) / 255.0
    ).permute(2, 0, 1)[None].cuda().half())


def run(model):
    outs = []
    with torch.no_grad():
        for x in imgs:
            outs.append(model(x).float().clamp(0, 1))
    return outs


def psnr_vs(outs, ref):
    vals = []
    for a, b in zip(outs, ref):
        mse = torch.mean((a - b) ** 2).item()
        vals.append(99.0 if mse == 0 else 10 * np.log10(1.0 / mse))
    return round(float(np.mean(vals)), 2)


baseline = run(base_model)
results = {"n_images": len(imgs)}

# floor: bicubic x4 (no teacher at all)
bic = []
for x in imgs:
    h, w = x.shape[-2:]
    bic.append(torch.nn.functional.interpolate(
        x.float(), size=(h * 4, w * 4), mode="bicubic",
        align_corners=False).clamp(0, 1))
results["floor_bicubic"] = psnr_vs(bic, baseline)

blocks = [m for m in base_model.modules()
          if type(m).__name__ == "PLKBlock"]
print(f"teacher: {len(blocks)} PLKBlocks", flush=True)

# A. kernel ring-zeroing
results["kernel_support"] = {}
for eff in (13, 9, 5, 3, 1):
    m = copy.deepcopy(base_model)
    for blk in m.modules():
        if type(blk).__name__ == "PLKConv2d":
            w = blk.conv.weight.data
            k = w.shape[-1]
            lo, hi = (k - eff) // 2, (k - eff) // 2 + eff
            mask = torch.zeros_like(w)
            mask[..., lo:hi, lo:hi] = 1
            w.mul_(mask)
    results["kernel_support"][eff] = psnr_vs(run(m), baseline)
    del m
    print(f"kernel {eff}: {results['kernel_support'][eff]}", flush=True)

# B. per-block skip (residual blocks -> forward = identity)
results["block_skip"] = {}
for i, blk in enumerate(blocks):
    orig = blk.forward
    blk.forward = lambda x: x
    results["block_skip"][i] = psnr_vs(run(base_model), baseline)
    blk.forward = orig
print("block_skip:", results["block_skip"], flush=True)

# C. depth truncation: keep first k blocks
results["depth_keep"] = {}
for k in (7, 14, 21):
    origs = [b.forward for b in blocks]
    for b in blocks[k:]:
        b.forward = lambda x: x
    results["depth_keep"][k] = psnr_vs(run(base_model), baseline)
    for b, f in zip(blocks, origs):
        b.forward = f
    print(f"depth {k}: {results['depth_keep'][k]}", flush=True)

# D. EA attention -> identity
m = copy.deepcopy(base_model)
n_ea = 0
for blk in m.modules():
    if type(blk).__name__ == "PLKBlock" and type(blk.attn).__name__ == "EA":
        blk.attn = torch.nn.Identity()
        n_ea += 1
results["ea_removed"] = psnr_vs(run(m), baseline) if n_ea else "no EA"
results["n_ea"] = n_ea
del m

with open(OUT, "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
