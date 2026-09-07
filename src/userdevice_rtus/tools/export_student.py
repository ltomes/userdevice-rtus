"""Export a trained student checkpoint to fixed-shape ONNX + parity ref.

Usage: python -m userdevice_rtus.tools.export_student <ckpt.safetensors> <outname> [arch]

arch defaults to rtmosr_l for backwards compatibility with the earlier
students. The film tier is a DIFFERENT architecture (RTMoSREA, dim 48 /
3 blocks) and this script silently could not export it before 2026-08-22 —
quality work had produced a checkpoint with no path to a deployable engine.
Writes: onnx_out/<outname>-{540p,720p,1080p}.onnx
        onnx_out/<outname>-parity.npz  (fixed real input + torch fp32 output)
Runs in the container image (CPU is fine for RTMoSR).
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

CKPT, NAME = sys.argv[1], sys.argv[2]
ARCH = sys.argv[3] if len(sys.argv) > 3 else "rtmosr_l"
from userdevice_rtus.paths import DATA_ROOT as _RTUS_ROOT
from userdevice_rtus.paths import v1_dir

OUT = f"{_RTUS_ROOT}/onnx_out"
SHAPES = {"540p": (540, 960), "720p": (720, 1280), "1080p": (1080, 1920)}

BASE = dict(scale=2, ffn_expansion=2, unshuffle_mod=True, dccm=True, se=True)


def build(arch):
    if arch == "rtmosr_l":
        from userdevice_rtus.rtmosr_vendored import RTMoSR
        return RTMoSR(dim=32, n_blocks=2, **BASE)
    if arch == "rtmosr_ea_film":
        from userdevice_rtus.rtmosr_ea_vendored import RTMoSREA
        return RTMoSREA(dim=48, n_blocks=3, **BASE)
    if arch == "rtmosr_ea_film_sd":
        from userdevice_rtus.rtmosr_ea_vendored import RTMoSREA
        return RTMoSREA(dim=64, n_blocks=6, **BASE)
    raise ValueError(f"unknown arch {arch!r} — expected rtmosr_l, "
                     f"rtmosr_ea_film or rtmosr_ea_film_sd")


os.makedirs(OUT, exist_ok=True)
model = build(ARCH)
model.load_state_dict(load_file(CKPT), strict=True)
model.eval()
print(f"built {ARCH}, "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params",
      flush=True)

for sname, (h, w) in SHAPES.items():
    x = torch.rand(1, 3, h, w)
    path = f"{OUT}/{NAME}-{sname}.onnx"
    torch.onnx.export(model, x, path, opset_version=17,
                      input_names=["input"], output_names=["output"],
                      do_constant_folding=True, dynamo=False)
    print(f"exported {path}", flush=True)

# parity reference on a real image at the 540p working point
_VAL_LR = str(v1_dir() / "val" / "lr")
src = sorted(os.listdir(_VAL_LR))[0]
img = Image.open(f"{_VAL_LR}/{src}").convert("RGB")
img = img.resize((960, 540), Image.BICUBIC)
x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0
                     ).permute(2, 0, 1)[None]
with torch.no_grad():
    y = model(x).clamp(0, 1)
np.savez_compressed(f"{OUT}/{NAME}-parity.npz",
                    x=x.numpy(), y=y.numpy())
print(f"parity ref written ({src})")
