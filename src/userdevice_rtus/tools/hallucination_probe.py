"""Texture gain vs invented structure — the two axes of a texture push.

The face gate answers one question well ("did we draw a face that isn't
there?"). The project's constraint is broader: chase more texture detail, but
reject detail the model invented rather than recovered. This measures both
axes at once so a texture push can be judged instead of guessed at:

  HF RECOVERY    high-frequency energy as a FRACTION of the source's own.
                 A 2x downsample destroys detail no model can fully restore,
                 so every candidate sits below 1.0 and the number reads as
                 "how much of the lost detail came back". Higher = more
                 apparent detail. This is what we WANT.
  INVENTION RATE fraction of tiles carrying real structure that does NOT
                 correspond to the ground truth. This is what we must NOT
                 have.

CALIBRATION NOTE (2026-08-22): the first version gated invention on a tile
exceeding 1.5x GT energy. Since every candidate averages BELOW 1.0x, that
threshold could never fire and reported a meaningless 0.00% for everything,
including bicubic — a detector that cannot detect. Invention is now keyed on
structural correlation alone, over tiles where the SR actually carries
structure. Amplitude was always the wrong axis: a model can invent detail
while still being globally softer than the source.

The discrimination between the two is structural correlation, not amplitude.
Sharpening amplifies detail that is already there, so its high-frequency map
stays ALIGNED with the ground truth's (high correlation). Hallucination
synthesises plausible texture that has no counterpart in the source, so the
correlation collapses even though energy rises. A tile is counted as
invented only when energy rises meaningfully AND correlation is poor.

Flat regions are excluded: an almost-featureless GT tile has no structure to
correlate against, so its correlation is meaningless noise, and grain added
to a smooth wall would otherwise dominate the score.

Scoring the teacher alongside our students is the point — it tells us
whether the model we distil from is itself over-inventing, which the user
observed by eye on 2026-08-22.

Usage (inside the container image, cwd /workspace):
  python hallucination_probe.py --n 60 \
      --ckpt stageC=<path>:rtmosr_ea_film [--out results/x.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter


from userdevice_rtus.paths import DATA_ROOT as _RTUS_ROOT

VAL_SETS = [f"{_RTUS_ROOT}/datasets/greyduck2x_v2", f"{_RTUS_ROOT}/datasets/greyduck2x"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TILE = 32
CORR_FLOOR = 0.35    # below this, SR structure does not match GT structure
FLAT_FLOOR = 2.0     # GT HF energy below this = flat, correlation meaningless
SR_FLOOR = 2.0       # SR must carry real structure to be judged at all


def build_model(arch, ckpt):
    from safetensors.torch import load_file
    from userdevice_rtus.rtmosr_ea_vendored import RTMoSREA
    dims = {"rtmosr_ea_film": (48, 3), "rtmosr_ea_film_sd": (64, 6)}
    dim, nb = dims[arch]
    m = RTMoSREA(scale=2, dim=dim, ffn_expansion=2, n_blocks=nb,
                 unshuffle_mod=True, dccm=True, se=True)
    m.load_state_dict(load_file(ckpt), strict=True)
    return m.eval().to(DEV)


def highfreq(img):
    """Luma high-pass: original minus a gaussian blur of itself."""
    g = img.convert("L")
    lo = g.filter(ImageFilter.GaussianBlur(radius=1.6))
    return np.asarray(g).astype(np.float32) - np.asarray(lo).astype(np.float32)


def score(sr_img, gt_img):
    """(texture_gain, invention_rate, n_tiles_considered)."""
    hs, hg = highfreq(sr_img), highfreq(gt_img)
    h, w = min(hs.shape[0], hg.shape[0]), min(hs.shape[1], hg.shape[1])
    hs, hg = hs[:h, :w], hg[:h, :w]

    gains, invented, considered, corrs = [], 0, 0, []
    for y in range(0, h - TILE + 1, TILE):
        for x in range(0, w - TILE + 1, TILE):
            a = hs[y:y + TILE, x:x + TILE].ravel()
            b = hg[y:y + TILE, x:x + TILE].ravel()
            eb = float(np.abs(b).mean())
            if eb < FLAT_FLOOR:
                continue          # no structure to correlate against
            ea = float(np.abs(a).mean())
            considered += 1
            gains.append(ea / eb)
            if ea < SR_FLOOR:
                continue          # SR is flat here; nothing to judge
            sa, sb = a.std(), b.std()
            if sa < 1e-6 or sb < 1e-6:
                continue
            corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
            corrs.append(corr)
            if corr < CORR_FLOOR:
                invented += 1
    if not considered:
        return None
    return (float(np.mean(gains)), invented / considered, considered, corrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="LABEL=path:arch")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = {}
    for spec in args.ckpt:
        label, rest = spec.split("=", 1)
        path, arch = rest.rsplit(":", 1)
        models[label] = build_model(arch, path)
        print(f"loaded {label}", flush=True)

    triples = []
    for root in VAL_SETS:
        for lr_p in sorted(glob.glob(f"{root}/val/lr/*.png")):
            b = os.path.basename(lr_p)
            gt_p, t_p = f"{root}/val/hr/{b}", f"{root}_teacher/val/hr/{b}"
            if os.path.exists(gt_p):
                triples.append((lr_p, gt_p, t_p))
    triples = triples[:: max(1, len(triples) // args.n)][:args.n]
    print(f"probing {len(triples)} frames", flush=True)

    acc = {}

    def record(label, sr, gt):
        r = score(sr, gt)
        if r is None:
            return
        d = acc.setdefault(label, {"gain": [], "inv": [], "tiles": 0})
        d["gain"].append(r[0])
        d["inv"].append(r[1])
        d["tiles"] += r[2]
        d.setdefault("corrs", []).extend(r[3])

    for lr_p, gt_p, t_p in triples:
        lr = Image.open(lr_p).convert("RGB")
        gt = Image.open(gt_p).convert("RGB")
        record("bicubic", lr.resize(gt.size, Image.BICUBIC), gt)
        if os.path.exists(t_p):
            record("teacher_web", Image.open(t_p).convert("RGB"), gt)
        x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0
                             ).permute(2, 0, 1)[None].to(DEV)
        for label, model in models.items():
            with torch.inference_mode():
                y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            record(label, Image.fromarray((y * 255).round().astype(np.uint8)), gt)

    summary = {}
    for k, v in acc.items():
        c = np.array(v.get("corrs", [0.0]))
        summary[k] = {"hf_recovery": float(np.mean(v["gain"])),
                      "invention_rate": float(np.mean(v["inv"])),
                      "corr_p05": float(np.percentile(c, 5)),
                      "corr_median": float(np.median(c)),
                      "tiles": v["tiles"], "frames": len(v["gain"])}

    print("\n| candidate | HF recovery | invention rate | corr p05 | "
          "corr median | tiles |")
    print("|---|---|---|---|---|---|")
    for k in ["bicubic"] + list(models) + ["teacher_web"]:
        if k not in summary:
            continue
        v = summary[k]
        print(f"| {k} | {v['hf_recovery']:.3f}x | "
              f"{v['invention_rate'] * 100:.2f}% | {v['corr_p05']:.3f} | "
              f"{v['corr_median']:.3f} | {v['tiles']} |")
    print("\nbicubic is the control: it CANNOT invent, so its invention rate "
          "is the detector's false-positive floor. Read every other row as a "
          "delta against it, not as an absolute.")

    if "teacher_web" in summary:
        t = summary["teacher_web"]["invention_rate"]
        print(f"\nteacher invention rate = {t * 100:.2f}% — a student ABOVE "
              f"this is inventing more than the model we distil from")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
