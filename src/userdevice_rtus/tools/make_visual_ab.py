"""Four-way visual A/B panels: original | bicubic | ours | web model.

Purpose (user, 2026-08-22): "compare original, to bicubic, to our new model,
to the web model so I have clear perspective on how close we are."

Perceptual differences are invisible at full-frame scale on a monitor, so
each frame produces TWO rows: the full frame, and a 3x nearest-neighbour
zoom of the highest-detail tile (picked by local gradient energy, so the
crop lands on texture rather than a flat wall). Columns are labelled and
always in the same order, and the ORIGINAL is first so the eye calibrates
on ground truth before judging the rest.

Usage (inside the container image, cwd /workspace):
  python -m userdevice_rtus.tools.make_visual_ab --ckpt ours=<path>:rtmosr_ea_film [--n 8]
      [--out /workspace/visual_ab]
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from userdevice_rtus.paths import DATA_ROOT as _RTUS_ROOT

VAL_SETS = [f"{_RTUS_ROOT}/datasets/rtus2x_v2", f"{_RTUS_ROOT}/datasets/rtus2x"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ZOOM = 3
TILE = 96          # crop size in GT pixels before zoom
PAD = 8
LABEL_H = 22


def build_model(arch, ckpt):
    from safetensors.torch import load_file

    from userdevice_rtus.rtmosr_ea_vendored import RTMoSREA
    dims = {"rtmosr_ea_film": (48, 3), "rtmosr_ea_film_sd": (64, 6)}
    if arch not in dims:
        raise ValueError(arch)
    dim, nb = dims[arch]
    m = RTMoSREA(scale=2, dim=dim, ffn_expansion=2, n_blocks=nb,
                 unshuffle_mod=True, dccm=True, se=True)
    m.load_state_dict(load_file(ckpt), strict=True)
    return m.eval().to(DEV)


def busiest_tile(gt):
    """Top-left of the TILE-sized window with the most gradient energy."""
    g = np.asarray(gt.convert("L")).astype(np.float32)
    gy, gx = np.gradient(g)
    e = gx * gx + gy * gy
    h, w = e.shape
    if h <= TILE or w <= TILE:
        return 0, 0
    best, bxy = -1.0, (0, 0)
    step = max(16, TILE // 2)
    for y in range(0, h - TILE, step):
        for x in range(0, w - TILE, step):
            s = float(e[y:y + TILE, x:x + TILE].sum())
            if s > best:
                best, bxy = s, (x, y)
    return bxy


def label_strip(w, text):
    strip = Image.new("RGB", (w, LABEL_H), (24, 24, 24))
    d = ImageDraw.Draw(strip)
    d.text((6, 5), text, fill=(240, 240, 240))
    return strip


def row(images, labels):
    w = sum(i.width for i in images) + PAD * (len(images) - 1)
    h = max(i.height for i in images) + LABEL_H
    canvas = Image.new("RGB", (w, h), (12, 12, 12))
    x = 0
    for img, lab in zip(images, labels):
        canvas.paste(label_strip(img.width, lab), (x, 0))
        canvas.paste(img, (x, LABEL_H))
        x += img.width + PAD
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[], help="LABEL=path:arch")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--frames", default=None,
                    help="comma-separated frame filenames to render instead "
                         "of an even sample. Feed it the actionable list "
                         "from analyze_outliers.py so the frames we LOSE on "
                         "get looked at, rather than an arbitrary sample "
                         "that may contain none of them.")
    ap.add_argument("--out", default=f"{_RTUS_ROOT}/visual_ab")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = {}
    for spec in args.ckpt:
        label, rest = spec.split("=", 1)
        path, arch = rest.rsplit(":", 1)
        models[label] = build_model(arch, path)

    triples = []
    for root in VAL_SETS:
        for lr_p in sorted(glob.glob(f"{root}/val/lr/*.png")):
            b = os.path.basename(lr_p)
            gt_p, t_p = f"{root}/val/hr/{b}", f"{root}_teacher/val/hr/{b}"
            if os.path.exists(gt_p) and os.path.exists(t_p):
                triples.append((lr_p, gt_p, t_p))
    if args.frames:
        want = {f.strip() for f in args.frames.split(",") if f.strip()}
        triples = [t for t in triples if os.path.basename(t[0]) in want]
        missing = want - {os.path.basename(t[0]) for t in triples}
        if missing:
            print(f"WARNING: not found in val sets: {sorted(missing)}",
                  flush=True)
    else:
        triples = triples[:: max(1, len(triples) // args.n)][:args.n]

    for lr_p, gt_p, t_p in triples:
        name = os.path.basename(lr_p)
        lr = Image.open(lr_p).convert("RGB")
        gt = Image.open(gt_p).convert("RGB")

        cols = [("original (GT)", gt),
                ("bicubic", lr.resize(gt.size, Image.BICUBIC))]
        x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0
                             ).permute(2, 0, 1)[None].to(DEV)
        for label, model in models.items():
            with torch.inference_mode():
                y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            cols.append((label, Image.fromarray((y * 255).round()
                                                .astype(np.uint8))))
        cols.append(("web model (teacher)", Image.open(t_p).convert("RGB")))

        labels = [c[0] for c in cols]
        full = row([c[1] for c in cols], labels)

        cx, cy = busiest_tile(gt)
        crops = [c[1].crop((cx, cy, cx + TILE, cy + TILE))
                 .resize((TILE * ZOOM, TILE * ZOOM), Image.NEAREST)
                 for c in cols]
        zoom = row(crops, [f"{l}  [{ZOOM}x detail]" for l in labels])

        out = Image.new("RGB", (max(full.width, zoom.width),
                                full.height + zoom.height + PAD), (12, 12, 12))
        out.paste(full, (0, 0))
        out.paste(zoom, (0, full.height + PAD))
        out.save(f"{args.out}/{name}")
        print(f"wrote {args.out}/{name}", flush=True)


if __name__ == "__main__":
    main()
