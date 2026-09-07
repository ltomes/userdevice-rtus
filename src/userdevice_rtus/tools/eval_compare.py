"""Four-way perceptual + fidelity comparison on the REAL-GT val set.

Answers the question PSNR/SSIM alone cannot: stage-C trades fidelity for
texture on purpose, so a PSNR drop is expected and says nothing about
whether the trade was worth it. This adds DISTS and LPIPS (lower = better,
perceptual) alongside PSNR/SSIM (higher = better, fidelity), and scores
every candidate on identical frames:

  original  — real GT (the ceiling; scored against itself only as a sanity 0)
  bicubic   — degenerate-upscaler floor
  <ckpts>   — our students (stage-A fidelity base, stage-C textured)
  teacher   — 4xNomosWebPhoto_RealPLKSR, the web model we distill from

Metrics come from traiNNer's own registry implementations, so they are
directly comparable to the numbers printed during training. crop_border=2
and test_y_channel=True match the stage-C val config.

Usage (inside the container image, cwd /workspace):
  python -m userdevice_rtus.tools.eval_compare [--n 300] [--out results/<name>.json] \
      [--ckpt LABEL=path.safetensors:arch] ...
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from traiNNer.archs.lpips_arch import LPIPS
from traiNNer.losses.dists_loss import DISTSLoss
from traiNNer.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from traiNNer.utils.img_util import img2batchedtensor

from userdevice_rtus.paths import corpus_dirs

# Metrics come from traiNNer's own registry implementations so they stay
# directly comparable to the figures printed during training.
# Label with the RESOLVED directory name so a report says which corpus it
# actually scored, not which one the source hoped for.
VAL_SETS = [(str(d), d.name) for d in corpus_dirs()]
CROP_BORDER = 2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(arch, ckpt):
    from safetensors.torch import load_file

    from userdevice_rtus import build_tier

    # Geometry comes from userdevice_rtus.TIERS. It used to be spelled out
    # here and in four other tools with no canonical definition, which is how
    # a tier's shape drifts away from the checkpoints it has to load.
    m = build_tier(arch)
    m.load_state_dict(load_file(ckpt), strict=True)
    return m.eval().to(DEV)


def pairs(n):
    """(lr_path, gt_path, teacher_path) triples, evenly sampled."""
    out = []
    for root, _ in VAL_SETS:
        for lr_p in sorted(glob.glob(f"{root}/val/lr/*.png")):
            base = os.path.basename(lr_p)
            gt_p = f"{root}/val/hr/{base}"
            t_p = f"{root}_teacher/val/hr/{base}"
            if os.path.exists(gt_p):
                out.append((lr_p, gt_p, t_p))
    if n and n < len(out):
        out = out[:: max(1, len(out) // n)][:n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="LABEL=path:arch")
    ap.add_argument("--sweep", default=None,
                    help="EXPDIR:arch — score every net_g_ema_*.safetensors "
                         "in EXPDIR/models. The best perceptual checkpoint is "
                         "usually NOT the last one once a GAN is involved, so "
                         "shipping net_g_ema_<total_iter> by default throws "
                         "away quality we already paid for.")
    ap.add_argument("--per-image", default=None,
                    help="write per-frame scores here (JSON) for outlier "
                         "analysis — which frames each model handles worst")
    args = ap.parse_args()

    dists = DISTSLoss(loss_weight=1.0, as_loss=False).to(DEV).eval()
    lpips = LPIPS(net="alex").to(DEV).eval()

    ckpt_specs = list(args.ckpt)
    if args.sweep:
        expdir, arch = args.sweep.rsplit(":", 1)
        # net_g_ema_latest.safetensors carries no iteration number and is a
        # duplicate of the final numbered checkpoint — including it crashes
        # the numeric sort and would double-score the same weights.
        def iter_of(path):
            digits = "".join(c for c in os.path.basename(path) if c.isdigit())
            return int(digits) if digits else None

        found = [f for f in glob.glob(f"{expdir}/models/net_g_ema_*.safetensors")
                 if iter_of(f) is not None]
        found.sort(key=iter_of)
        for f in found:
            ckpt_specs.append(f"iter{iter_of(f):06d}={f}:{arch}")
        print(f"sweep: {len(found)} checkpoints from {expdir}", flush=True)

    models = {}
    for spec in ckpt_specs:
        label, rest = spec.split("=", 1)
        path, arch = rest.rsplit(":", 1)
        models[label] = build_model(arch, path)
        print(f"loaded {label}: {path} ({arch})", flush=True)

    triples = pairs(args.n)
    print(f"scoring {len(triples)} val frames on {DEV}", flush=True)

    acc = {}
    per_image = {}

    def add(label, sr_u8, gt_u8):
        """sr_u8/gt_u8 are uint8 [0,255].

        PSNR/SSIM take float64 [0,255]; DISTS/LPIPS go through
        img2batchedtensor, whose tensor2float32 only divides by 255 for
        INTEGER dtypes — handing it floats would clamp every pixel to 1.0
        and yield plausible-looking garbage. So uint8 in, deliberately.
        """
        m = acc.setdefault(label, {"psnr": [], "ssim": [],
                                   "dists": [], "lpips": [], "n": 0})
        srf = sr_u8.astype(np.float64)
        gtf = gt_u8.astype(np.float64)
        m["psnr"].append(calculate_psnr(srf, gtf, CROP_BORDER,
                                        test_y_channel=True))
        m["ssim"].append(calculate_ssim(srf, gtf, CROP_BORDER,
                                        test_y_channel=True))
        a = img2batchedtensor(sr_u8, DEV, from_bgr=False)
        b = img2batchedtensor(gt_u8, DEV, from_bgr=False)
        with torch.inference_mode():
            m["dists"].append(float(dists(a, b)))
            m["lpips"].append(float(lpips(a, b)))
        m["n"] += 1
        per_image.setdefault(cur_name[0], {})[label] = {
            "psnr": m["psnr"][-1], "ssim": m["ssim"][-1],
            "dists": m["dists"][-1], "lpips": m["lpips"][-1]}

    cur_name = [""]
    for i, (lr_p, gt_p, t_p) in enumerate(triples):
        cur_name[0] = os.path.basename(lr_p)
        lr = Image.open(lr_p).convert("RGB")
        gt_img = Image.open(gt_p).convert("RGB")
        gt = np.asarray(gt_img).astype(np.uint8)

        add("bicubic", np.asarray(lr.resize(gt_img.size, Image.BICUBIC)
                                  ).astype(np.uint8), gt)

        if os.path.exists(t_p):
            add("teacher_web", np.asarray(Image.open(t_p).convert("RGB")
                                          ).astype(np.uint8), gt)

        x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0
                             ).permute(2, 0, 1)[None].to(DEV)
        for label, model in models.items():
            with torch.inference_mode():
                y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            add(label, (y * 255).round().astype(np.uint8), gt)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(triples)}", flush=True)

    summary = {}
    for label, m in acc.items():
        summary[label] = {k: (sum(v) / len(v)) for k, v in m.items()
                          if k != "n"}
        summary[label]["n"] = m["n"]

    order = ["bicubic"] + [k for k in models] + ["teacher_web"]
    t = summary.get("teacher_web")

    def gap(label, m):
        """Percent gap to the teacher on a lower-is-better metric.

        This is the SHIP BAR: the student
        must land COMFORTABLY NEGATIVE — perceptually closer to ground truth
        than the web model, not merely level with it. Printed on every run so
        the bar is never something we have to recompute by hand.
        """
        if not t or label == "teacher_web" or m not in t:
            return "     —"
        return f"{(summary[label][m] - t[m]) / t[m] * 100:+6.1f}%"

    print("\n| candidate | PSNR up | SSIM up | DISTS down | LPIPS down "
          "| dDISTS vs teacher | dLPIPS vs teacher | n |")
    print("|---|---|---|---|---|---|---|---|")
    for label in order:
        if label not in summary:
            continue
        r = summary[label]
        print(f"| {label} | {r['psnr']:.4f} | {r['ssim']:.4f} | "
              f"{r['dists']:.4f} | {r['lpips']:.4f} | "
              f"{gap(label, 'dists')} | {gap(label, 'lpips')} | {r['n']} |")

    if t:
        cands = [k for k in order if k in summary and k != "teacher_web"
                 and k != "bicubic"]
        passing = [k for k in cands
                   if summary[k]["dists"] < t["dists"]
                   and summary[k]["lpips"] < t["lpips"]]
        print("\nSHIP BAR (both perceptual gaps negative vs teacher): "
              + (", ".join(passing) if passing else "NO CANDIDATE PASSES"))

    # Which checkpoint to actually ship. With a GAN in the loop the last
    # iter is rarely the best perceptually, and PSNR will disagree with
    # DISTS/LPIPS — so report both winners instead of silently picking one.
    swept = [k for k in summary if k.startswith("iter")]
    if swept:
        best_d = min(swept, key=lambda k: summary[k]["dists"])
        best_l = min(swept, key=lambda k: summary[k]["lpips"])
        best_p = max(swept, key=lambda k: summary[k]["psnr"])
        print(f"\nbest DISTS : {best_d} ({summary[best_d]['dists']:.4f})")
        print(f"best LPIPS : {best_l} ({summary[best_l]['lpips']:.4f})")
        print(f"best PSNR  : {best_p} ({summary[best_p]['psnr']:.4f})")
        if best_d != swept[-1]:
            print(f"NOTE: best perceptual ckpt is {best_d}, NOT the final "
                  f"{swept[-1]} — shipping the last checkpoint would lose "
                  f"quality.")

    if args.per_image:
        os.makedirs(os.path.dirname(args.per_image), exist_ok=True)
        with open(args.per_image, "w") as f:
            json.dump(per_image, f, indent=2)
        print(f"wrote {args.per_image}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
