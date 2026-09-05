"""Dataset v2: motion-clip pairs with REAL P/B-frame codec artifacts.

Per source photo (512px HR):
  1. Generate N HR frames by sub-pixel drifting/zooming a 480x480 window
     (PIL affine, bicubic) — HR motion FIRST, so the LR = downscale(HR)
     contract is pixel-exact (fixes the flaw in the naive zoompan recipe:
     motion applied on the LR path breaks the pairing).
  2. Downscale each HR frame /2 (random kernel per clip) -> 240x240 LR.
  3. Pipe LR frames into libx264: sampled CRF 18-34, GOP {24,48} with
     scenecut=0, sampled VBV cap, yuv420p. NO synthetic noise stage —
     grain-preservation rule: HR and pre-encode LR carry the same
     (photo-native) noise; only the codec degrades.
  4. Decode, harvest K frames from [GOP/4, GOP-1] (P/B frames carrying
     reference drift + inter deblocking), pair with the stored HR frames.
Filenames are prefixed v2_<src>_t<idx> so v1 and v2 can be mixed in one
traiNNer dataroot list without collisions. Every 20th source -> val.

Runs in the container image (has ffmpeg now). Sequential clips, bounded
tmp; the container is memory/cpu-capped by the caller.
"""
import glob
import os
import random
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

SRC = "/workspace/datasets/nomosrealweb/hr"
OUT = "/workspace/datasets/greyduck2x_v2"
TMP = "/tmp/clip"
WIN, N_HARVEST = 480, 6
KERNELS = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS, Image.BOX]

random.seed(1024)


def hr_frames(img, n, motion):
    """Sub-pixel window track over the source image."""
    w, h = img.size
    max_off = (w - WIN) / 2 - 1
    cx0, cy0 = (w - WIN) / 2, (h - WIN) / 2
    if motion == "pan":
        ang = random.uniform(0, 2 * np.pi)
        speed = random.uniform(0.1, max_off / n)  # px/frame, subpixel
        dx, dy = np.cos(ang) * speed, np.sin(ang) * speed
        zooms = [1.0] * n
        xs = [cx0 + dx * i for i in range(n)]
        ys = [cy0 + dy * i for i in range(n)]
    else:  # zoom (in or out) with slight drift
        z0, z1 = (1.0, random.uniform(1.03, 1.08))
        if random.random() < 0.5:
            z0, z1 = z1, z0
        zooms = list(np.linspace(z0, z1, n))
        xs = [cx0 + random.uniform(-0.3, 0.3) * i for i in range(n)]
        ys = [cy0 + random.uniform(-0.3, 0.3) * i for i in range(n)]
    frames = []
    for i in range(n):
        z = zooms[i]
        # affine: output pixel (u,v) samples source at (x0 + u/z, y0 + v/z)
        a, b, c = 1 / z, 0.0, xs[i]
        d, e, f = 0.0, 1 / z, ys[i]
        fr = img.transform((WIN, WIN), Image.AFFINE, (a, b, c, d, e, f),
                           resample=Image.BICUBIC)
        frames.append(fr)
    return frames


def process(src_path, split):
    name = os.path.splitext(os.path.basename(src_path))[0]
    img = Image.open(src_path).convert("RGB")
    if img.size[0] < WIN + 8 or img.size[1] < WIN + 8:
        return 0
    gop = random.choice([24, 48])
    n = gop  # one GOP per clip
    crf = random.choices([random.randint(18, 23), random.randint(24, 29),
                          random.randint(30, 34)], weights=[2, 5, 3])[0]
    vbv = random.choice([0, 1500, 3000, 6000])  # kbps; 0 = uncapped
    kern = random.choice(KERNELS)
    motion = random.choice(["pan", "pan", "zoom"])

    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    hrs = hr_frames(img, n, motion)
    for i, fr in enumerate(hrs):
        fr.resize((WIN // 2, WIN // 2), kern).save(f"{TMP}/lr_{i:03d}.png")

    x264 = f"keyint={gop}:min-keyint={gop}:scenecut=0"
    if vbv:
        x264 += f":vbv-maxrate={vbv}:vbv-bufsize={vbv}"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-framerate", "24", "-i", f"{TMP}/lr_%03d.png",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "medium",
         "-crf", str(crf), "-x264-params", x264, "-threads", "2",
         f"{TMP}/clip.mp4"], check=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", f"{TMP}/clip.mp4", "-threads", "2", f"{TMP}/dec_%03d.png"],
        check=True)

    picks = sorted(random.sample(range(gop // 4, gop), N_HARVEST))
    wrote = 0
    for t in picks:
        dec = f"{TMP}/dec_{t+1:03d}.png"  # ffmpeg output is 1-indexed
        if not os.path.exists(dec):
            continue
        base = f"v2_{name}_t{t:02d}.png"
        hrs[t].save(f"{OUT}/{split}/hr/{base}")
        shutil.copy(dec, f"{OUT}/{split}/lr/{base}")
        wrote += 1
    return wrote


def main():
    # args: [limit] [worker_id num_workers] — stride partitioning lets N
    # workers run concurrently with zero filename overlap (disjoint sources)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    wid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for sp in ("train", "val"):
        os.makedirs(f"{OUT}/{sp}/hr", exist_ok=True)
        os.makedirs(f"{OUT}/{sp}/lr", exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(SRC, "*.png")))
    total = 0
    for i, p in enumerate(srcs):
        if limit and i >= limit:
            break
        if i % nw != wid:
            continue
        split = "val" if i % 20 == 0 else "train"
        name = os.path.splitext(os.path.basename(p))[0]
        if glob.glob(f"{OUT}/{split}/hr/v2_{name}_t*.png"):
            continue  # idempotent resume
        try:
            total += process(p, split)
        except subprocess.CalledProcessError as e:
            print(f"SKIP {name}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(srcs)}] pairs so far: {total}", flush=True)
    print(f"v2 complete: train={len(os.listdir(f'{OUT}/train/lr'))} "
          f"val={len(os.listdir(f'{OUT}/val/lr'))}", flush=True)


if __name__ == "__main__":
    main()
