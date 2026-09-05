"""TEMPORAL evaluation of the wave-1 candidates. This should have been in wave 1.

The film tier is a per-frame model applied to VIDEO. Wave 1 ranked seven probes
purely on still-frame DISTS/LPIPS/PSNR, which cannot see frame-to-frame
instability at all. A GAN-weighted model resolves ambiguity differently on
consecutive frames; on moving film that reads as shimmer and is more
objectionable than a soft frame.

METRIC. The val set contains ADJACENT frames of the same clip
(v2_<clip>_t<frame>.png, e.g. t08/t09). For each adjacent pair we compare how
the model's output CHANGES between frames against how the ground truth changes:

    flicker = mean | (SR_t1 - SR_t0) - (GT_t1 - GT_t0) |

Real motion appears in both terms and cancels. What is left is temporal change
the model invented. Lower is better; 0 means the model's frame-to-frame
behaviour exactly tracks the truth.

CONTROLS (both must behave sensibly or the metric is not trustworthy):
  - bicubic: a fixed linear operator. It CANNOT flicker, so it should score
    near the floor. If bicubic scores high the metric is broken.
  - gt_vs_gt: ground truth against itself = 0 by construction, a sanity zero.
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np, torch
from PIL import Image
sys.path.insert(0, "/ar"); sys.path.insert(0, "/workspace/traiNNer-redux")
from safetensors.torch import load_file
from rtmosr_ea_vendored import RTMoSREA

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "/workspace/datasets/greyduck2x_v2/val"
MAXPAIRS = int(os.environ.get("MAXPAIRS", "120"))

def build(arch):
    if arch == "rtmosr_ea_film":
        return RTMoSREA(scale=2, dim=48, ffn_expansion=2, n_blocks=3,
                        unshuffle_mod=True, dccm=True, se=True)
    if arch == "rtmosr_ea_film_sd":
        return RTMoSREA(scale=2, dim=64, ffn_expansion=2, n_blocks=6,
                        unshuffle_mod=True, dccm=True, se=True)
    raise ValueError(arch)

def load(path, arch):
    m = build(arch); m.load_state_dict(load_file(path), strict=True)
    return m.eval().to(DEV)

# adjacent frame pairs within a clip
clips = defaultdict(list)
for p in sorted(glob.glob(f"{ROOT}/hr/*.png")):
    b = os.path.basename(p)
    m = re.match(r"(v2_\d+)_t(\d+)\.png$", b)
    if m: clips[m.group(1)].append((int(m.group(2)), b))
pairs = []
for c, fs in clips.items():
    fs.sort()
    for i in range(len(fs) - 1):
        if fs[i+1][0] - fs[i][0] == 1:      # strictly adjacent only
            pairs.append((fs[i][1], fs[i+1][1]))
pairs = pairs[:MAXPAIRS]
print(f"adjacent frame pairs: {len(pairs)}", flush=True)

def t(u8):
    return torch.from_numpy(u8.transpose(2,0,1)[None].astype(np.float32)/255.0).to(DEV)

specs = json.load(open("/ar/temporal_specs.json"))
models = {k: load(v["path"], v["arch"]) for k, v in specs.items()}
print(f"loaded {len(models)} candidates", flush=True)

acc = defaultdict(list)
for n,(a,b) in enumerate(pairs):
    gt0 = np.asarray(Image.open(f"{ROOT}/hr/{a}").convert("RGB"), dtype=np.float64)
    gt1 = np.asarray(Image.open(f"{ROOT}/hr/{b}").convert("RGB"), dtype=np.float64)
    lr0 = np.asarray(Image.open(f"{ROOT}/lr/{a}").convert("RGB"))
    lr1 = np.asarray(Image.open(f"{ROOT}/lr/{b}").convert("RGB"))
    dgt = gt1 - gt0
    bi0 = np.asarray(Image.fromarray(lr0).resize((gt0.shape[1],gt0.shape[0]), Image.BICUBIC), dtype=np.float64)
    bi1 = np.asarray(Image.fromarray(lr1).resize((gt0.shape[1],gt0.shape[0]), Image.BICUBIC), dtype=np.float64)
    acc["bicubic"].append(float(np.mean(np.abs((bi1-bi0)-dgt))))
    acc["gt_control"].append(0.0)
    with torch.no_grad():
        for k, m in models.items():
            s0 = (m(t(lr0)).clamp(0,1)[0].cpu().numpy().transpose(1,2,0)*255.0).astype(np.float64)
            s1 = (m(t(lr1)).clamp(0,1)[0].cpu().numpy().transpose(1,2,0)*255.0).astype(np.float64)
            acc[k].append(float(np.mean(np.abs((s1-s0)-dgt))))
    if (n+1) % 40 == 0: print(f"  {n+1}/{len(pairs)}", flush=True)

print(f"\nTEMPORAL FLICKER (mean |d_SR - d_GT| over {len(pairs)} adjacent pairs; LOWER = steadier)\n")
order = sorted([k for k in acc if k not in ("gt_control",)], key=lambda k: np.mean(acc[k]))
base = np.mean(acc["bicubic"])
print(f"{'candidate':<14} {'flicker':>9} {'vs bicubic':>11}")
for k in order:
    v = np.mean(acc[k])
    print(f"{k:<14} {v:9.4f} {(v-base)/base*100:+10.1f}%")
print(f"\ncontrol gt_vs_gt = 0.0000 (sanity zero)")
json.dump({k: float(np.mean(v)) for k, v in acc.items()},
          open("/ar/results/2026-08-27-temporal.json","w"), indent=1)
print("wrote /ar/results/2026-08-27-temporal.json")
