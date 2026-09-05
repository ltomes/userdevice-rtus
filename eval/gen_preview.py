"""Side-by-side val previews: bicubic | pretrained RTMoSR-L | smoke-2k | HR.

Runs on ms in greyduck-train:dev (GPU shared with the training container —
tiny inference, memory-capped). Outputs labeled JPEG strips + zoom crops to
preview/ for remote (artifact) viewing.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from rtmosr_vendored import RTMoSR

BASE = "/workspace"
OUT = os.path.join(BASE, "preview")
os.makedirs(OUT, exist_ok=True)

CFG = dict(scale=2, dim=32, ffn_expansion=2, n_blocks=2,
           unshuffle_mod=True, dccm=True, se=True)

# sample 6 evenly across the val set
_val = sorted(glob.glob("/workspace/datasets/greyduck2x/val/hr/*.png"))
PICKS = [os.path.basename(_val[i])[:-4]
         for i in range(0, len(_val), max(1, len(_val) // 6))][:6]


def load_model(path):
    m = RTMoSR(**CFG)
    if path.endswith(".safetensors"):
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        for k in ("params_ema", "params", "state_dict", "model"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
    m.load_state_dict(sd, strict=True)
    return m.eval().cuda()


def infer(model, lr_img):
    x = torch.from_numpy(np.asarray(lr_img).astype(np.float32) / 255.0
                         ).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        y = model(x)
    y = y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255).round().astype(np.uint8))


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return img


pre = load_model(os.path.join(BASE, "pretrained/2x_RTMoSR_L.pth"))
stageb = load_model(os.path.join(
    BASE, "experiments/2x_RTMoSR_L_greyduck_stageB/models/net_g_ema_50000.safetensors"))

for pick in PICKS:
    hits = glob.glob(os.path.join(BASE, f"datasets/greyduck2x/val/hr/{pick}.png"))
    if not hits:
        continue
    hr = Image.open(hits[0]).convert("RGB")
    lr = Image.open(hits[0].replace("/hr/", "/lr/")).convert("RGB")
    W, H = hr.size

    hr_np = np.asarray(hr).astype(np.float64) / 255.0

    def psnr(img):
        a = np.asarray(img).astype(np.float64) / 255.0
        mse = np.mean((a - hr_np) ** 2)
        return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)

    bic = lr.resize((W, H), Image.BICUBIC)
    p_pre = infer(pre, lr)
    p_sb = infer(stageb, lr)
    panels = [
        label(bic, f"bicubic {psnr(bic):.2f}dB"),
        label(p_pre, f"RTMoSR-L pretrained {psnr(p_pre):.2f}dB"),
        label(p_sb, f"greyduck stage-B 50k {psnr(p_sb):.2f}dB"),
        label(hr.copy(), "ground truth"),
    ]
    strip = Image.new("RGB", (W * 4 + 6, H), (20, 20, 20))
    for i, p in enumerate(panels):
        strip.paste(p, (i * (W + 2), 0))
    strip.save(os.path.join(OUT, f"val{pick}_full.jpg"), quality=88)

    # 2x zoom crop at the most detailed GT region (max local variance),
    # not blind center — center often lands on flat/dark areas
    cs = 128
    gray = np.asarray(hr.convert("L")).astype(np.float64)
    best, cx, cy = -1.0, W // 2, H // 2
    for yy in range(cs // 2, H - cs // 2 + 1, 64):
        for xx in range(cs // 2, W - cs // 2 + 1, 64):
            v = gray[yy - cs // 2:yy + cs // 2, xx - cs // 2:xx + cs // 2].var()
            if v > best:
                best, cx, cy = v, xx, yy
    zoom = Image.new("RGB", (cs * 2 * 4 + 6, cs * 2), (20, 20, 20))
    for i, p in enumerate(panels):
        crop = p.crop((cx - cs // 2, cy - cs // 2, cx + cs // 2, cy + cs // 2))
        zoom.paste(crop.resize((cs * 2, cs * 2), Image.NEAREST), (i * (cs * 2 + 2), 0))
    zoom.save(os.path.join(OUT, f"val{pick}_zoom.jpg"), quality=88)
    print(f"done {pick}", flush=True)
print("preview complete")
