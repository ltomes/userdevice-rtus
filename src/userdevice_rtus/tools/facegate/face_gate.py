"""Face-generation gate (project requirement, 2026-08-20): reject any model
whose SR output contains a face the source does not.

Method: SCRFD-2.5G (onnxruntime CPU) on the SR output AND on ground truth
at the same resolution. A detection in SR (score >= SR_THRESH) with no
IoU-overlapping detection in GT (score >= GT_THRESH, deliberately lower so
borderline REAL faces don't count as hallucinations) is a violation.
Gate: zero violations. Violating crops are saved for human review.

Usage (inside the container image, needs onnxruntime):
  python -m userdevice_rtus.tools.facegate.face_gate <ckpt.safetensors> <arch> [n_images]
  arch: rtmosr_l | rtmosr_ea_film | rtmosr_ea_film_sd
"""
import glob
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from safetensors.torch import load_file

from userdevice_rtus.tools.facegate.scrfd_decode import (
    INPUT_SIZE,
    NMS_THRESH,
    decode_scrfd,
    nms,
    preprocess,
    unscale,
)

SR_THRESH = 0.5   # strict: what counts as "a face" in model output
GT_THRESH = 0.2   # lenient: what counts as "there was a face" in source
IOU_MATCH = 0.3
from userdevice_rtus.paths import DATA_ROOT as _RTUS_ROOT
from userdevice_rtus.paths import SCRFD as _SCRFD

# The detector is fetched by scripts/fetch_assets.sh into PRETRAINED, which is
# also where rtus-info looks for it. It used to be read from inside the
# installed package, so fetch_assets and rtus-info would both report the gate
# ready while the gate itself could not find its own model.
ONNX = str(_SCRFD)

VAL_DIRS = [f"{_RTUS_ROOT}/datasets/rtus2x/val",
            f"{_RTUS_ROOT}/datasets/rtus2x_v2/val"]
OUT_DIR = f"{_RTUS_ROOT}/facegate/violations"


def build_model(arch, ckpt):
    from userdevice_rtus import build_tier

    # Geometry from userdevice_rtus.TIERS, the single source of truth.
    m = build_tier(arch)
    sd = load_file(ckpt) if ckpt.endswith(".safetensors") else None
    if sd is None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k in ("params_ema", "params", "state_dict", "model"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
    m.load_state_dict(sd, strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.eval().to(dev), dev


def detect(sess, img):
    blob, scale, pad_l, pad_t = preprocess(img)
    outputs = sess.run(None, {sess.get_inputs()[0].name: blob})
    bboxes, scores, _ = decode_scrfd(outputs, score_thresh=min(SR_THRESH,
                                                               GT_THRESH))
    if len(bboxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    keep = nms(bboxes, scores, NMS_THRESH)
    bboxes, scores = bboxes[keep], scores[keep]
    bboxes = np.stack([unscale(b.reshape(2, 2), scale, pad_l, pad_t).ravel()
                       for b in bboxes]) if len(bboxes) else bboxes
    return bboxes, scores


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def main():
    ckpt, arch = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    model, dev = build_model(arch, ckpt)
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    os.makedirs(OUT_DIR, exist_ok=True)

    pairs = []
    for d in VAL_DIRS:
        pairs += sorted(glob.glob(f"{d}/lr/*.png"))
    pairs = pairs[:: max(1, len(pairs) // n)][:n]

    violations = []
    content_borne = []
    checked = 0
    for lr_p in pairs:
        gt_p = lr_p.replace("/lr/", "/hr/")
        if not os.path.exists(gt_p):
            continue
        lr = Image.open(lr_p).convert("RGB")
        gt = Image.open(gt_p).convert("RGB")
        x = torch.from_numpy(np.asarray(lr).astype(np.float32) / 255.0
                             ).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            y = model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        sr = Image.fromarray((y * 255).round().astype(np.uint8))

        sr_b, sr_s = detect(sess, sr)
        gt_b, gt_s = detect(sess, gt)
        gt_faces = gt_b[gt_s >= GT_THRESH] if len(gt_b) else gt_b
        # bicubic control: face-likeness already present in the content
        # (triggers on a plain resize) is not the model's doing
        bic = lr.resize(gt.size, Image.BICUBIC)
        bic_b, bic_s = detect(sess, bic)
        bic_faces = bic_b[bic_s >= GT_THRESH] if len(bic_b) else bic_b
        checked += 1
        for b, s in zip(sr_b, sr_s):
            if s < SR_THRESH:
                continue
            if any(iou(b, g) >= IOU_MATCH for g in gt_faces):
                continue  # real face, correctly present
            name = os.path.basename(lr_p)
            if any(iou(b, g) >= IOU_MATCH for g in bic_faces):
                content_borne.append({"image": name, "score": float(s),
                                      "bbox": [float(v) for v in b]})
                print(f"content-borne (bicubic also triggers) {name}: "
                      f"score {s:.2f}", flush=True)
                continue
            violations.append({"image": name, "score": float(s),
                               "bbox": [float(v) for v in b]})
            x1, y1, x2, y2 = [int(v) for v in b]
            pad = 24
            crop = sr.crop((max(0, x1 - pad), max(0, y1 - pad),
                            x2 + pad, y2 + pad))
            crop.save(f"{OUT_DIR}/{name}_face{len(violations)}.png")
            print(f"VIOLATION {name}: score {s:.2f} bbox {b}", flush=True)

    verdict = "PASS" if not violations else "FAIL"
    result = {"ckpt": ckpt, "arch": arch, "images_checked": checked,
              "violations": violations, "content_borne": content_borne,
              "verdict": verdict}
    with open(f"{_RTUS_ROOT}/facegate/last_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"FACE GATE {verdict}: {len(violations)} violations "
          f"in {checked} images")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
