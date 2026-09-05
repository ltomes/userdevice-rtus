#!/usr/bin/env python3
"""SCRFD-2.5G ONNX output decoder, used by the face gate.

Runs `scrfd_2.5g_bnkps.onnx` through onnxruntime, decodes its 9 output
tensors into bounding boxes and 5-point landmarks, and applies NMS. The
anchor generation and decoding maths are documented inline below, because
the model's raw outputs are stride-major and not self-describing.

`face_gate` imports `preprocess`, `decode_scrfd`, `nms`, `unscale`,
`INPUT_SIZE` and `NMS_THRESH` from here. Running this module directly is a
correctness aid: it annotates one image so the decode can be eyeballed.

CPU inference is deliberate — this exercises decoder correctness, not
throughput, and the gate runs over tens of images rather than a video stream.

    python -m userdevice_rtus.tools.facegate.scrfd_decode \
        --onnx <pretrained>/scrfd_2.5g_bnkps.onnx \
        --image <frame.jpg> --out /tmp/scrfd_result.jpg

Fetch the detector with `scripts/fetch_assets.sh`.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── SCRFD constants ────────────────────────────────────────────────────────────

INPUT_SIZE = 640          # network expects 640×640
STRIDES    = [8, 16, 32]  # feature pyramid strides
NUM_ANCHORS = 2           # anchors per grid cell
SCORE_THRESH = 0.5        # confidence threshold before NMS
NMS_THRESH   = 0.45       # IoU threshold for NMS

# Preprocessing: (pixel_value - MEAN) / STD
# Applied per-channel after converting BGR → RGB.
# Source: InsightFace SCRFD python-package/insightface/model_zoo/scrfd.py
MEAN = 127.5
STD  = 128.0


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img: Image.Image) -> tuple[np.ndarray, float, int, int]:
    """Resize + normalise image to 1×3×640×640 float32 NCHW.

    Returns:
        blob      : (1, 3, 640, 640) float32 in [-1, 1]
        scale     : resize scale factor (original / INPUT_SIZE)
        pad_left  : horizontal padding added
        pad_top   : vertical padding added
    """
    orig_w, orig_h = img.size
    scale = INPUT_SIZE / max(orig_w, orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    resized = img.resize((new_w, new_h), Image.BILINEAR).convert("RGB")

    # Letterbox pad to 640×640 with grey (127.5)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (127, 127, 127))
    pad_left = (INPUT_SIZE - new_w) // 2
    pad_top  = (INPUT_SIZE - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    arr = np.asarray(canvas, dtype=np.float32)   # H×W×C, RGB, [0,255]
    arr = (arr - MEAN) / STD                      # [-1, 1]
    blob = arr.transpose(2, 0, 1)[np.newaxis]     # 1×C×H×W
    return blob, scale, pad_left, pad_top


# ── Anchor generation ──────────────────────────────────────────────────────────

def generate_anchors(stride: int, map_h: int, map_w: int) -> np.ndarray:
    """Return anchor centres shape (map_h*map_w*NUM_ANCHORS, 2) [cx, cy] in px."""
    out = []
    for row in range(map_h):
        for col in range(map_w):
            cx = col * stride
            cy = row * stride
            for _ in range(NUM_ANCHORS):
                out.append([cx, cy])
    return np.array(out, dtype=np.float32)


# ── Output decoding ────────────────────────────────────────────────────────────

def decode_scrfd(outputs: list[np.ndarray],
                 score_thresh: float = SCORE_THRESH,
                 nms_thresh: float = NMS_THRESH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode SCRFD 9-output tensors into bboxes, scores, keypoints.

    SCRFD output order (3 strides × 3 tensors each):
        stride 8:  score [1,H8*W8*2,1], bbox [1,H8*W8*2,4], kps [1,H8*W8*2,10]
        stride 16: score [1,H16*W16*2,1], ...
        stride 32: ...

    The outputs list from onnxruntime may be in arbitrary order; we sort by
    the number of anchor positions to assign strides correctly.

    Returns:
        bboxes  : (N, 4)  [x1, y1, x2, y2] in input-image pixels
        scores  : (N,)
        keypts  : (N, 5, 2)  [[kx, ky], ...] × 5 landmarks
    """
    # Group outputs by tensor element count to identify stride groups.
    # counts: 12800 → stride 8, 3200 → stride 16, 800 → stride 32
    stride_map = {
        INPUT_SIZE // 8  * INPUT_SIZE // 8  * NUM_ANCHORS: 8,
        INPUT_SIZE // 16 * INPUT_SIZE // 16 * NUM_ANCHORS: 16,
        INPUT_SIZE // 32 * INPUT_SIZE // 32 * NUM_ANCHORS: 32,
    }

    # Separate into score / bbox / kps by shape heuristic:
    #   score → last dim == 1
    #   kps   → last dim == 10
    #   bbox  → last dim == 4
    groups: dict[int, dict[str, np.ndarray]] = {}
    for out in outputs:
        # Outputs are (N, C) — no batch dim in this ONNX export.
        n  = out.shape[0]    # number of anchors
        ld = out.shape[-1]   # last dim identifies score/bbox/kps
        stride = stride_map.get(n)
        if stride is None:
            continue
        key = {1: "score", 4: "bbox", 10: "kps"}.get(ld)
        if key is None:
            continue
        if stride not in groups:
            groups[stride] = {}
        groups[stride][key] = out   # already (N, C)

    all_bboxes, all_scores, all_kps = [], [], []

    for stride in STRIDES:
        g = groups.get(stride, {})
        if len(g) < 3:
            continue

        score_raw = g["score"].squeeze(-1)      # (N,)
        bbox_raw  = g["bbox"]                    # (N, 4)
        kps_raw   = g["kps"]                     # (N, 10)

        # SCRFD exports post-sigmoid scores; apply sigmoid defensively if >1.
        if score_raw.max() > 1.01:
            score_raw = 1.0 / (1.0 + np.exp(-score_raw))

        map_side = INPUT_SIZE // stride
        anchors  = generate_anchors(stride, map_side, map_side)  # (N, 2) cx,cy

        # Bbox decode: centre_x/y ± distance_left/top/right/bottom × stride
        x1 = anchors[:, 0] - bbox_raw[:, 0] * stride
        y1 = anchors[:, 1] - bbox_raw[:, 1] * stride
        x2 = anchors[:, 0] + bbox_raw[:, 2] * stride
        y2 = anchors[:, 1] + bbox_raw[:, 3] * stride
        bboxes = np.stack([x1, y1, x2, y2], axis=1)

        # Keypoint decode: anchor_cx/y + offset × stride per landmark
        kps = np.reshape(kps_raw, (-1, 5, 2))
        kps[:, :, 0] = anchors[:, 0:1] + kps[:, :, 0] * stride
        kps[:, :, 1] = anchors[:, 1:2] + kps[:, :, 1] * stride

        mask = score_raw > score_thresh
        all_bboxes.append(bboxes[mask])
        all_scores.append(score_raw[mask])
        all_kps.append(kps[mask])

    if not all_bboxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 5, 2))

    bboxes = np.concatenate(all_bboxes)
    scores = np.concatenate(all_scores)
    kps    = np.concatenate(all_kps)

    keep = nms(bboxes, scores, nms_thresh)
    return bboxes[keep], scores[keep], kps[keep]


# ── NMS ───────────────────────────────────────────────────────────────────────

def nms(bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Standard IoU-based NMS. Returns indices to keep, sorted by score desc."""
    if len(bboxes) == 0:
        return []
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


# ── Coordinate mapping back to original image ─────────────────────────────────

def unscale(coords_xy: np.ndarray, scale: float, pad_l: int, pad_t: int) -> np.ndarray:
    """Map coordinates from 640×640 letterbox space to original image space."""
    out = coords_xy.copy().astype(np.float32)
    out[..., 0] = (out[..., 0] - pad_l) / scale
    out[..., 1] = (out[..., 1] - pad_t) / scale
    return out


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_results(img: Image.Image,
                 bboxes: np.ndarray, scores: np.ndarray,
                 kps: np.ndarray,
                 scale: float, pad_l: int, pad_t: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i, (box, score) in enumerate(zip(bboxes, scores)):
        # Unscale each corner independently from 640×640 letterbox to original.
        x1 = int((box[0] - pad_l) / scale)
        y1 = int((box[1] - pad_t) / scale)
        x2 = int((box[2] - pad_l) / scale)
        y2 = int((box[3] - pad_t) / scale)
        x1, x2 = min(x1, x2), max(x1, x2)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1, max(0, y1 - 14)), f"{score:.2f}", fill=(0, 255, 0))
        # Landmarks: 5 points per face
        kp = unscale(kps[i], scale, pad_l, pad_t)  # (5, 2)
        colours = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 165, 0), (0, 255, 255)]
        for j, (kx, ky) in enumerate(kp):
            r = 3
            draw.ellipse([kx-r, ky-r, kx+r, ky+r], fill=colours[j])
    return img


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx",  required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--out",   required=True, type=Path)
    ap.add_argument("--score-thresh", type=float, default=SCORE_THRESH)
    ap.add_argument("--nms-thresh",   type=float, default=NMS_THRESH)
    args = ap.parse_args()

    import onnxruntime as ort

    score_thresh = args.score_thresh
    nms_thresh   = args.nms_thresh
    sess = ort.InferenceSession(str(args.onnx),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"Input:  {in_name}")
    print(f"Outputs ({len(out_names)}): {out_names}")

    img = Image.open(args.image).convert("RGB")
    print(f"Image:  {img.size[0]}×{img.size[1]}")
    blob, scale, pad_l, pad_t = preprocess(img)
    print(f"Blob:   {blob.shape}  scale={scale:.4f}  pad=({pad_l},{pad_t})")

    outputs = sess.run(out_names, {in_name: blob})
    for name, out in zip(out_names, outputs):
        print(f"  {name}: {out.shape}  range=[{out.min():.3f}, {out.max():.3f}]")

    bboxes, scores, kps = decode_scrfd(outputs, score_thresh, nms_thresh)
    print(f"Detections: {len(bboxes)}")
    for i, (b, s) in enumerate(zip(bboxes, scores)):
        x1 = int((b[0] - pad_l) / scale)
        y1 = int((b[1] - pad_t) / scale)
        x2 = int((b[2] - pad_l) / scale)
        y2 = int((b[3] - pad_t) / scale)
        print(f"  [{i}] score={s:.3f}  box=({min(x1,x2)},{min(y1,y2)},{max(x1,x2)},{max(y1,y2)})")

    annotated = draw_results(img.copy(), bboxes, scores, kps, scale, pad_l, pad_t)
    annotated.save(args.out)
    print(f"Saved:  {args.out}")


if __name__ == "__main__":
    main()
