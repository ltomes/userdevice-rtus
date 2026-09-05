# userDevice RTUS — real-time 2x upscaling

**The goal of this model is a frame rate, not a score.**

Most super-resolution models are optimised for PSNR or perceptual metrics
and report no frame time at all — they are built to make one image better,
offline, at whatever cost that takes. This one is built to run **inside a
live video pipeline on a specific piece of hardware**, and every design
decision below is downstream of that. A quality gain that misses the frame
budget is not a gain here; it is a regression.

## Target hardware and measured throughput

Target: **NVIDIA Jetson Thor class**, TensorRT FP16.

Measured on Thor — TensorRT 10.13.3, FP16, `builderOptimizationLevel=5`,
CUDA-graph enabled, median `enqueueV3` GPU time, compute-only:

| Input | Output | Latency (median) | **Throughput** | Verdict |
|---|---|---|---|---|
| 540p | 1080p | 6.335 ms | **158 fps** | passes a 120fps budget |
| 720p | 1440p | 12.350 ms | **81 fps** | passes a 60fps budget |
| 1080p | 4K | 28.742 ms | **35 fps** | fits the 38 ms 24fps film floor |

p95 and p99 sit within 0.2 ms of the median at every shape — the model is
not merely fast on average, it is *steady*, which is what a playback
pipeline actually needs. A model that averages 30 fps but stalls for 60 ms
once a second drops frames and is useless here.

Numbers are for the d48 tier: 9.72M params, TensorRT parity 75.51 dB
against the torch fp32 reference (>40 dB required).

> **Provenance of these figures:** measured on the stage-D2 checkpoint. The
> architecture is unchanged in later checkpoints, so the latency is expected
> to carry over, but the current release candidate has **not been re-timed**.
> Do not quote an fps number for a checkpoint that has not been measured.

## Two tiers, because the budget differs per resolution

| Tier | Params | Target input | Why |
|---|---|---|---|
| d48 | 9.72M | 1080p | the only tier that fits 1080p->4K inside the frame budget |
| d64 | 33.9M | <=720p | 18.51 ms at 540p, 79.49 ms at 1080p — too slow for 1080p, so it spends the headroom that lower resolutions leave over |

The d64 tier exists because at 540p the d48 model uses only 6.3 ms of a
~38 ms budget. That unused capacity is quality left on the table.

## How it is built

A distilled student. The teacher is `4xNomosWebPhoto_RealPLKSR` (Philip
Hofmann, CC-BY-4.0). The student is an RTMoSR-style small-kernel backbone
with a per-pixel EA gate on every block output — a design derived from
ablating the teacher, which showed its quality lives in EA gating and
collective depth, not in its 17px large kernels. So the student keeps the
cheap fast backbone and buys depth over width. That ablation is in
`eval/ablate_teacher.py`; it is the reason this architecture looks the way
it does.

## What it was NOT trained on

**No film, television, or private media.** The corpus is public photographs
(the NomosRealWeb release) with *synthesised* motion and *real* H.264
degradation. The `_film` in the config names denotes a latency tier — the
38 ms of a 24fps frame — not content. See `data/README.md`.

## Quality is measured on four legs, not one number

`eval/` holds the verdict harness: a checkpoint sweep on PSNR/SSIM/DISTS/
LPIPS picking best-by-DISTS rather than by last iteration, a **blocking**
face gate, a temporal flicker measurement, and an invention probe that kills
any student hallucinating more detail than the teacher it distils from. See
`docs/BENCHMARKS.md`, which also covers why standard bicubic-degraded
benchmark sets are out-of-distribution for a codec-trained model.

## Layout

    src/userdevice_upscale/   architectures (student, backbone, teacher-side)
    configs/                  traiNNer-redux configs, per stage and tier
    scripts/                  data generation, teacher targets, ONNX export
    eval/                     the four-leg verdict harness
    data/                     the data CONTRACT — never the data itself
    docs/                     provenance, benchmarks, naming

## Data and weights are never in this repository

Training data is regenerated from public sources by `scripts/`; checkpoints,
ONNX exports and TensorRT engines are build outputs. Both are gitignored.
Weights are published separately as a Hugging Face model repository, and a
release cites the commit it was trained from.

## Before releasing anything

Read `docs/PROVENANCE.md`. Every code dependency is permissive (MIT /
Apache-2.0) and the teacher is CC-BY-4.0, so the model is releasable — but
attribution to Philip Hofmann is **required**, not optional.
