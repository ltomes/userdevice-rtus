# userDevice RTUS — real-time 2x upscaling

**This model was *trained* for real time, not made fast afterwards.**

Plenty of super-resolution models end up running quickly. They get there
after the fact — quantised, pruned, distilled, or handed to TensorRT once the
weights already exist. The frame budget is something the deployment engineer
inherits and fights.

Here the budget came first and shaped what was trained. The architecture was
chosen by ablating a teacher to find which of its parts survive at speed. The
model is split into two tiers because 1080p and 540p inputs leave different
amounts of time. The loss recipe was selected under a latency ceiling, and
every training run was scored against one. **A quality gain that misses the
frame budget was never a gain here — it was a rejected arm.**

That is the difference this repository is about. Most models optimise a
quality metric and report no frame time at all; this one treats the deadline
as the constraint and spends quality inside it.

## Target hardware and measured throughput

Target: **NVIDIA Jetson Thor class**, TensorRT FP16.

Measured on Thor — TensorRT 10.13.3, FP16, `builderOptimizationLevel=5`,
CUDA-graph enabled, median `enqueueV3` GPU time, compute-only:

| Input | Output | Latency (median) | **Throughput** | Verdict |
|---|---|---|---|---|
| 540p | 1080p | 6.335 ms | **158 fps** | passes a 120fps budget |
| 720p | 1440p | 12.350 ms | **81 fps** | passes a 60fps budget |
| 1080p | 4K | 28.742 ms | **35 fps** | fits the 38 ms budget for 24fps |

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
with a per-pixel EA gate on the residual branch of every backbone block — a
design derived from ablating the teacher, which showed its quality lives in
EA gating and collective depth, not in its 17px large kernels. So the student
keeps the cheap fast backbone and buys depth over width. That ablation is in
`eval/ablate_teacher.py`; it is the reason this architecture looks the way
it does.

## What it was NOT trained on

**No film, television, or private media.** The corpus is public photographs
(the NomosRealWeb release) with *synthesised* motion and *real* H.264
degradation. The `_film` in the config names denotes a latency tier — the
38 ms budget inside the 41.7 ms of a 24fps frame — not content. See `data/README.md`.

## Quality is gated, not scored

`eval/` holds the verdict harness. A checkpoint is not accepted because a
number went up:

1. **Sweep** — PSNR/SSIM/DISTS/LPIPS across every checkpoint, selecting
   best-by-DISTS rather than the last iteration. The last checkpoint is
   frequently not the best one.
2. **Face gate** — **blocking**, zero tolerance. Faces are where invented
   detail is most visible and least forgivable.
3. **Invention probe** — measures how much high-frequency detail the student
   *invented* rather than recovered, and kills any arm that invents more than
   the teacher it distils from. A model that wins on perceptual metrics by
   hallucinating has not won.

A fourth leg, temporal flicker on adjacent frames, is part of the project's
verdict process but **its implementation is not in this repository yet**.

See `docs/BENCHMARKS.md`, which also covers why standard bicubic-degraded
benchmark sets are out-of-distribution for a codec-trained model.

## Layout

    src/userdevice_rtus/     architectures (student, backbone, teacher-side)
    configs/                  traiNNer-redux configs, per stage and tier
    scripts/                  data generation, teacher targets, ONNX export
    eval/                     the verdict harness (sweep, face gate, probe)
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

## Licence and attribution

Code in this repository is **MIT** (see `LICENSE`).

The released models are **distilled from `4xNomosWebPhoto_RealPLKSR` by
Philip Hofmann**, licensed CC BY 4.0. That attribution is required by the
teacher's licence and carries to the weights, not just to this source tree.

Third-party notices for RTMoSR (MIT), neosr and traiNNer-redux (Apache-2.0)
are in `LICENSE`; the full provenance record, including the one open item, is
in `docs/PROVENANCE.md`.
