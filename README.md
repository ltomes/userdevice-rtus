# userDevice RTUS — real-time 2x video upscaling

**RTUS** (Real Time Upscale) is a 2x super-resolution model trained under a
hard per-frame deadline: **38 ms** inside the 41.7 ms of a 24 fps frame, on
**NVIDIA Jetson Thor** with TensorRT FP16. It is trained with
[traiNNer-redux](https://github.com/the-database/traiNNer-redux) and
distilled from `4xNomosWebPhoto_RealPLKSR`.

## Why the deadline comes first

Most super-resolution models optimise a quality metric and report no frame
time at all; speed, if it arrives, is added afterwards by quantising, pruning,
distilling, or handing the finished weights to TensorRT. Here the budget
shaped what was trained:

- **The architecture** was chosen by ablating the teacher to find which of its
  mechanisms survive at speed (`userdevice_rtus.tools.ablate_teacher`).
- **Two tiers** exist because 1080p and 540p inputs leave different amounts of
  time in the same budget.
- **The loss recipe** was selected under the latency ceiling, and every
  training arm was scored against it. A quality gain that misses the frame
  budget was a rejected arm, not a gain.

## Two tiers

| Tier | Arch name | Params | Target input | Why |
|---|---|---|---|---|
| d48 | `rtmosr_ea_film` | 9.72M | 1080p | the only tier that fits 1080p→4K inside the budget |
| d64 | `rtmosr_ea_film_sd` | 33.9M | ≤720p | too slow for 1080p, so it spends the headroom that lower resolutions leave over |

Both parameter counts are asserted at image build time, so they cannot drift
from the code. Configs for the d64 tier carry `_sd` in the filename.

## Measured throughput (stage-D2 checkpoint)

> **Read this first.** These figures were measured on the **stage-D2**
> checkpoint of the d48 tier. The architecture is unchanged in later
> checkpoints, so the latency is expected to carry over, but the current
> release candidate has **not been re-timed**, and this repository contains no
> timing harness. Do not quote an fps number for a checkpoint that has not
> been measured.

Jetson Thor, TensorRT 10.13.3, FP16, `builderOptimizationLevel=5`, CUDA graph
enabled, median `enqueueV3` GPU time, compute-only:

| Input | Output | Latency (median) | Throughput | Verdict |
|---|---|---|---|---|
| 540p | 1080p | 6.335 ms | **158 fps** | passes a 120 fps budget |
| 720p | 1440p | 12.350 ms | **81 fps** | passes a 60 fps budget |
| 1080p | 4K | 28.742 ms | **35 fps** | fits the 38 ms budget for 24 fps |

p95 and p99 sat within 0.2 ms of the median at every shape. Steadiness is
what a playback pipeline needs: a model that averages 30 fps but stalls once a
second drops frames. TensorRT parity for this checkpoint was 75.51 dB against
the torch fp32 reference (>40 dB required).

The d64 tier exists because at 540p the d48 model uses about 6.3 ms of a
38 ms budget — capacity left on the table.

## How it is built

A distilled student. The teacher is `4xNomosWebPhoto_RealPLKSR` (Philip
Hofmann, CC-BY-4.0). The student is an RTMoSR-style small-kernel backbone with
a per-pixel EA gate on the residual branch of every backbone block. The
teacher ablation showed its quality lives in EA gating and collective depth,
not in its 17px large kernels, so the student keeps the cheap backbone and
buys depth over width.

The release line trains **from scratch**: stage A initialises from nothing and
every later stage initialises from the project's own stage-A checkpoint. No
third-party weights are in the lineage (see `docs/PROVENANCE.md`).

## Training data

The corpus is regenerated locally from public sources and is never committed.

- **HR** — the public NomosRealWeb release (`hr.tar`, 6000 × 512² Nomos-v2
  HR images).
- **LR** — generated from those HRs: synthesised sub-pixel motion on the HR
  path, downscale 1/2, then real libx264 encoding (CRF 18–34, GOP 24/48),
  harvesting P/B frames. No synthetic noise stage: grain is signal in the
  target domain, and only the codec degrades the LR.
- **Teacher targets** — the teacher run over the LR and downscaled 2x.

The `_film` in the config names denotes the **latency tier** (the 38 ms
budget), not the content: the model was not trained on film, television, or
any private media. Full contract and recipe rationale in `data/README.md`.

## Quality is gated, not scored

A checkpoint is accepted by passing four legs on identical frames, not
because a number went up. The harness lives in `userdevice_rtus.tools`:

1. **Sweep** (`eval_compare`) — PSNR/SSIM/DISTS/LPIPS across every
   checkpoint, against bicubic and the teacher, picking **best-by-DISTS**
   rather than the last iteration. The last checkpoint is frequently not the
   best one.
2. **Face gate** (`facegate.face_gate`) — **blocking**, zero tolerance. Faces
   are where invented detail is most visible and least forgivable.
3. **Invention probe** (`hallucination_probe`) — how much high-frequency
   detail the student *invented* rather than recovered. Any arm that invents
   more than the teacher is killed.
4. **Temporal** (`temporal_eval`) — flicker on adjacent frames of the same
   clip. Real motion cancels between output and ground-truth differences;
   what remains is temporal change the model invented. A per-frame model can
   shimmer on video while scoring well on every still-frame metric.

`docs/BENCHMARKS.md` describes the legs in detail, what a public release still
needs for comparability, and why bicubic-degraded benchmark sets are
out-of-distribution for a codec-trained model.

## Quickstart

```bash
# 1. Build the image. Package, configs and traiNNer are all baked in.
podman build -f ContainerFile -t rtus-train:dev .   # arm64/Jetson: -f ContainerFile.tegra

# 2. Check what the environment can actually do.
docker run --rm rtus-train:dev rtus-info

# 3. Fetch the two third-party assets this repo does not redistribute
#    (teacher for distillation targets, SCRFD detector for the face gate).
bash scripts/fetch_assets.sh ./pretrained

# 4. Build the training corpus from the public HR set.
docker run --rm -v "$PWD:/workspace" rtus-train:dev \
    python -m userdevice_rtus.tools.gen_pairs_v2

# 5. Train a config.
bash scripts/run-train.sh configs/2x_RTMoSREA_film_stageE_dists90.yml

# 6. Score every checkpoint and pick the best by DISTS.
#    Tools run as modules; only rtus-train and rtus-info are console commands.
docker run --rm -v "$PWD:/workspace" rtus-train:dev \
    python -m userdevice_rtus.tools.eval_compare \
      --sweep /workspace/experiments/<name>:rtmosr_ea_film

# 7. Export to ONNX (writes onnx_out/<name>-{540p,720p,1080p}.onnx + parity ref).
docker run --rm -v "$PWD:/workspace" rtus-train:dev \
    python -m userdevice_rtus.tools.export_student <ckpt> <name> rtmosr_ea_film
```

Notes:

- `ContainerFile` builds an x86_64 image; `ContainerFile.tegra` builds the
  arm64 Jetson image and **must be built on arm64 hardware**.
- `scripts/run-train.sh` reads `RTUS_DATA_ROOT`, `RTUS_IMAGE` and
  `RTUS_MEMORY` from the environment and applies a hard memory cap by
  default — dataloader workers dominate host RAM.
- **Without containers:** `uv pip install -e '.[train,teacher,export]'`, set
  `RTUS_DATA_ROOT` to wherever your data lives, and run `rtus-info` to see
  what is missing. The base install (no extras) is enough to build a model,
  load weights and score it; `train` pulls traiNNer-redux from git.
- **Kubernetes:** see `k8s/README.md`. The image is self-contained, so a Job
  supplies data and a config name and nothing else.

## Repository layout

    src/userdevice_rtus/      the package: architectures, CLI, and
      tools/                  the verdict harness and data generators
    configs/                  traiNNer-redux configs, per stage and tier
    ContainerFile[.tegra]     x86_64 image / arm64 Jetson image
    scripts/                  host-side helpers (build corpus, fetch assets, launch training)
    k8s/                      kustomize base + example overlay
    data/                     the data CONTRACT — never the data itself
    docs/                     provenance, benchmarks, naming

Training data, checkpoints, ONNX exports and TensorRT engines are all
gitignored. Weights are published separately as a Hugging Face model
repository, and a release cites the commit it was trained from.

## Provenance, licence and attribution

Code in this repository is **MIT** (see `LICENSE`). Third-party notices for
RTMoSR (MIT), neosr and traiNNer-redux (Apache-2.0) are in `LICENSE`.

The released models are **distilled from `4xNomosWebPhoto_RealPLKSR` by
Philip Hofmann**, licensed CC BY 4.0. That attribution is required by the
teacher's licence and carries to the weights, not just to this source tree.

`docs/PROVENANCE.md` is the full record — every component, its origin,
licence and evidence, the corpus lineage (the NomosRealWeb HRs are Nomos-v2,
itself distilled from 14 upstream datasets), the from-scratch weight lineage,
one recorded upstream discrepancy, and the obligations a release carries.
Read it before releasing anything.

Naming (`userdevice-rtus-<version>-<shape>`) is explained in `docs/NAMING.md`.
