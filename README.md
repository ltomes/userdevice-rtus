# userDevice FAMILY_NAME_PENDING — 2x upscaler, student/teacher training and evaluation

Training and evaluation code for a small, fast 2x super-resolution model
built to run in **real time on Thor-class hardware**: the design constraint
is a 38 ms frame budget at 24fps, and quality is spent only where it fits
inside that budget.

> **The family name is not chosen yet.** `userdevice` is the make. The
> literal token `FAMILY_NAME_PENDING` stands in for the family throughout
> this repo; `scripts/set-family-name.sh <name>` stamps the real one in one
> pass. See `docs/NAMING.md`.

## What this is

A distilled student. The teacher is `4xNomosWebPhoto_RealPLKSR` (Philip
Hofmann, CC-BY-4.0). The student is an RTMoSR-style small-kernel backbone
with a per-pixel EA gate grafted onto every block output — a design derived
from ablating the teacher, which found its quality lives in EA gating and
collective depth rather than in its 17px large kernels. So the student keeps
the fast small-kernel backbone and buys depth over width.

Two tiers, differing in width:

| Tier | Target input | Trade |
|---|---|---|
| d64 | <=720p | wider; more quality per frame |
| d48 | 1080p | narrower; fits the 1080p->4K budget |

## What it was NOT trained on

**No film, television, or private media.** The corpus is public photographs
(the NomosRealWeb release) with *synthesised* motion and *real* H.264
degradation. The `_film` in the config names denotes a latency tier, not
content. See `data/README.md` — this distinction constrains what a model
card may claim.

## Layout

    src/userdevice_upscale/   architectures (student, backbone, teacher-side)
    configs/                  traiNNer-redux configs, per stage and tier
    scripts/                  data generation, teacher targets, ONNX export
    eval/                     the four-leg verdict harness
    data/                     the data CONTRACT — never the data itself
    docs/                     provenance, benchmarks, naming

## Data and weights are never in this repository

Training data is regenerated from public sources by `scripts/`; checkpoints,
ONNX exports and engines are build outputs. Both are gitignored. Weights are
published separately as a Hugging Face model repository, and a release cites
the commit it was trained from.

## Before releasing anything

Read `docs/PROVENANCE.md`. Every code dependency is permissive (MIT /
Apache-2.0) and the teacher is CC-BY-4.0, so the model is releasable — but
attribution to Philip Hofmann is **required**, not optional, and the
Nomos-v2 dataset licence is still unestablished.
