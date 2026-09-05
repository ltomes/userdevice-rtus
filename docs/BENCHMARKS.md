# Benchmarks

## What already exists

The analysis chain in `userdevice_rtus.tools` is the project's existing verdict harness and
is stronger than what most published SR models ship. Every verdict ran
these legs on identical frames:

1. **Sweep** — `eval_compare`: PSNR, SSIM, DISTS, LPIPS across every
   checkpoint, against bicubic (floor) and the teacher (reference), picking
   best-by-DISTS rather than by last iteration. Metrics come from traiNNer's
   own registry implementations so they are directly comparable to the
   figures printed during training. `crop_border=2`, `test_y_channel=True`.
2. **Face gate** — `facegate.face_gate`, BLOCKING. Zero tolerance for
   face-region violations.
3. **Invention probe** — `hallucination_probe`: what fraction of
   high-frequency detail the student invents rather than recovers. The kill
   condition is inventing more than the teacher (8.50%).
4. **Temporal** — `temporal_eval`: flicker on adjacent frames, against a
   bicubic control. The val set holds adjacent frames of the same clip; real
   motion cancels between the two difference terms, leaving only invented
   temporal change.

Numbered to match the README, so "leg 3" means the same thing in both.

A known gotcha, measured: traiNNer-redux's built-in `Best: <v> @ <iter>`
line is unreliable for DISTS and LPIPS — it tracks the maximum, which is
backwards for metrics where lower is better. Always pick with the sweep.

## What is missing for a public release

Every internal number is computed on the project's own validation split.
That split is reproducible (the corpus is public — see `data/README.md`),
but it is **not comparable** to the SR literature. A public release needs:

### Comparability sets (research-use)

Set5, Set14, BSD100, Urban100, Manga109 at x2 — PSNR/SSIM on the **Y
channel**, which is the convention; RGB numbers cannot be placed beside
published figures. Plus DIV2K validation.

Note their licences differ and several are research-use rather than open:
**Manga109 requires an application and an academic agreement** and must not
be redistributed. Fetch from official sources; do not vendor.

### The in-distribution problem — read this before reporting a number

This student is trained against **codec degradation**: libx264, CRF 18–34,
GOP 24/48, real P/B-frame artifacts. The standard benchmark sets are
**bicubic-downsampled**. Bicubic degradation is therefore out of
distribution for this model, and its scores on those sets will understate
it relative to models trained on the bicubic assumption.

Report them anyway, for comparability — but state the mismatch plainly in
the model card rather than omitting the table. The headline evidence should
be a codec-degraded set the model was actually built for.

### A reproducible in-distribution set (recommended)

Build one from **CC-BY open film** — the Blender Foundation open movies
(Sintel, Tears of Steel, Cosmos Laundromat, Spring, Big Buck Bunny) are
CC-BY, available at high resolution, and freely redistributable. Degrade
them with the *same* recipe as training (`gen_pairs_h264.sh`), and the
result is a public, redistributable, in-distribution benchmark that also
carries **real temporal content** — which the current still-frame
validation split fundamentally cannot provide, and which this model's
temporal leg needs.

### Cost, which is the point of this model

Parameter count, FLOPs at 1080p->4K, fp16 latency and VRAM on a **named**
GPU. A real-time upscaler that does not report its latency has not reported
anything.

## Not verified in this document

The download URLs and checksums in `scripts/fetch_testsets.sh` have not
been executed or verified. Pin checksums on first successful fetch.
