# Provenance and licensing

Current state of every component this model is built from, with the evidence
for each. **Unknown is recorded as unknown.** Nothing here is inferred.

## Verdict

**The model is releasable.** Every code dependency is permissive, the teacher
permits derivatives and commercial use with attribution, and the release line
inherits no third-party weights. One item remains open, and it bounds what may
be *claimed* rather than what may be shipped.

## Code

| Component | Origin | Licence | Evidence |
|---|---|---|---|
| `traiNNer-redux` (training framework, metrics) | github.com/the-database/traiNNer-redux | **Apache-2.0** | `LICENSE.txt` in the upstream repository |
| `rtmosr_vendored.py` (backbone) | github.com/rewaifu/RTMoSR | **MIT** | upstream repository licence |
| `realplksr_vendored.py` | Derived from `neosr` (`neosr/archs/realplksr_arch.py`), with umzi2's dysample/layernorm modifications; descended from spandrel's PLKSR/RealPLKSR | **Apache-2.0** | github.com/neosr-project/neosr |
| `rtmosr_ea_vendored.py` (the student) | Ours. RTMoSR backbone plus an EA gate grafted from RealPLKSR, derived from our own teacher ablation | Ours, inheriting MIT and Apache-2.0 above | `userdevice_rtus/tools/ablate_teacher.py` is the ablation it came from |
| `scrfd_2.5g_bnkps.onnx` (face-gate detector) | InsightFace SCRFD | **UNVERIFIED** | Third-party weights, fetched rather than committed. Only matters if redistributed — we do not redistribute it. |

## Teacher

| Component | Origin | Licence | Evidence |
|---|---|---|---|
| `4xNomosWebPhoto_RealPLKSR` | Philip Hofmann (Phhofm) | **CC-BY-4.0** | The author's Hugging Face model card declares `license: cc-by-4.0`; OpenModelDB lists CC-BY-4.0 |

The student is a distillation of this teacher, so the teacher's terms
propagate to the student weights regardless of our architecture and training
code being our own. CC-BY-4.0 permits derivatives and commercial use and
**requires attribution**.

### One upstream discrepancy, recorded rather than smoothed over

The teacher's GitHub release page states its licence as **"CC-BY-0.4"**, which
is not a real Creative Commons version. The author's own Hugging Face card
declares `cc-by-4.0` and OpenModelDB lists CC-BY-4.0, so reading it as 4.0 is
the only coherent interpretation — but it is an upstream typo we cannot
resolve unilaterally. A commercial release should confirm with the author.

## Data

| Component | Origin | Licence | Evidence |
|---|---|---|---|
| HR training images | NomosRealWeb release — `hr.tar`, 6000 x 512² Nomos-v2 HRs | **A STACK — see below** | Nomos-v2 states no licence of its own, and is distilled from 14 upstream datasets, at least one of which (FFHQ) is CC BY-NC-SA 4.0. **The open item.** |
| LR training images | Generated locally from the HRs by our own scripts | Ours | `scripts/gen_pairs_h264.sh`, `scripts/gen_pairs_v2.py` |
| Teacher targets | Output of the teacher over our LR | Follows the teacher | `scripts/gen_teacher_targets.py` |

### The dataset licence is a stack, not a single unknown

Nomos-v2 is not an originally-captured dataset. Its own description states it
was distilled from **14 upstream datasets**: Adobe-MIT-5k, RAISE, LSDIR,
LIU4k-v2, KONIQ-10k, Nikon LL RAW, DIV8k, **FFHQ**, Flickr2k,
ModernAnimation1080_v2, Rawsamples, SignatureEdits, Hasselblad raw samples,
and Unsplash.

At least one of those carries terms that matter:

| Upstream | Licence | Consequence |
|---|---|---|
| **FFHQ** (NVIDIA) | **CC BY-NC-SA 4.0** — verified 2026-09-05 | **Non-commercial**, and share-alike |

Several others (DIV8K, Flickr2K, RAISE, Adobe-MIT-5k) are published for
academic research; their individual terms have **not** been read here and
should be before any commercial claim is made.

**What this does and does not mean.** We redistribute no images, so nothing
about this restricts what this repository ships. What is genuinely unsettled
is whether dataset licence terms propagate to the *weights* trained on them.
That is a contested legal question, jurisdiction-dependent, and it is not
resolved by anything in this document — do not read the presence of this
section as either permission or prohibition.

It matters here for two specific decisions, and both need a human answer:

1. **Commercial use of the released weights**, given a non-commercial
   component in the corpus lineage.
2. **The weights licence.** A share-alike component upstream sits more
   comfortably with a share-alike release than a permissive one.

The same lineage reaches us twice: our HR corpus is the NomosRealWeb release
of Nomos-v2 HRs, and the teacher we distil from was itself trained on
Nomos-v2. Retraining on a different corpus would not by itself clear the
teacher path.

**No private media is involved.** The corpus contains no film, television or
personal library content: the HR source is a public photograph dataset
(`scripts/gen_pairs_v2.py`, `SRC = .../nomosrealweb/hr`), motion is
synthesised by sub-pixel drift over stills, and the only real-video element is
the H.264 degradation stage. An earlier note in this project claiming the
corpus was a private film library was incorrect and has been withdrawn. See
`data/README.md`.

## Weight lineage — no third-party weights

The release line trains from scratch. Traced through `pretrain_network_g` in
every config of the chain:

    2x_RTMoSREA_film_stageA.yml       pretrain_network_g: ~   <- from scratch
    2x_RTMoSREA_film_sd_stageA.yml    pretrain_network_g: ~   <- from scratch
      stageC / stageD1 / stageD2 / stageE_dists90
      sd_stageC_dists90(_ext)
        -> all initialise from our own stage-A net_g_ema_150000.safetensors

A separate, superseded line (`2x_RTMoSR_L_greyduck`) did initialise from a
third-party pretrained checkpoint. Its configs were removed from this
repository so the two lineages cannot be confused. If that line is ever
revived, its weight origin must be resolved first.

## Obligations this creates for a release

1. **Attribute Philip Hofmann** for `4xNomosWebPhoto_RealPLKSR`, as CC-BY-4.0
   requires. The student is distilled from it. This is not optional and is not
   discharged by a footnote.
2. **Carry the MIT notice** for RTMoSR (rewaifu) with the architecture code.
3. **Carry Apache-2.0 attribution** for traiNNer-redux and neosr.
4. State the Nomos-v2 lineage of the corpus, even though no images ship.

## Standing rule

Do not write a licence, permission or exception into this file that cannot be
quoted from its source. An inferred permission outlives the person who guessed
it.

## Changelog

- **2026-09-05** — Checked every dependency against its upstream. Resolved
  traiNNer-redux (Apache-2.0), RTMoSR (MIT — the vendored file carried no
  attribution header, and this identified its upstream), neosr (Apache-2.0)
  and the teacher (CC-BY-4.0). Traced the weight lineage to from-scratch.
  Corrected the corpus claim. Dataset licence remains the single open item.
