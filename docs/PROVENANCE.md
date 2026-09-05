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
| HR training images | NomosRealWeb release — `hr.tar`, 6000 x 512² Nomos-v2 HRs | **UNKNOWN** | No dataset licence is stated on the release page or on OpenModelDB. **The one open item.** |
| LR training images | Generated locally from the HRs by our own scripts | Ours | `scripts/gen_pairs_h264.sh`, `scripts/gen_pairs_v2.py` |
| Teacher targets | Output of the teacher over our LR | Follows the teacher | `scripts/gen_teacher_targets.py` |

We redistribute no images, so the open dataset licence bounds what a model
card may **claim** about the corpus, not what may be shipped. It should still
be settled before the card asserts anything about the training data's terms.

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
