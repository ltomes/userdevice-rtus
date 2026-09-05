# Provenance and licensing

Every component of this model, with the evidence for its licence status.
**Unknown is recorded as unknown.** Nothing here is inferred; each row
cites where it was checked. A release must not be cut while any row in the
BLOCKING column is open.

## Code

| Component | Origin | Licence | Evidence | Blocking? |
|---|---|---|---|---|
| `traiNNer-redux` (training framework) | github.com/the-database/traiNNer-redux | **Apache 2.0** | `LICENSE.txt` in the upstream clone, verified 2026-09-05 | No — resolved. Consumed as a submodule, not vendored. |
| `realplksr_vendored.py` | Derived from `muslll/neosr` `neosr/archs/realplksr_arch.py`, with umzi2's dysample/layernorm modifications; itself descended from spandrel's PLKSR/RealPLKSR | **UNVERIFIED** | File header cites the source URL; upstream licence not yet read | **YES** |
| `rtmosr_vendored.py` | Unknown. No provenance header. | **UNKNOWN** | Checked 2026-09-05: the file carries no attribution comment | **YES — most serious open item** |
| `rtmosr_ea_vendored.py` | Ours. Original work: RTMoSR backbone + an EA gate grafted from RealPLKSR, derived from our own teacher-ablation results | Ours, but inherits from the two rows above | File docstring, dated 2026-08-20 | Follows the rows above |
| `scrfd_2.5g_bnkps.onnx` (face gate detector) | InsightFace SCRFD | **UNVERIFIED** | Third-party weights; fetched, not committed | Only if we redistribute it — we do not |

## Data

| Component | Origin | Licence | Evidence | Blocking? |
|---|---|---|---|---|
| HR training images | NomosRealWeb release — `hr.tar`, 6000 x 512^2 Nomos-v2 HRs | **UNVERIFIED** | `log.md:126` records the dataset as public; the specific licence text has not been read | **YES** — but only for *claims*, since we redistribute no images |
| LR training images | Generated locally from the above by our own scripts | Ours | `scripts/gen_pairs_h264.sh`, `scripts/gen_pairs_v2.py` | No |
| Teacher targets | Output of `4xNomosWebPhoto_RealPLKSR` over our LR | Follows the teacher row | `scripts/gen_teacher_targets.py` | Follows teacher |

**No private media is involved.** The corpus contains no film, television,
or personal library content. Verified 2026-09-05 against
`log.md:148`, `log.md:1751` ("this dataset is NomosRealWeb + synthetic
H.264 clips, NOT the user's film library") and `gen_pairs_v2.py:33`
(`SRC = /workspace/datasets/nomosrealweb/hr`). An earlier claim in this
project that the corpus was the operator's film library was wrong and has
been withdrawn.

## Teacher

| Component | Origin | Licence | Evidence | Blocking? |
|---|---|---|---|---|
| `4xNomosWebPhoto_RealPLKSR` | Phhofm | **UNVERIFIED** | Loaded via spandrel in `gen_teacher_targets.py`; licence text not yet read | **YES** |

The student is a distillation of this teacher. If the teacher's licence
restricts derivative or commercial use, that restriction propagates to the
student weights regardless of the fact that our architecture and training
code are our own. This must be settled before any weight release.

## Weight lineage

Training resumes from earlier stages rather than from scratch, so the
release inherits every ancestor's licence. The chain must be written out
in full here before release; it has not been, and that is itself an open
item. Known link: the dists90 configs initialise
`pretrain_network_g` from a stage-A `net_g_ema_150000.safetensors` of our
own training. Whether any ancestor of stage A began from third-party
weights is **not yet established** — an early smoke run is recorded as an
"RTMoSR-L fine-tune from released 2x weights" (`log.md`, 2026-08-16), which
if it is in this lineage would be a third-party origin.

## Standing rule

Do not write a licence, permission, or exception into this file that
cannot be quoted from its source. An inferred permission outlives the
session that guessed it.

---

## Weight lineage — RESOLVED 2026-09-05

The release line inherits **no third-party weights**. Traced by reading
`pretrain_network_g` in every config of the chain:

    2x_RTMoSREA_film_stageA.yml       pretrain_network_g: ~   <- from scratch
    2x_RTMoSREA_film_sd_stageA.yml    pretrain_network_g: ~   <- from scratch
      stageC / stageD1 / stageD2 / stageE_dists90
      sd_stageC_dists90(_ext)
        -> all initialise from our own stage-A net_g_ema_150000.safetensors

Corroborated in `log.md`: :1036 "from-scratch charbonnier", :574 "Both need
from-scratch stage-A", :1208 "(iter 10,000, from-scratch, no [pretrain])".

This closes what was recorded above as an open blocking item. The earlier
concern — an "RTMoSR-L fine-tune from released 2x weights" noted in the
2026-08-16 log — belongs to a **different and superseded line**
(`2x_RTMoSR_L_greyduck`), which did initialise from a third-party
`pretrained/2x_RTMoSR_L.pth`. Those configs have been removed from this
repository precisely so that lineage cannot be confused with the release
line or contaminate its provenance. If that line is ever revived, its
third-party weight origin must be resolved first.

**Still open** (see the tables above): the origin of `rtmosr_vendored.py`,
the licence of `4xNomosWebPhoto_RealPLKSR`, the licence of the NomosRealWeb
release, and the neosr licence behind `realplksr_vendored.py`. Those are
about *code and teacher*, not weights.

---

## Licence findings — checked against upstream sources 2026-09-05

| Component | Licence | Source |
|---|---|---|
| `traiNNer-redux` | **Apache-2.0** | `LICENSE.txt` in the local clone |
| RTMoSR (`rtmosr_vendored.py`) | **MIT** | github.com/rewaifu/RTMoSR — the upstream this file had no header for |
| neosr (behind `realplksr_vendored.py`) | **Apache-2.0** | github.com/neosr-project/neosr, sidebar |
| Teacher `4xNomosWebPhoto_RealPLKSR` | **CC-BY-4.0** | huggingface.co/Phips/4xNomosWebPhoto_RealPLKSR card front-matter (`license: cc-by-4.0`); OpenModelDB lists CC-BY-4.0 |
| Nomos-v2 / 4xNomosRealWeb dataset | **STILL UNKNOWN** | Neither the GitHub release page nor OpenModelDB states a dataset licence |

**Every code dependency is permissive** (MIT / Apache-2.0), and the teacher
is CC-BY-4.0 — attribution required, commercial use permitted, derivatives
permitted. A distilled student is therefore releasable, which was the single
largest open risk.

### Obligations this creates for the release

1. **Attribute Philip Hofmann** for `4xNomosWebPhoto_RealPLKSR` in the model
   card, as CC-BY-4.0 requires. The student is distilled from it; this is not
   optional and not satisfied by a passing mention in a footnote.
2. **Carry the MIT notice** for RTMoSR (rewaifu) with the architecture code.
3. **Carry Apache-2.0 NOTICE/attribution** for traiNNer-redux and neosr.
4. State the Nomos-v2 lineage even though we redistribute none of it.

### One discrepancy, recorded rather than smoothed over

The GitHub release page for the teacher states its licence as
**"CC-BY-0.4"**, which is not a real Creative Commons version. The author's
own Hugging Face card declares `cc-by-4.0` and OpenModelDB lists CC-BY-4.0.
Reading it as CC-BY-4.0 is the only coherent interpretation, but it is an
upstream typo and not something we can resolve unilaterally. If the release
is commercial, confirm with the author rather than relying on this note.

### Remaining unknown

The **dataset** licence (Nomos-v2, and Phhofm's 4xNomosRealWeb degraded LR
release). We redistribute no images, so this bounds what may be *claimed*
about the corpus, not what may be shipped. It should still be settled before
the model card asserts anything about the training data's licence.
