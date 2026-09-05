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
